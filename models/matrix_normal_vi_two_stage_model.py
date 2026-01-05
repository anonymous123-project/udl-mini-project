import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss

from .spd_matrix import LearnableSPDMatrix


@MODELS.register()
class MatrixNormal2StageModel(nn.Module):
    def __init__(self, loss_dict_stage_1, loss_dict_stage_2, prior_column_cov_eq_lik_cov=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.25)

        # 28x28 -> conv4 -> 25x25 -> conv4 -> 22x22 -> pool2 -> 11x11
        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(p=0.5)

        k = 128
        d = 10
        self.k = 128
        self.d = 10

        # for pretraining
        self.fc2 = nn.Linear(128, 10)

        # create posterior parameters:
        # q(w) ~ Matrix N(M, U, V)
        self.M = nn.Parameter(torch.zeros(k, d))
        self.U = LearnableSPDMatrix(128, init_identity=True)
        self.V = LearnableSPDMatrix(10, init_identity=True)

        # likelihood
        # not learned
        self.likelihood_cov = nn.Parameter(torch.eye(d, dtype=torch.float), requires_grad=False)  # (d, d)

        # prior params
        # ~ Matrix N(0, I_k, V_0)
        # if the following is set to true, then we set V_0 = self.likelihood_cov.detach() during optimization
        self.prior_column_cov_eq_lik_cov = prior_column_cov_eq_lik_cov

        self.losses_stage_1 = {n: build_loss(l) for n, l in loss_dict_stage_1.items()}
        self.losses_stage_2 = {n: build_loss(l) for n, l in loss_dict_stage_2.items()}

        self.register_buffer(
            "vi_mod_on",
            torch.tensor(False, dtype=torch.bool)
        )

        self.register_buffer("Sigma_inv", torch.empty(0))
        self.register_buffer("logdet_Sigma", torch.empty(0))

    # returns the preds, as well as the predictive variance
    @torch.no_grad()
    def forward_to_acquire(self, x):
        assert not self.training
        feats = self.forward_train(x, return_feats=True)
        predictive_mean, epistemic, aleatoric = self.get_explicit_predictive_params(feats)
        predictive_var = aleatoric + epistemic
        return predictive_mean, (predictive_var, epistemic, aleatoric)

    def get_predictive_params(self, f):
        assert bool(self.vi_mod_on)
        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov  # d, d
        fU = f @ self.U.get()  # bs, k
        fUf = (fU * f).sum(1)  # bs,
        fUfV = fUf[:, None, None] * self.V.get()[None, :, :]  # bs, d, d

        predictive_var = lik_cov[None, :, :] + fUfV  # likelihood uncertainty + posterior uncertainty

        return predictive_mean, predictive_var  # (bs, d), (bs, d, d)

    def get_explicit_predictive_params(self, f):
        assert bool(self.vi_mod_on)
        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov  # d, d
        fU = f @ self.U.get()  # bs, k
        fUf = (fU * f).sum(1)  # bs,
        fUfV = fUf[:, None, None] * self.V.get()[None, :, :]  # bs, d, d

        aleatoric = lik_cov[None, :, :].expand(f.size(0), -1, -1)
        epistemic = fUfV

        return predictive_mean, epistemic, aleatoric  # (bs, d), (bs, d, d), (bs, d, d)

    @torch.no_grad()
    def forward_test(self, x, y=None, return_metrics=False):
        assert not self.training
        preds = self.forward_train(x)

        if return_metrics:
            assert y is not None
            mse = F.mse_loss(preds, y, reduction="mean")
            rmse = torch.sqrt(mse)
            return preds, {"RMSE": rmse}
        else:
            return preds  # bs, 10

    @torch.no_grad()
    def get_preds_and_metrics_w_feats(self, feats, y):
        assert not self.training
        assert y is not None
        assert y.ndim == 2 and y.size(0) == feats.size(0) and y.size(1) == 10
        assert bool(self.vi_mod_on)

        preds, _ = self.get_predictive_params(feats)
        mse = F.mse_loss(preds, y, reduction="mean")
        rmse = torch.sqrt(mse)
        return preds, {"RMSE": rmse}

    # if return loss is set, only the loss is returned, not the outputs
    # return probs and return loss cannot be set both
    def forward_train(self, x, y=None, return_feats=False, return_loss=False):
        # expects y has shape (bs, 10) of type float
        if self.training:
            assert return_loss
        assert not (return_feats and return_loss)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if not bool(self.vi_mod_on):
            x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if not bool(self.vi_mod_on):
            feats = self.drop2(x)  # bs, k=128
        else:
            feats = x

        if return_feats:
            return feats

        if bool(self.vi_mod_on):
            if self.training:
                return self.get_loss_dict(feats, y)
            else:
                preds, _ = self.get_predictive_params(feats)
                return preds
        else:
            preds = self.fc2(feats)
            if self.training:
                return self.get_loss_dict(preds, y)
            else:
                return preds

    def get_loss_dict(self, feats_or_preds, y):
        assert y is not None
        assert y.ndim == 2 and y.size(0) == feats_or_preds.size(0) and y.size(1) == 10

        loss_dict = {}
        losses = self.losses_stage_2 if bool(self.vi_mod_on) else self.losses_stage_1
        for name, loss_fn in losses.items():
            if name == "MatrixNormalVI":
                assert bool(self.vi_mod_on)
                feats = feats_or_preds

                Sigma_inv = self.Sigma_inv # likelihood cov inv
                logdet_Sigma = self.logdet_Sigma
                if self.prior_column_cov_eq_lik_cov:
                    v0_inv = Sigma_inv.detach()
                    logdet_v0 = logdet_Sigma.detach()
                else:
                    device = Sigma_inv.device
                    dtype = Sigma_inv.dtype
                    v0_inv = torch.eye(y.size(1), device=device, dtype=dtype)
                    logdet_v0 = torch.tensor(0.0, device=device, dtype=dtype)

                loss_val, info = loss_fn(y=y, f=feats,
                                         M=self.M, U=self.U.get(), V=self.V.get(),
                                         Sigma_inv=Sigma_inv, logdet_Sigma=logdet_Sigma,
                                         logdet_U=self.U.get_logdet(), logdet_V=self.V.get_logdet(),
                                         V0_inv=v0_inv, logdet_V0=logdet_v0)
                loss_dict[name] = {"loss": loss_val, "info": info}
            elif name == "MSE":
                assert not bool(self.vi_mod_on)
                preds = feats_or_preds
                loss_dict[name] = loss_fn(preds, y)
            elif name == "WeightDecay":
                assert not bool(self.vi_mod_on)
                loss_dict[name] = loss_fn(self)
            else:
                assert False, "No other loss should be present"

        return loss_dict

    @torch.no_grad()
    def estimate_likelihood_cov(self, all_x, all_y):
        preds = self.forward_train(all_x)
        residuals = all_y - preds
        Sigma_hat = (residuals.T @ residuals) / residuals.size(0)

        # add inherent noise: std = 0.1
        # d = Sigma_hat.size(0)
        # I = torch.eye(d, device=Sigma_hat.device, dtype=Sigma_hat.dtype)
        # Sigma_hat = Sigma_hat + (0.1 ** 2) * I

        Sigma, lam = make_spd_by_shrinkage(Sigma_hat)

        return Sigma, lam

    # all x and y required to finalize likelihood cov
    def open_vi_mode(self, all_x, all_y):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            lik_cov, lam = self.estimate_likelihood_cov(all_x, all_y)
        print("When estimating the likelihood covariance, final lambda used:", lam)
        self.likelihood_cov.data.copy_(lik_cov)

        Sigma_inv = torch.linalg.inv(self.likelihood_cov)
        sign, logdet_Sigma = torch.linalg.slogdet(self.likelihood_cov)
        assert sign > 0
        self.Sigma_inv.resize_as_(Sigma_inv).copy_(Sigma_inv)
        self.logdet_Sigma.resize_as_(logdet_Sigma).copy_(logdet_Sigma)

        if was_training:
            self.train()

        # freeze all
        for p in self.parameters():
            p.requires_grad_(False)

        # unfreeze posterior params
        self.M.requires_grad_(True)
        for p in self.U.parameters():
            p.requires_grad_(True)
        for p in self.V.parameters():
            p.requires_grad_(True)

        self.vi_mod_on.fill_(True)


# convert a given correlation matrix to SPD
@torch.no_grad()
def make_spd_by_shrinkage(Sigma_hat):
    # Sigma_hat: (d, d)
    d = Sigma_hat.size(0)
    device = Sigma_hat.device
    dtype = Sigma_hat.dtype

    # enforce symmetry
    assert torch.allclose(
        Sigma_hat, Sigma_hat.T, atol=1e-6
    ), "Sigma_hat is not symmetric; covariance estimation is broken"

    I = torch.eye(d, device=device, dtype=dtype)
    alpha = torch.trace(Sigma_hat) / d

    lam = 0.0
    bump = 2.0 ** (-10)

    while True:
        if lam == 0.0:
            Sigma = Sigma_hat
        else:
            Sigma = (1.0 - lam) * Sigma_hat + lam * alpha * I

        try:
            torch.linalg.cholesky(Sigma)
            return Sigma, lam
        except RuntimeError:
            if lam == 0.0:
                lam = bump
            else:
                lam = lam * 2.0
