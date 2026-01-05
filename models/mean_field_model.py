import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss


@MODELS.register()
class MeanFieldModel(nn.Module):
    def __init__(self, loss_dict_stage_1, loss_dict_stage_2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(p=0.5)

        k = 128
        d = 10
        self.k = k
        self.d = d

        self.fc2 = nn.Linear(128, d)

        # q(W) = N(M, diag(S)) over vec(W)
        self.M = nn.Parameter(torch.zeros(k, d))
        self.log_S = nn.Parameter(torch.full((k, d), -5.0))

        self.prior_var = nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=False)

        self.likelihood_cov = nn.Parameter(torch.eye(d, dtype=torch.float), requires_grad=False)

        self.losses_stage_1 = {n: build_loss(l) for n, l in loss_dict_stage_1.items()}
        self.losses_stage_2 = {n: build_loss(l) for n, l in loss_dict_stage_2.items()}

        self.register_buffer(
            "vi_mod_on",
            torch.tensor(False, dtype=torch.bool)
        )

    @torch.no_grad()
    def forward_to_acquire(self, x):
        assert not self.training
        feats = self.forward_train(x, return_feats=True)
        predictive_mean, epistemic, aleatoric = self.get_explicit_predictive_params(feats)
        predictive_var = aleatoric + epistemic
        return predictive_mean, (predictive_var, epistemic, aleatoric)

    def get_predictive_params(self, f):
        assert bool(self.vi_mod_on)

        predictive_mean = f @ self.M  # (bs, d)

        S = torch.exp(self.log_S)  # (k, d)
        epistemic_diag = (f * f) @ S  # (bs, d)

        bs = f.size(0)
        epistemic = torch.diag_embed(epistemic_diag)  # (bs, d, d)

        lik_cov = self.likelihood_cov  # (d, d)
        aleatoric = lik_cov[None, :, :].expand(bs, -1, -1)

        predictive_var = aleatoric + epistemic
        return predictive_mean, predictive_var

    def get_explicit_predictive_params(self, f):
        assert bool(self.vi_mod_on)

        predictive_mean = f @ self.M  # (bs, d)

        S = torch.exp(self.log_S)  # (k, d)
        epistemic_diag = (f * f) @ S  # (bs, d)

        bs = f.size(0)
        epistemic = torch.diag_embed(epistemic_diag)  # (bs, d, d)

        lik_cov = self.likelihood_cov  # (d, d)
        aleatoric = lik_cov[None, :, :].expand(bs, -1, -1)

        return predictive_mean, epistemic, aleatoric

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
            return preds

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

    def forward_train(self, x, y=None, return_feats=False, return_loss=False):
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
            if name == "MeanFieldVI":
                assert bool(self.vi_mod_on)
                feats = feats_or_preds

                lik_cov = self.likelihood_cov
                Sigma2 = torch.diag(lik_cov).mean()  # BECAUSE WE SET IT ISOTROPIC

                loss_val, info = loss_fn(
                    y=y,
                    f=feats,
                    M=self.M,
                    log_S=self.log_S,
                    Sigma2=Sigma2,
                    prior_var=self.prior_var,
                )
                loss_dict[name] = {"loss": loss_val, "info": info}

            elif name == "MSE":
                assert not bool(self.vi_mod_on)
                preds = feats_or_preds
                loss_dict[name] = loss_fn(preds, y)

            elif name == "WeightDecay":
                assert not bool(self.vi_mod_on)
                loss_dict[name] = loss_fn(self)

            else:
                assert False

        return loss_dict

    @torch.no_grad()
    def estimate_likelihood_cov(self, all_x, all_y):
        preds = self.forward_train(all_x)
        residuals = all_y - preds
        Sigma_hat = (residuals.T @ residuals) / residuals.size(0)
        Sigma, lam = make_spd_by_shrinkage(Sigma_hat)

        # calculate mean trace:
        # mean trace = average variance per dimension
        d = Sigma.size(0)
        mean_variance = torch.trace(Sigma) / d

        return mean_variance, lam

    def open_vi_mode(self, all_x, all_y):
        was_training = self.training
        self.eval()

        with torch.no_grad():
            mean_variance, lam = self.estimate_likelihood_cov(all_x, all_y)

        print("When estimating the likelihood covariance, final lambda used:", lam)

        lik_data = self.likelihood_cov.data
        self.likelihood_cov.data.copy_(mean_variance * lik_data)

        if was_training:
            self.train()

        for p in self.parameters():
            p.requires_grad_(False)

        self.M.requires_grad_(True)
        self.log_S.requires_grad_(True)

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
