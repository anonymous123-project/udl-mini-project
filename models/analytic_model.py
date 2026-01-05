import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss


@MODELS.register()
class MatrixNormalAnalyticModel(nn.Module):
    def __init__(self, loss_dict):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.25)

        # 28x28 -> conv4 -> 25x25 -> conv4 -> 22x22 -> pool2 -> 11x11
        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(p=0.5)

        # for pretraining
        self.fc2 = nn.Linear(128, 10)

        k = 128
        d = 10
        self.k = 128
        self.d = 10

        # for analytical inference
        # q(w) ~ Matrix N(M, U, V)
        self.M = nn.Parameter(torch.zeros(k, d, dtype=torch.float), requires_grad=False)  # (k, d)
        self.U = nn.Parameter(torch.eye(k, dtype=torch.float), requires_grad=False)  # (k, k)
        self.V = nn.Parameter(torch.eye(d, dtype=torch.float), requires_grad=False)  # (d, d)

        # likelihood
        self.likelihood_cov = nn.Parameter(torch.eye(d, dtype=torch.float),requires_grad=False)  # (d, d)

        # prior params
        # ~ Matrix N(0, I_k, V_0=Likelihood_Cov)
        self.V0 = nn.Parameter(torch.eye(d, dtype=torch.float), requires_grad=False)  # (d, d)

        self.losses = {n: build_loss(l) for n, l in loss_dict.items()}

        self.register_buffer(
            "analytical_params_found",
            torch.tensor(False, dtype=torch.bool)
        )

    @torch.no_grad()
    def estimate_likelihood_cov(self, all_x, all_y):
        preds = self.forward_train(all_x)
        residuals = all_y - preds
        Sigma_hat = (residuals.T @ residuals) / residuals.size(0)

        Sigma, lam = make_spd_by_shrinkage(Sigma_hat)

        return Sigma, lam

    @torch.no_grad()
    def find_analytical_params(self, all_x, all_y):
        assert not self.training
        assert not bool(self.analytical_params_found), "Analytical params already found!"

        # get feats
        X = self.forward_train(all_x, return_feats=True)
        Y = all_y

        # find U
        U0 = torch.eye(self.k, dtype=X.dtype, device=X.device)
        xTx = X.T @ X
        U_inv = U0 + xTx
        U = torch.linalg.inv(U_inv)

        # find M
        M = U @ X.T @ Y

        # find likelihood cov
        lik_cov, lam = self.estimate_likelihood_cov(all_x, all_y)
        print("When estimating the likelihood covariance, final lambda used:", lam)
        V0 = lik_cov

        self.M.data.copy_(M)
        self.U.data.copy_(U)
        self.likelihood_cov.data.copy_(lik_cov)
        self.V0.data.copy_(V0)
        self.V.data.copy_(V0)

        self.analytical_params_found.fill_(True)

    @torch.no_grad()
    def forward_to_acquire(self, x):
        assert not self.training
        feats = self.forward_train(x, return_feats=True)
        predictive_mean, epistemic, aleatoric = self.get_explicit_predictive_params(feats)
        predictive_var = aleatoric + epistemic
        return predictive_mean, (predictive_var, epistemic, aleatoric)

    def get_predictive_params(self, f):
        assert bool(self.analytical_params_found)

        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov  # d, d
        fU = f @ self.U  # bs, k
        fUf = (fU * f).sum(1)  # bs,
        fUfV = fUf[:, None, None] * self.V[None, :, :]  # bs, d, d

        predictive_var = lik_cov[None, :, :] + fUfV  # likelihood uncertainty + posterior uncertainty

        return predictive_mean, predictive_var  # (bs, d), (bs, d, d)

    def get_explicit_predictive_params(self, f):
        assert bool(self.analytical_params_found)

        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov  # d, d
        fU = f @ self.U  # bs, k
        fUf = (fU * f).sum(1)  # bs,
        fUfV = fUf[:, None, None] * self.V[None, :, :]  # bs, d, d

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

    # if return loss is set, only the loss is returned, not the outputs
    # return probs and return loss cannot be set both
    def forward_train(self, x, y=None, return_feats=False, return_loss=False):
        if self.training:
            assert return_loss
        assert not (return_feats and return_loss)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        feats = self.drop2(x)

        if return_feats:
            return feats

        if bool(self.analytical_params_found):
            preds, _ = self.get_predictive_params(feats)
        else:
            preds = self.fc2(feats)

        if return_loss:
            return self.get_loss_dict(preds, y)

        return preds

    def get_loss_dict(self, preds, y):
        assert y is not None
        assert y.ndim == 2 and y.size(0) == preds.size(0) and y.size(1) == 10

        loss_dict = {}
        for name, loss_fn in self.losses.items():
            loss_dict[name] = loss_fn(self) if name == "WeightDecay" else loss_fn(preds, y)
        return loss_dict


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
