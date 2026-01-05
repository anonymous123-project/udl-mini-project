
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss

from .spd_matrix import LearnableSPDMatrix


@MODELS.register()
class MatrixNormalModel(nn.Module):
    def __init__(self, loss_dict, prior_column_cov_eq_lik_cov=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.25)

        # 28x28 -> conv4 -> 25x25 -> conv4 -> 22x22 -> pool2 -> 11x11
        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(p=0.5)

        # create posterior parameters:
        # q(w) ~ Matrix N(M, U, V)
        self.M = nn.Parameter(torch.randn(128, 10))
        self.U = LearnableSPDMatrix(128, init_identity=True)
        self.V = LearnableSPDMatrix(10, init_identity=True)

        # likelihood
        # homoscedastic learnable noise
        self.likelihood_cov = LearnableSPDMatrix(10, init_identity=True)

        # prior params
        # ~ Matrix N(0, I_k, V_0)
        # if the following is set to true, then we set V_0 = self.likelihood_cov.detach() during optimization
        self.prior_column_cov_eq_lik_cov = prior_column_cov_eq_lik_cov

        self.losses = {n: build_loss(l) for n, l in loss_dict.items()}

    # returns the preds, as well as the predictive variance
    @torch.no_grad()
    def forward_to_acquire(self, x):
        assert not self.training
        feats = self.forward_train(x, return_feats=True)
        predictive_mean, epistemic, aleatoric = self.get_explicit_predictive_params(feats)
        predictive_var = aleatoric + epistemic
        return predictive_mean, (predictive_var, epistemic, aleatoric)

    def get_predictive_params(self, f):
        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov.get()  # d, d
        fU = f @ self.U.get()  # bs, k
        fUf = (fU * f).sum(1)  # bs,
        fUfV = fUf[:, None, None] * self.V.get()[None, :, :]  # bs, d, d

        predictive_var = lik_cov[None, :, :] + fUfV  # likelihood uncertainty + posterior uncertainty

        return predictive_mean, predictive_var  # (bs, d), (bs, d, d)

    def get_explicit_predictive_params(self, f):
        # f: features extracted from the rest of the network of shape (bs, k = 128)
        predictive_mean = f @ self.M  # bs, d
        lik_cov = self.likelihood_cov.get()  # d, d
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
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        feats = self.drop2(x)  # bs, k=128

        if return_loss:
            return self.get_loss_dict(feats, y)

        if return_feats:
            return feats

        preds, _ = self.get_predictive_params(feats)
        return preds  # bs, 10

    def get_loss_dict(self, feats, y):
        assert y is not None
        assert y.ndim == 2 and y.size(0) == feats.size(0) and y.size(1) == 10

        loss_dict = {}
        for name, loss_fn in self.losses.items():
            if name == "MatrixNormalVI":
                lik_cov = self.likelihood_cov

                Sigma_inv = lik_cov.get_inv()
                logdet_Sigma = lik_cov.get_logdet()
                if self.prior_column_cov_eq_lik_cov:
                    v0_inv = Sigma_inv.detach()
                    logdet_v0 = logdet_Sigma.detach()
                else:
                    device = lik_cov.raw_L.device
                    dtype = lik_cov.raw_L.dtype
                    v0_inv = torch.eye(y.size(1), device=device, dtype=dtype)
                    logdet_v0 = torch.tensor(0.0, device=device, dtype=dtype)

                loss_val, info = loss_fn(y=y, f=feats,
                        M=self.M, U=self.U.get(), V=self.V.get(),
                        Sigma_inv=Sigma_inv, logdet_Sigma=logdet_Sigma,
                        logdet_U=self.U.get_logdet(), logdet_V=self.V.get_logdet(),
                        V0_inv=v0_inv, logdet_V0=logdet_v0)

                loss_dict[name] = {"loss": loss_val, "info": info}
            else:
                assert False, "No other loss should be present"

        return loss_dict
