import torch.nn as nn
import torch
import math

from registry.registry import LOSSES


@LOSSES.register()
class MatrixNormalVILoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,
                y,  # (bs, d)
                f,  # (bs, k)  == f(x)
                M,  # (k, d)
                U,  # (k, k)
                V,  # (d, d)
                Sigma_inv,  # (d, d)
                logdet_Sigma,  # scalar  (log|Sigma|)
                logdet_U,  # scalar
                logdet_V,  # scalar
                V0_inv,  # (d, d)
                logdet_V0 # scalar
                ):
        bs, d = y.shape
        _, k = f.shape
        assert M.shape == (k, d)

        # ------- E_q[log p(Y|X,W)] -------
        mean = f @ M  # (bs, d)
        residuals = y - mean  # (bs, d)

        # SUM TERMS
        # term1: sum_n (y_n - mean_n)^T Sigma^(-1) (y_n - mean_n)
        temp1 = residuals @ Sigma_inv
        sum_quad_res = (temp1 * residuals).sum()  # scalar

        # term2: tr(V Sigma^(-1)) * sum_n (f_n^T U f_n)
        trace_V_Sigma_inv = torch.trace(V @ Sigma_inv)  # scalar
        temp2 = f @ U
        sum_fn_U_fn = (temp2 * f).sum()  # scalar
        term2 = trace_V_Sigma_inv * sum_fn_U_fn

        sum_term = -0.5 * (sum_quad_res + term2)

        # log(|Sigma|) term:
        logdet_sigma_term = bs * -0.5 * logdet_Sigma

        # constant, kept anyway
        const = -0.5 * bs * d * math.log(2.0 * math.pi)

        expectation_term = logdet_sigma_term + sum_term  # + const

        # ------- KL(q||p) -------
        kl_term = 0.5 * (
                (k * logdet_V0) - (k * logdet_V) - (d * logdet_U) - k*d +
                torch.trace(U) * torch.trace(V0_inv @ V) +
                torch.trace(M @ V0_inv @ M.T)
        )

        mean_expectation_term = expectation_term / bs
        mean_kl_term = kl_term / bs

        mean_elbo = mean_expectation_term - mean_kl_term
        loss = -mean_elbo

        info_dict = {
            "expectation_loss_term": (-(expectation_term + const) / bs).detach().item(),
            "kl_loss_term": mean_kl_term.detach().item(),
            "elbo": ((expectation_term + const) / bs).detach().item() - mean_kl_term.detach().item(),
            "valid_elbo": mean_elbo.detach().item(),
            "valid_expectation_loss_term": (-mean_expectation_term).detach().item(),
        }

        return loss, info_dict
