import math
import torch
import torch.nn as nn

from registry.registry import LOSSES


@LOSSES.register()
class MeanFieldVILoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        y,              # (bs, d)
        f,              # (bs, k)
        M,              # (k, d)
        log_S,          # (k, d)
        Sigma2,         # scalar (likelihood var)
        prior_var       # scalar
    ):
        assert torch.is_tensor(Sigma2)
        assert Sigma2.ndim == 0

        assert torch.is_tensor(prior_var)
        assert prior_var.ndim == 0

        bs, d = y.shape
        _, k = f.shape
        assert M.shape == (k, d)

        mean = f @ M
        residuals = y - mean  # bs, d

        sum_term_1 = (residuals * residuals).sum()  # scalar

        S = torch.exp(log_S) # k, d
        S_row_sums = torch.sum(S, dim=-1)  # k
        diag_S_row_sum = torch.diag(S_row_sums)  # k, k

        temp1 = f @ diag_S_row_sum  # bs, k
        sum_term_2 = (temp1 * f).sum()

        sum_term = -0.5 * (sum_term_1 + sum_term_2) / Sigma2

        constant_term = -bs*d*0.5*torch.log(2*math.pi*Sigma2)

        expectation_term = constant_term + sum_term

        # KL
        constant = -k*d*0.5
        log_term = (torch.log(prior_var) - log_S).sum() * 0.5
        MnS_term = 0.5 * (S + M*M).sum() / prior_var
        kl_term = constant + log_term + MnS_term
        mean_expectation_term = expectation_term / bs
        mean_kl_term = kl_term / bs

        mean_elbo = mean_expectation_term - mean_kl_term
        loss = -mean_elbo

        info_dict = {
            "expectation_loss_term": (-mean_expectation_term).detach().item(),
            "kl_loss_term": mean_kl_term.detach().item(),
            "elbo": mean_elbo.detach().item(),
        }

        return loss, info_dict
