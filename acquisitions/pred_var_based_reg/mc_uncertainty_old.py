import torch

from .uncertainty_over_cov.logdet import Logdet
from .uncertainty_over_cov.trace import Trace
from .uncertainty_over_cov.max_eigval import MaxEigVal

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class MCUncertainty:
    def __init__(self, uncert_method=None):
        assert uncert_method is not None
        if uncert_method=="Logdet":
            self.uncert_method = Logdet()
        elif uncert_method=="Trace":
            self.uncert_method = Trace()
        elif uncert_method=="MaxEigVal":
            self.uncert_method = MaxEigVal()
        else:
            assert False

    def __call__(self, mean_preds, all_preds):
        # mean_probs: tensor of (bs,10)
        # all_probs: tensor of (T,bs,10) --> if T = 1, then it means it is deterministic, no BALD!
        #   in that case, BALD assigns random probability
        assert isinstance(mean_preds, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert isinstance(all_preds, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert mean_preds.ndim == 2, f"mean_probs must be 2D (bs, C=10), got {mean_preds.shape}"
        assert all_preds.ndim == 3, f"all_probs must be 3D (T, bs, C=10), got {all_preds.shape}"

        if all_preds.size(0) == 1:
            assert False

        deviation = all_preds - mean_preds.unsqueeze(0)  # (T, bs, d)
        cov = (deviation.unsqueeze(-1) @ deviation.unsqueeze(-2)).mean(0)  # (bs, d, d)
        self.check_rank_deff(cov)
        return self.uncert_method(cov)

    def check_rank_deff(self, x):
        assert x.ndim == 3, f"Expected (bs,d,d), got {x.shape}"
        assert x.shape[-1] == x.shape[-2], "Last two dims must be equal"

        eigvals = torch.linalg.eigvalsh(x)  # (bs, d)
        thr = 1e-8

        rank = (eigvals > thr).sum(dim=-1)  # (bs,)
        d = x.size(-1)

        assert torch.all(rank == d), "Rank-deficient covariance detected"


