import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class MaxVariationRatios:
    def __call__(self, mean_probs, all_probs):
        # mean_probs: tensor of (bs,10)
        # all_probs: tensor of (T,bs,10) --> not used, for signature match
        assert isinstance(mean_probs, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert mean_probs.ndim == 2, f"mean_probs must be 2D (bs, C=10), got {mean_probs.shape}"

        max_p, _ = mean_probs.max(dim=1)  # (bs,)
        return 1.0 - max_p  # (bs,)
