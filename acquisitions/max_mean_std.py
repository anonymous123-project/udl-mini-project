import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class MaxMeanStandardDeviation:
    def __call__(self, mean_probs, all_probs):
        # mean_probs: tensor of (bs,10)
        # all_probs: tensor of (T,bs,10) --> if T = 1,
        #   it means it is deterministic so no variance in all_probs
        assert isinstance(mean_probs, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert isinstance(all_probs, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert mean_probs.ndim == 2, f"mean_probs must be 2D (bs, C=10), got {mean_probs.shape}"
        assert all_probs.ndim == 3, f"all_probs must be 3D (T, bs, C=10), got {all_probs.shape}"

        if all_probs.size(0) == 1:
            # otherwise (if we set all to 0), topk would deterministic pick the first elements
            return self.assign_random_scores(mean_probs)

        e_p2 = (all_probs ** 2).mean(0) # (bs, 10)
        e2_p = all_probs.mean(0) ** 2   # (bs, 10)
        var_c = (e_p2 - e2_p).clamp_min(0.0)
        std_c = torch.sqrt(var_c)   # (bs, 10)
        return std_c.mean(-1)

    def assign_random_scores(self, mean_probs):
        bs = mean_probs.size(0)
        return torch.rand(bs, device=mean_probs.device)



