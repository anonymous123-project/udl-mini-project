import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class BALD:
    def __call__(self, mean_probs, all_probs):
        # mean_probs: tensor of (bs,10)
        # all_probs: tensor of (T,bs,10) --> if T = 1, then it means it is deterministic, no BALD!
        #   in that case, BALD assigns random probability
        assert isinstance(mean_probs, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert isinstance(all_probs, torch.Tensor), "mean_probs must be a torch.Tensor"
        assert mean_probs.ndim == 2, f"mean_probs must be 2D (bs, C=10), got {mean_probs.shape}"
        assert all_probs.ndim == 3, f"all_probs must be 3D (T, bs, C=10), got {all_probs.shape}"

        if all_probs.size(0) == 1:
            # otherwise (if we set all to 0), topk would deterministic pick the first elements
            return self.assign_random_scores(mean_probs)

        all_entropies = self.get_entropy(all_probs)
        all_entropies = all_entropies.permute(1, 0).mean(1)  # (bs, )

        pred_entropies = self.get_entropy(mean_probs)  # (bs, )

        return pred_entropies - all_entropies # (bs, )

    # calculate entropy over the last dimension
    def get_entropy(self, probs):
        p = probs.clamp(min=1e-8)  # avoids log(0)
        return -(p * p.log()).sum(dim=-1)  # (bs,) or (T, bs,)

    def assign_random_scores(self, mean_probs):
        bs = mean_probs.size(0)
        return torch.rand(bs, device=mean_probs.device)


