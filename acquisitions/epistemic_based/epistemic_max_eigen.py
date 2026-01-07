import torch

from registry.registry import ACQUISITION_FUNCTIONS


# THE MOST SIMILAR ONE TO PREDICTIVE ENTROPY
@ACQUISITION_FUNCTIONS.register()
class EpistemicMaxEigen:
    def __call__(self, mean_preds, predict_var):
        assert isinstance(predict_var, tuple)

        predictive_var, epistemic, aleatoric = predict_var
        assert torch.is_tensor(epistemic)

        assert epistemic.ndim == 3 and epistemic.size(-1) == epistemic.size(-2), \
            f"epistemic must be (bs, d, d), got {tuple(epistemic.shape)}"

        eigvals = torch.linalg.eigvalsh(epistemic)  # (bs, d)
        scores = eigvals[:, -1]  # max eig_val per sample

        return scores
