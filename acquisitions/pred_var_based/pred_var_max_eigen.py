import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class PredictiveVarMaxEigen:
    def __call__(self, mean_preds, predict_var):
        predictive_var = predict_var
        aleatoric = None  # not used
        epistemic = None  # not used
        if isinstance(predict_var, tuple):
            predictive_var, epistemic, aleatoric = predict_var

        assert torch.is_tensor(predictive_var)
        assert predictive_var.ndim == 3 and predictive_var.size(-1) == predictive_var.size(-2), \
            f"predictive_var must be (bs, d, d), got {tuple(predictive_var.shape)}"

        eigvals = torch.linalg.eigvalsh(predictive_var)  # (bs, d)
        scores = eigvals[:, -1]  # max eig_val per sample

        return scores
