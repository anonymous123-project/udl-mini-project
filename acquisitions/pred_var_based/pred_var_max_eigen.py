import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class PredictiveVarMaxEigen:
    def __call__(self, mean_preds, predict_var):
        predictive_var = predict_var
        if isinstance(predict_var, tuple):
            predictive_var, _, _ = predict_var

        assert torch.is_tensor(predictive_var)
        assert predictive_var.ndim == 3 and predictive_var.size(-1) == predictive_var.size(-2), \
            f"predictive_var must be (bs, d, d), got {tuple(predictive_var.shape)}"

        d = predictive_var.size(-1)
        I = torch.eye(d, device=predictive_var.device, dtype=predictive_var.dtype)[None]

        eigvals = torch.linalg.eigvalsh(predictive_var + 1e-6 * I)  # (bs, d)
        return eigvals[:, -1]
