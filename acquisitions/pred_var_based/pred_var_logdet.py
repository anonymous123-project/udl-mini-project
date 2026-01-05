import torch

from registry.registry import ACQUISITION_FUNCTIONS


@ACQUISITION_FUNCTIONS.register()
class PredictiveVarLogDet:
    def __call__(self, mean_preds, predict_var):
        predictive_var = predict_var
        aleatoric = None  # not used
        epistemic = None  # not used
        if isinstance(predict_var, tuple):
            predictive_var, epistemic, aleatoric = predict_var

        assert torch.is_tensor(predictive_var)
        assert predictive_var.ndim == 3 and predictive_var.size(-1) == predictive_var.size(-2), \
            f"predictive_var must be (bs, d, d), got {tuple(predictive_var.shape)}"

        d = predictive_var.size(-1)
        jitter = 1e-12 * torch.eye(d, device=predictive_var.device, dtype=predictive_var.dtype)

        scores = torch.logdet(predictive_var + jitter[None, :, :])  # (bs,)
        return scores
