import torch.nn.functional as F
import torch.nn as nn

from registry.registry import LOSSES


@LOSSES.register()
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gts):
        # preds, gts: (bs, d) or any same-shape tensors
        assert preds.shape == gts.shape, f"Shape mismatch: {preds.shape} vs {gts.shape}"
        return F.mse_loss(preds, gts, reduction="mean")
