import torch.nn as nn
import torch.nn.functional as F

from registry.registry import LOSSES


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, y):
        assert y is not None
        assert y.ndim == 1 and y.size(0) == logits.size(0)
        return F.cross_entropy(logits, y, reduction=self.reduction)
