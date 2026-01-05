import torch
import torch.nn as nn

from registry.registry import LOSSES


# L2 penalty
# lambda is applied outside.
@LOSSES.register()
class L2WeightDecay(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model):
        assert model is not None
        device = next(model.parameters()).device

        penalty = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad:
                penalty = penalty + (p ** 2).sum()
        return penalty
