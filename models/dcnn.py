import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss


@MODELS.register()
class DCNN(nn.Module):
    def __init__(self, loss_dict):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(p=0.25)

        # 28x28 -> conv4 -> 25x25 -> conv4 -> 22x22 -> pool2 -> 11x11
        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

        self.losses = {n: build_loss(l) for n, l in loss_dict.items()}

    # deterministic: forward pass producing probabilities.
    @torch.no_grad()
    def forward_to_acquire(self, x):
        assert not self.training
        probs = self.forward_train(x, return_probs=True)
        all_probs = probs.unsqueeze(0)  # (1, bs, 10) to match acquisition signature
        return probs, all_probs

    @torch.no_grad()
    def forward_test(self, x, y=None, return_metrics=False):
        assert not self.training
        mean_probs = self.forward_train(x, return_probs=True)

        if return_metrics:
            assert y is not None
            log_mean_probs = torch.log(mean_probs.clamp(min=1e-12))
            return mean_probs, {"PredictiveNLL": F.nll_loss(log_mean_probs, y)}
        else:
            return mean_probs

    # if return loss is set, only the loss is returned, not the outputs
    # return probs and return loss cannot be set both
    def forward_train(self, x, y=None, return_probs=False, return_loss=False):
        assert not (return_probs and return_loss)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        logits = self.fc2(x)

        if return_loss:
            return self.get_loss_dict(logits, y)

        if return_probs:
            return F.softmax(logits, dim=1)
        return logits

    def get_loss_dict(self, logits, y):
        assert y is not None
        assert y.ndim == 1 and y.size(0) == logits.size(0)

        loss_dict = {}
        for name, loss_fn in self.losses.items():
            loss_dict[name] = loss_fn(self) if name == "WeightDecay" else loss_fn(logits, y)
        return loss_dict
