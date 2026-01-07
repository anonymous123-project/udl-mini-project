import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS
from builder import build_loss


@MODELS.register()
class BCNNReg(nn.Module):
    def __init__(self, loss_dict, T):
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
        self.T = T

    @torch.no_grad()
    def mc_drop_forward_test(self, x):
        assert not self.training
        T = self.T
        self.mc_eval()
        preds = torch.stack([self.forward_train(x) for _ in range(T)])  # T, bs, 10
        self.eval()
        return preds.mean(0), preds   # (bs, 10), (T, bs, 10)

    # returns: (a, b) where first is the aggregated results, second contains results for each stochastic run
    def forward_to_acquire(self, x):
        return self.mc_drop_forward_test(x)   # (bs, 10), (T, bs, 10)

    @torch.no_grad()
    def forward_test(self, x, y=None, return_metrics=False):
        assert not self.training
        mean_preds, _ = self.mc_drop_forward_test(x)
        if return_metrics:
            assert y is not None
            mse = F.mse_loss(mean_preds, y, reduction="mean")
            rmse = torch.sqrt(mse)
            return mean_preds, {"RMSE": rmse}
        else:
            return mean_preds

    # if return loss is set, only the loss is returned, not the outputs
    # return probs and return loss cannot be set both
    def forward_train(self, x, y=None, return_loss=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)  # (bs, 3872)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        logits = self.fc2(x)  # (bs, 10)

        if return_loss:
            return self.get_loss_dict(logits, y)

        return logits

    def get_loss_dict(self, logits, y):
        assert y is not None
        assert y.ndim == 2 and y.size(0) == logits.size(0)

        loss_dict = dict()
        for name, loss_fn in self.losses.items():
            loss_dict[name] = loss_fn(self) if name == "WeightDecay" else loss_fn(logits, y)
        return loss_dict

    def mc_eval(self):
        super().eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        return self

