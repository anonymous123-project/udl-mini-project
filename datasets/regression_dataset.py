import torch
from torch.utils.data import Dataset


class MNISTRegressionDataset(Dataset):
    def __init__(self, base_dataset, num_classes=10):
        self.base = base_dataset
        self.num_classes = num_classes

    @property
    def targets(self):
        return self.base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]   # y is int
        y_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        y_onehot[y] = 1.0
        return x, y_onehot
