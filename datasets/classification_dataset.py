from torch.utils.data import Dataset


class MNISTClassificationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    @property
    def targets(self):
        return self.base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]   # (x, y_int)
