import torch


class Logdet:
    def __init__(self):
        pass

    def __call__(self, X):
        assert X.ndim >= 3, "Input must have at least 3 dims"
        assert X.shape[-1] == X.shape[-2], "Last two dims must be equal"

        return torch.logdet(X)
