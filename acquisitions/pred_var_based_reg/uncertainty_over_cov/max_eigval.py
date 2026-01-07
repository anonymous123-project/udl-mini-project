import torch


class MaxEigVal:
    def __init__(self):
        pass

    def __call__(self, X):
        assert X.ndim >= 3, "Input must have at least 3 dims"
        assert X.shape[-1] == X.shape[-2], "Last two dims must be equal"

        eigvals = torch.linalg.eigvalsh(X)
        return eigvals[..., -1]
