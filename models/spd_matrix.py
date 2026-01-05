import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.registry import MODELS


@MODELS.register()
class LearnableSPDMatrix(nn.Module):
    """
    spd matrix following cholesky factorization V = L * L^T
    enforce softplus diagonal!
    """
    def __init__(self, k, init_identity=False):
        super().__init__()
        self.k = k

        if init_identity:
            # raw_diag = ln(exp(1 + eps) - 1)
            raw_diag_value = torch.log(torch.exp(torch.tensor(1.0 + 1e-6)) - 1.0)

            raw_L = torch.zeros(k, k)
            raw_L = torch.tril(raw_L)
            raw_L += torch.diag(raw_diag_value.repeat(k))

            self.raw_L = nn.Parameter(raw_L)
        else:
            self.raw_L = nn.Parameter(torch.randn(k, k))

    def get_diagonal(self):
        return F.softplus(torch.diag(self.raw_L)) + 1e-6

    def get_lower_triangular(self):
        L = torch.tril(self.raw_L)
        diagonal = self.get_diagonal()
        L = L - torch.diag(torch.diag(L)) + torch.diag(diagonal)
        return L

    # returns SPD matrix V = L L^T
    def get(self):
        L = self.get_lower_triangular()
        return L @ L.T

    def get_logdet(self):
        diagonal = self.get_diagonal()
        return 2.0 * torch.sum(torch.log(diagonal))

    def get_det(self):
        return torch.exp(self.get_logdet())

    # return V^(-1) = (L^T)^(-1) @ L^(-1) = (L^(-1))^T @ L^(-1)
    def get_inv(self):
        L = self.get_lower_triangular()
        I = torch.eye(self.k, device=L.device, dtype=L.dtype)

        L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        return L_inv.T @ L_inv
