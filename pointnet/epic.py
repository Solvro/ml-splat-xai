import torch
import torch.nn as nn

class EpicDisentangler(nn.Module):
    def __init__(self, C: int = 1024, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.C = C
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

        self.A_raw = nn.Parameter(torch.zeros(self.C, self.C, device=device, dtype=dtype))

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        anti_sym = self.A_raw - self.A_raw.T
        W = torch.matrix_exp(anti_sym)

        return torch.einsum("cd,bdn->bcn", W, point_features)

    @torch.no_grad()
    def get_weight(self) -> torch.Tensor:
        anti_sym = self.A_raw - self.A_raw.T
        W = torch.matrix_exp(anti_sym)
        return W.detach().cpu()

    @torch.no_grad()
    def inverse(self) -> torch.Tensor:
        anti_sym = self.A_raw - self.A_raw.T
        W = torch.matrix_exp(anti_sym)
        return W.t().detach()
