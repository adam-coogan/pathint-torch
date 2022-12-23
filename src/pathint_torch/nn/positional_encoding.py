import torch
from torch import nn
from torchtyping import TensorType


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        L_max: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.L_max = L_max
        self.omegas = 2 ** torch.linspace(
            0, L_max - 1, L_max, dtype=dtype, device=device
        )

    def __call__(self, t: TensorType["b"]) -> TensorType["b", "2*L_max"]:
        angles = t[:, None] * self.omegas[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), -1)
