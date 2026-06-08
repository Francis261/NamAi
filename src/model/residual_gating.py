import torch
import torch.nn as nn


class ResidualGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * sublayer_out
