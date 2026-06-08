import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2 as Mamba2Kernel
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False


class SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor, delta: torch.Tensor,
                B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        Bs, L, D_inner = x.shape
        A = -self.A_log.exp()
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(Bs, D_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        return y + self.D.unsqueeze(0).unsqueeze(0) * x


class MambaBlockPyTorch(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.bc_proj = nn.Linear(d_inner, 2 * d_state)

        self.ssm = SelectiveSSM(d_inner, d_state)

        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        d_inner = self.d_model * self.expand

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x_conv = x.transpose(-1, -2)
        x_conv = self.conv1d(x_conv)[..., :L]
        x = self.act(x_conv.transpose(-1, -2))

        dt = self.dt_proj(x)
        dt = F.softplus(dt)

        bc = self.bc_proj(x)
        B_proj, C_proj = bc.chunk(2, dim=-1)

        y = self.ssm(x, dt, B_proj, C_proj)
        y = y * self.act(z)

        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        if HAS_CUDA_KERNEL:
            self.mamba = Mamba2Kernel(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = MambaBlockPyTorch(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)
