import torch
from src.model.mamba_block import MambaBlockPyTorch, MambaBlock, HAS_CUDA_KERNEL


def test_mamba_fallback_shapes():
    block = MambaBlockPyTorch(d_model=64, d_state=4, d_conv=4, expand=2)
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out = block(x)
    assert out.shape == (B, L, D)


def test_mamba_fallback_no_nan():
    block = MambaBlockPyTorch(d_model=64, d_state=4, d_conv=4, expand=2)
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out = block(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_mamba_fallback_gradient():
    block = MambaBlockPyTorch(d_model=32, d_state=2, d_conv=4, expand=2)
    B, L, D = 2, 8, 32
    x = torch.randn(B, L, D, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mamba_wrapper():
    block = MambaBlock(d_model=64, d_state=4, d_conv=4, expand=2)
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out = block(x)
    assert out.shape == (B, L, D)
