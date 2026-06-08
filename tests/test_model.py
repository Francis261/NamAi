import torch
import pytest
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.model.hybrid_layer import RMSNorm, SwiGLU, MambaLayer, AttentionLayer, HybridLayer
from src.model.residual_gating import ResidualGate


def mini_config():
    return ModelConfig(
        vocab_size=50257,
        d_model=128,
        d_state=4,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        attn_every_n=2,
        num_attention_heads=4,
        num_kv_heads=2,
        d_head=32,
        sliding_window_size=32,
        num_global_tokens=8,
        max_seq_len=128,
    )


def test_model_creation():
    config = mini_config()
    model = MambaAttentionLM(config)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


def test_model_forward():
    config = mini_config()
    model = MambaAttentionLM(config)
    B, L = 2, 64
    x = torch.randint(0, config.vocab_size, (B, L))
    logits, loss = model(x, labels=x)
    assert logits.shape == (B, L, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_model_backward():
    config = mini_config()
    model = MambaAttentionLM(config)
    B, L = 2, 32
    x = torch.randint(0, config.vocab_size, (B, L))
    logits, loss = model(x, labels=x)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(p.grad).any(), f"{name} has NaN gradient"


def test_model_no_labels():
    config = mini_config()
    model = MambaAttentionLM(config)
    B, L = 2, 64
    x = torch.randint(0, config.vocab_size, (B, L))
    logits, loss = model(x)
    assert logits.shape == (B, L, config.vocab_size)
    assert loss is None


def test_rms_norm():
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_swiglu():
    ffn = SwiGLU(64, 176)
    x = torch.randn(2, 16, 64)
    out = ffn(x)
    assert out.shape == (2, 16, 64)


def test_residual_gate_init():
    gate = ResidualGate(64)
    assert gate.alpha.item() == 0.0


def test_residual_gate_forward():
    gate = ResidualGate(64)
    x = torch.randn(2, 16, 64)
    y = torch.randn(2, 16, 64)
    out = gate(x, y)
    assert torch.allclose(out, x)
    assert id(gate.alpha) is not None


def test_mamba_layer():
    config = mini_config()
    layer = MambaLayer(config)
    B, L, D = 2, 32, 128
    x = torch.randn(B, L, D)
    out = layer(x)
    assert out.shape == (B, L, D)


def test_attention_layer():
    config = mini_config()
    layer = AttentionLayer(config)
    B, L, D = 2, 32, 128
    x = torch.randn(B, L, D)
    out = layer(x)
    assert out.shape == (B, L, D)


def test_hybrid_layer_mamba():
    config = mini_config()
    layer = HybridLayer(config, layer_idx=0)
    assert not layer.is_attention
    B, L, D = 2, 32, 128
    x = torch.randn(B, L, D)
    out = layer(x)
    assert out.shape == (B, L, D)


def test_hybrid_layer_attention():
    config = mini_config()
    layer = HybridLayer(config, layer_idx=1)
    assert layer.is_attention
    B, L, D = 2, 32, 128
    x = torch.randn(B, L, D)
    out = layer(x)
    assert out.shape == (B, L, D)
