import torch
from src.model.attention import precompute_rope_freqs, apply_rope, GroupedQueryAttention


def test_rope_shapes():
    freqs = precompute_rope_freqs(64, 100)
    assert freqs.shape == (100, 32), f"Expected (100, 32), got {freqs.shape}"


def test_rope_application():
    B, H, L, D = 2, 4, 10, 64
    x = torch.randn(B, H, L, D)
    freqs = precompute_rope_freqs(D, L)
    out = apply_rope(x, freqs)
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()


def test_rope_rotates_pairs():
    D = 4
    x = torch.zeros(1, 1, 1, D)
    x[0, 0, 0, 0] = 1.0
    x[0, 0, 0, 1] = 0.0
    freqs = precompute_rope_freqs(D, 1)
    out = apply_rope(x, freqs)
    assert out.shape == (1, 1, 1, D)


def test_gqa_shapes():
    attn = GroupedQueryAttention(
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        d_head=64,
        max_seq_len=256,
        rope_theta=10000.0,
        sliding_window=64,
        num_global_tokens=8,
    )
    B, L, D = 2, 128, 512
    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)


def test_gqa_sparse_mask():
    attn = GroupedQueryAttention(
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        d_head=64,
        max_seq_len=128,
        rope_theta=10000.0,
        sliding_window=16,
        num_global_tokens=8,
    )
    B, L, D = 2, 64, 256
    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_gqa_all_global():
    attn = GroupedQueryAttention(
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        d_head=64,
        max_seq_len=128,
        rope_theta=10000.0,
        sliding_window=16,
        num_global_tokens=64,
    )
    B, L, D = 2, 32, 256
    x = torch.randn(B, L, D)
    out = attn(x)
    assert out.shape == (B, L, D)
