import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs


def apply_rope(x: torch.Tensor, freqs: torch.Tensor):
    B, H, L, D = x.shape
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = freqs[:L].cos().unsqueeze(0).unsqueeze(0)
    sin = freqs[:L].sin().unsqueeze(0).unsqueeze(0)
    x_rotated = torch.stack([
        x_[..., 0] * cos - x_[..., 1] * sin,
        x_[..., 1] * cos + x_[..., 0] * sin,
    ], dim=-1)
    return x_rotated.flatten(-2).type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: int,
                 max_seq_len: int, rope_theta: float, sliding_window: int,
                 num_global_tokens: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.sliding_window = sliding_window
        self.num_global_tokens = num_global_tokens
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        rope_freqs = precompute_rope_freqs(d_head, max_seq_len, rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        device = x.device

        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = self._sparse_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_head)
        return self.o_proj(out)

    def _sparse_attention(self, q, k, v):
        B, H, L, D = q.shape
        scale = D ** -0.5
        G = min(self.num_global_tokens, L)
        W = min(self.sliding_window, L - 1)

        if G >= L:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        q_g, q_l = q[:, :, :G], q[:, :, G:]
        k_g, k_l = k[:, :, :G], k[:, :, G:]
        v_g, v_l = v[:, :, :G], v[:, :, G:]

        attn_g = F.scaled_dot_product_attention(q_g, k_g, v_g, is_causal=True)

        L_l = L - G
        if L_l == 0:
            return attn_g

        k_pad = F.pad(k_l, (0, 0, W, 0))
        v_pad = F.pad(v_l, (0, 0, W, 0))
        k_unfold = k_pad.unfold(2, W + 1, 1).transpose(-1, -2).contiguous()
        v_unfold = v_pad.unfold(2, W + 1, 1).transpose(-1, -2).contiguous()

        arange_col = torch.arange(W + 1, device=q.device).unsqueeze(0)
        pos = torch.arange(L_l, device=q.device).unsqueeze(-1)
        pad_mask = torch.where(arange_col < (W - pos), float("-inf"), 0.0)

        logits_lg = torch.einsum("bhld,bhgd->bhlg", q_l, k_g) * scale
        logits_ll = torch.einsum("bhld,bhlwd->bhlw", q_l, k_unfold) * scale + pad_mask
        logits_cat = torch.cat([logits_lg, logits_ll], dim=-1)
        attn_weights = F.softmax(logits_cat, dim=-1)

        attn_lg, attn_ll = attn_weights.split([G, W + 1], dim=-1)
        out_l = torch.einsum("bhlg,bhgd->bhld", attn_lg, v_g) + \
                torch.einsum("bhlw,bhlwd->bhld", attn_ll, v_unfold)

        return torch.cat([attn_g, out_l], dim=2)
