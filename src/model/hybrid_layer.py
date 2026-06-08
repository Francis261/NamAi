import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import ModelConfig
from .mamba_block import MambaBlock
from .attention import GroupedQueryAttention
from .residual_gating import ResidualGate


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MambaLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_checkpointing = config.use_checkpointing
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mamba = MambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand_factor,
        )
        self.gate = ResidualGate(config.d_model)

    def _forward_impl(self, x):
        return self.mamba(self.norm(x))

    def forward(self, x):
        if self.use_checkpointing and self.training:
            mamba_out = torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        else:
            mamba_out = self._forward_impl(x)
        return self.gate(x, mamba_out)


class AttentionLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_checkpointing = config.use_checkpointing
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_kv_heads,
            d_head=config.d_head,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            sliding_window=config.sliding_window_size,
            num_global_tokens=config.num_global_tokens,
        )
        self.gate_attn = ResidualGate(config.d_model)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ffn)
        self.gate_ffn = ResidualGate(config.d_model)

    def _attn_impl(self, x):
        return self.attn(self.norm1(x))

    def _ffn_impl(self, x):
        return self.ffn(self.norm2(x))

    def forward(self, x):
        if self.use_checkpointing and self.training:
            attn_out = torch.utils.checkpoint.checkpoint(
                self._attn_impl, x, use_reentrant=False
            )
        else:
            attn_out = self._attn_impl(x)
        x = self.gate_attn(x, attn_out)

        if self.use_checkpointing and self.training:
            ffn_out = torch.utils.checkpoint.checkpoint(
                self._ffn_impl, x, use_reentrant=False
            )
        else:
            ffn_out = self._ffn_impl(x)
        return self.gate_ffn(x, ffn_out)


class HybridLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        is_attn = (layer_idx + 1) % config.attn_every_n == 0
        if is_attn:
            self.layer = AttentionLayer(config)
        else:
            self.layer = MambaLayer(config)
        self.is_attention = is_attn

    def forward(self, x):
        return self.layer(x)
