from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    num_layers: int = 24
    attn_every_n: int = 6
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    d_head: int = 64
    swiglu_hidden_mult: float = 2.75
    sliding_window_size: int = 2048
    num_global_tokens: int = 64
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    use_checkpointing: bool = False

    @property
    def d_ffn(self) -> int:
        return int(self.d_model * self.swiglu_hidden_mult)

    @property
    def num_attn_layers(self) -> int:
        return self.num_layers // self.attn_every_n

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
