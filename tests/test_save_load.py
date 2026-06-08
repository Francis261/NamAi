import torch
import tempfile
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.model.save_load import save_checkpoint, load_checkpoint


def test_save_load_roundtrip():
    config = ModelConfig(
        d_model=64, d_state=2, d_conv=4, expand_factor=2,
        num_layers=2, attn_every_n=2,
        num_attention_heads=2, num_kv_heads=1, d_head=32,
        sliding_window_size=16, num_global_tokens=4, max_seq_len=32,
        vocab_size=50257,
    )
    model = MambaAttentionLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(tmpdir, "test", model, config,
            step=42, loss=1.23, tokens_seen=1000)
        loaded, extra = load_checkpoint(f"{tmpdir}/test")
        assert extra["step"] == 42
        assert extra["tokens_seen"] == 1000

        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)


def test_baseline_configs_load():
    for path in ["configs/baseline_pure_mamba.yaml",
                 "configs/baseline_pure_attn.yaml",
                 "configs/350m.yaml"]:
        config = ModelConfig.from_yaml(path)
        model = MambaAttentionLM(config)
        assert sum(p.numel() for p in model.parameters()) > 0
