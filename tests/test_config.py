from src.model.config import ModelConfig


def test_default_config():
    config = ModelConfig()
    assert config.vocab_size == 50257
    assert config.d_model == 1024
    assert config.num_layers == 24


def test_d_ffn():
    config = ModelConfig(d_model=1024, swiglu_hidden_mult=2.75)
    assert config.d_ffn == 2816


def test_num_attn_layers():
    config = ModelConfig(num_layers=24, attn_every_n=6)
    assert config.num_attn_layers == 4


def test_custom_config():
    config = ModelConfig(
        d_model=512,
        num_layers=8,
        attn_every_n=4,
    )
    assert config.d_model == 512
    assert config.num_layers == 8
    assert config.d_ffn == int(512 * 2.75)
    assert config.num_attn_layers == 2


def test_350m_yaml(tmp_path):
    import yaml
    cfg = {
        "d_model": 1024,
        "d_state": 16,
        "num_layers": 24,
        "attn_every_n": 6,
        "num_attention_heads": 16,
        "num_kv_heads": 4,
        "d_head": 64,
        "sliding_window_size": 2048,
        "num_global_tokens": 64,
        "max_seq_len": 8192,
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = ModelConfig.from_yaml(str(path))
    assert config.d_model == 1024
    assert config.num_layers == 24
