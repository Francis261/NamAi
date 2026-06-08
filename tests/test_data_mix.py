import sys
sys.path.insert(0, ".")
from src.training.data_mix import DataMixConfig


def test_default_mix_creates_sources():
    cfg = DataMixConfig.default_mix()
    assert len(cfg.sources) > 0
    for s in cfg.sources:
        assert s.name
        assert s.category


def test_yaml_roundtrip(tmp_path):
    import yaml
    data = {"sources": [
        {"name": "test", "category": "general",
         "hf_path": "c4", "hf_config": "en", "weight": 1.0},
    ], "max_seq_len": 512}
    path = tmp_path / "mix.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    cfg = DataMixConfig.from_yaml(str(path))
    assert len(cfg.sources) == 1
    assert cfg.sources[0].name == "test"


def test_default_mix_has_key_categories():
    cfg = DataMixConfig.default_mix()
    categories = {s.category for s in cfg.sources}
    for cat in ["general", "qa", "conversation", "instruction",
                "coding", "math", "reasoning"]:
        assert cat in categories, f"Missing category: {cat}"
