import os
import json
import torch
from safetensors.torch import save_file, load_file

from .config import ModelConfig
from .model import MambaAttentionLM


def save_checkpoint(save_dir: str, name: str, model: MambaAttentionLM,
                    config: ModelConfig, **extra):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)

    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
        }, f, indent=2)

    if extra:
        extra_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save(extra, extra_path)

    print(f"Saved to {ckpt_dir}/")


def load_model_from_dir(model_dir: str, device: str = "cpu") -> MambaAttentionLM:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config_data = json.load(f)
    config = ModelConfig(**config_data)

    model = MambaAttentionLM(config)

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device=device)
    else:
        pt_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(pt_path, map_location=device, weights_only=True)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return model.to(device)


def load_checkpoint(ckpt_dir: str, device: str = "cpu"):
    model = load_model_from_dir(ckpt_dir, device=device)
    extra = {}
    extra_path = os.path.join(ckpt_dir, "training_state.pt")
    if os.path.exists(extra_path):
        extra = torch.load(extra_path, map_location=device, weights_only=True)
    return model, extra
