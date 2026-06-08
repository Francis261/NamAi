import torch


def configure_optimizer(model, weight_decay: float = 0.1, lr: float = 3e-4,
                        betas=(0.9, 0.95)):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int,
                        min_lr: float = 1e-5):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr / 3e-4 + 0.5 * (1 - min_lr / 3e-4) * (1 + math.cos(math.pi * progress))

    import math
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
