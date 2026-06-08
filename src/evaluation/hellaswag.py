import torch
import torch.nn.functional as F
from datasets import load_dataset
from src.model.tokenizer import get_tokenizer


@torch.no_grad()
def evaluate_hellaswag(model, max_seq_len: int = 512, max_samples: int = None):
    model.eval()
    tokenizer = get_tokenizer()
    dataset = load_dataset("hellaswag", split="validation")
    if max_samples:
        dataset = dataset.select(range(max_samples))

    correct = 0
    total = 0
    device = model.device

    for example in dataset:
        ctx = example["ctx"]
        endings = example["endings"]

        scores = []
        for ending in endings:
            text = ctx + " " + ending
            tokens = tokenizer.encode(text)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            if tokens.numel() > max_seq_len:
                tokens = tokens[:max_seq_len]
            if tokens.numel() <= 1:
                scores.append(1e10)
                continue
            input_ids = tokens[:-1].unsqueeze(0)
            targets = tokens[1:]
            logits, _ = model(input_ids)
            logits = logits[0]
            losses = F.cross_entropy(logits, targets, reduction="none")
            scores.append(losses.mean().item())

        if scores:
            if scores.index(min(scores)) == example["label"]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
