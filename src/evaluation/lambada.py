import torch
from datasets import load_dataset
from src.model.tokenizer import get_tokenizer


@torch.no_grad()
def evaluate_lambada(model, max_seq_len: int = 512):
    model.eval()
    tokenizer = get_tokenizer()
    dataset = load_dataset("lambada", split="test")
    device = model.device

    correct = 0
    total = 0

    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        if tokens.numel() > max_seq_len:
            tokens = tokens[-(max_seq_len + 1):]

        if tokens.numel() <= 1:
            continue

        input_ids = tokens[:-1].unsqueeze(0)
        target = tokens[-1:]

        logits, _ = model(input_ids)
        last_logits = logits[0, -1]
        pred = last_logits.argmax()

        if pred == target.item():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
