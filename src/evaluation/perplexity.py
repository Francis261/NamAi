import math
import torch
from datasets import load_dataset
from src.model.tokenizer import get_tokenizer


@torch.no_grad()
def evaluate_perplexity(model, max_seq_len: int = 8192, stride: int = 512,
                        max_samples: int = None):
    model.eval()
    tokenizer = get_tokenizer()
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    if max_samples:
        dataset = dataset.select(range(max_samples))

    total_nll = 0.0
    total_tokens = 0

    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=next(model.parameters()).device)

        if tokens.numel() <= 1:
            continue

        nll = 0.0
        n_tokens = 0
        for i in range(0, tokens.numel() - 1, stride):
            chunk = tokens[i: i + max_seq_len + 1]
            if chunk.numel() <= 1:
                break
            input_ids = chunk[:-1].unsqueeze(0)
            labels = chunk[1:].unsqueeze(0)
            logits, loss = model(input_ids, labels)
            n_tokens += labels.numel()
            nll += loss.item() * labels.numel()

        total_nll += nll
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    ppl = math.exp(total_nll / total_tokens)
    return ppl
