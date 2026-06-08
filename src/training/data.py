import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from src.model.tokenizer import get_tokenizer


_DATASET_ALIASES = {
    "c4": "allenai/c4",
}

def _resolve_dataset(path: str, config: str, split: str):
    try:
        return load_dataset(path, config, split=split, streaming=True)
    except Exception:
        alias = _DATASET_ALIASES.get(path)
        if alias:
            return load_dataset(alias, config, split=split, streaming=True)
        raise


class TokenizedDataset(IterableDataset):
    def __init__(self, dataset_name: str = "c4", split: str = "train",
                 max_seq_len: int = 8192, dataset_size: int = None):
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer()
        dataset = _resolve_dataset(dataset_name, "en", split)
        if dataset_size:
            dataset = dataset.take(dataset_size)
        self.dataset = dataset

    def __iter__(self):
        tokens = []
        for example in self.dataset:
            text = example["text"]
            token_ids = self.tokenizer.encode(text)
            tokens.extend(token_ids)
            while len(tokens) >= self.max_seq_len + 1:
                chunk = tokens[: self.max_seq_len + 1]
                tokens = tokens[self.max_seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def create_dataloader(batch_size: int = 8, max_seq_len: int = 8192,
                      dataset_name: str = "c4", split: str = "train",
                      num_workers: int = 0):
    dataset = TokenizedDataset(
        dataset_name=dataset_name,
        split=split,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
