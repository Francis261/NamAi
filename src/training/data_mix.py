import math
import random
from dataclasses import dataclass, field
from typing import Optional, Iterator
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, get_dataset_config_names, Dataset

from src.model.tokenizer import get_tokenizer


@dataclass
class DataSourceConfig:
    name: str
    category: str
    hf_path: str
    hf_config: Optional[str] = None
    hf_split: str = "train"
    text_column: str = "text"
    weight: float = 1.0
    max_samples: Optional[int] = None


@dataclass
class DataMixConfig:
    sources: list = field(default_factory=list)
    seed: int = 42
    max_seq_len: int = 8192
    prefetch_factor: int = 4
    global_batch_size: int = 8

    @classmethod
    def default_mix(cls) -> "DataMixConfig":
        return cls(sources=[
            DataSourceConfig("c4", "general", "c4", "en", weight=12.0),
            DataSourceConfig("wiki", "general", "wikipedia", "20220301.en", weight=8.0),
            DataSourceConfig("squad", "qa", "squad", weight=4.0),
            DataSourceConfig("triviaqa", "qa", "trivia_qa", "rc", weight=2.0),
            DataSourceConfig("natural_questions", "qa", "natural_questions", weight=2.0),
            DataSourceConfig("sharegpt", "conversation", "OpenAssistant/oasst1", weight=4.0),
            DataSourceConfig("ultrachat", "conversation", "HuggingFaceH4/ultrachat_200k", weight=4.0),
            DataSourceConfig("alpaca", "instruction", "tatsu-lab/alpaca", weight=3.0),
            DataSourceConfig("dolly", "instruction", "databricks/databricks-dolly-15k", weight=2.0),
            DataSourceConfig("ultrafeedback", "preference", "HuggingFaceH4/ultrafeedback_binarized", weight=3.0),
            DataSourceConfig("gsm8k", "math", "gsm8k", "main", weight=3.0),
            DataSourceConfig("math", "math", "competition_math", weight=2.0),
            DataSourceConfig("hellaswag", "reasoning", "hellaswag", weight=3.0),
            DataSourceConfig("arc", "reasoning", "ai2_arc", "ARC-Challenge", weight=2.0),
            DataSourceConfig("cnn_dailymail", "summarization", "cnn_dailymail", "3.0.0", weight=2.0),
            DataSourceConfig("race", "reading", "race", "high", weight=2.0),
            DataSourceConfig("pubmedqa", "science", "pubmed_qa", "pqa_labeled", weight=1.0),
            DataSourceConfig("codealpaca", "coding", "HuggingFaceH4/CodeAlpaca_20K", weight=3.0),
            DataSourceConfig("anthropic_hh", "safety", "anthropic/hh-rlhf", weight=3.0),
            DataSourceConfig("pile", "general", "monology/pile-uncopyrighted", weight=6.0),
        ])

    @classmethod
    def from_yaml(cls, path: str) -> "DataMixConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        sources = [DataSourceConfig(**s) for s in data.get("sources", [])]
        return cls(
            sources=sources,
            seed=data.get("seed", 42),
            max_seq_len=data.get("max_seq_len", 8192),
            global_batch_size=data.get("global_batch_size", 8),
        )


_DEPRECATED_ALIASES = {
    "c4": "allenai/c4",
}

def _load_hf_source(cfg: DataSourceConfig, seed: int):
    _FALLBACK_SPLITS = ["train", "train_sft", "train_prefs"]

    def _try_load(path, split=None):
        kwargs = dict(
            path=path,
            split=split or cfg.hf_split,
            streaming=True,
            trust_remote_code=True,
        )
        if cfg.hf_config:
            kwargs["name"] = cfg.hf_config
        return load_dataset(**kwargs)

    def _try_splits(path):
        for split in _FALLBACK_SPLITS:
            try:
                return _try_load(path, split=split)
            except (ValueError, KeyError):
                continue
        return _try_load(path)

    try:
        return _try_splits(cfg.hf_path)
    except Exception as e:
        alias = _DEPRECATED_ALIASES.get(cfg.hf_path)
        if alias:
            try:
                print(f"  Retrying {cfg.name} with {alias}")
                return _try_splits(alias)
            except Exception as e2:
                msg = str(e2).replace("\n", " ")[:100]
                print(f"  [WARN] Failed to load {cfg.name} ({alias}): {msg}")
        else:
            msg = str(e).replace("\n", " ")[:100]
            print(f"  [WARN] Failed to load {cfg.name} ({cfg.hf_path}): {msg}")
        return None


def _get_text(example, text_column: str):
    if text_column in example:
        return example[text_column]
    for fallback in ["text", "content", "instruction", "output", "question", "answer",
                      "dialog", "messages", "conversation", "input"]:
        if fallback in example:
            val = example[fallback]
            if isinstance(val, list):
                if isinstance(val[0], dict) and "content" in val[0]:
                    return " ".join(m.get("content", "") for m in val)
                return " ".join(str(v) for v in val)
            return str(val)
    return str(example)


class MultiSourceDataset(IterableDataset):
    def __init__(self, mix_config: DataMixConfig):
        self.config = mix_config
        self.tokenizer = get_tokenizer()
        self.rng = random.Random(mix_config.seed)

        self.sources = []
        self.weights = []
        print("Loading data sources:")
        for src_cfg in mix_config.sources:
            ds = _load_hf_source(src_cfg, mix_config.seed)
            if ds is not None:
                self.sources.append((src_cfg, ds))
                self.weights.append(src_cfg.weight)
                print(f"  ✓ {src_cfg.name:20s} ({src_cfg.category:15s}) weight={src_cfg.weight}")
            else:
                print(f"  ✗ {src_cfg.name:20s} ({src_cfg.category:15s}) SKIPPED")

        total = sum(self.weights)
        total = sum(self.weights)
        self.probs = [w / total for w in self.weights]
        print(f"\nActive sources: {len(self.sources)}  Total weight: {total}")
        self._iterators = {}
        self._source_counts = [0] * len(self.sources)
        self._max_per_source = [s.max_samples or float("inf") for s in self.config.sources]

    def _get_iterator(self, idx: int):
        if idx not in self._iterators:
            self._iterators[idx] = iter(self.sources[idx][1])
        return self._iterators[idx]

    def _sample_source(self, rng):
        active = [i for i in range(len(self.sources))
                  if self._source_counts[i] < self._max_per_source[i]]
        if not active:
            raise StopIteration("All sources exhausted")
        weights = [self.probs[i] for i in active]
        idx = rng.choices(active, weights=weights, k=1)[0]
        return idx

    def __iter__(self) -> Iterator[dict]:
        token_buffer = []
        rng = random.Random(self.config.seed)

        while True:
            try:
                idx = self._sample_source(rng)
            except StopIteration:
                return

            src_cfg, _ = self.sources[idx]
            it = self._get_iterator(idx)
            try:
                example = next(it)
                self._source_counts[idx] += 1
            except StopIteration:
                del self._iterators[idx]
                continue

            text = _get_text(example, src_cfg.text_column)
            tokens = self.tokenizer.encode(text)

            token_buffer.extend(tokens)
            while len(token_buffer) >= self.config.max_seq_len + 1:
                chunk = token_buffer[:self.config.max_seq_len + 1]
                token_buffer = token_buffer[self.config.max_seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
