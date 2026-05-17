from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset

from gelu_1l.constants import DEFAULT_DATASET_NAME, DEFAULT_MODEL_NAME
from gelu_1l.sae import SparseAutoencoder


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gelu_1l(device: str, cache_dir: Path):
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        DEFAULT_MODEL_NAME,
        device=device,
        cache_dir=str(cache_dir),
        dtype="float32",
        default_prepend_bos=True,
    ) 


def load_sae(run: int | str, device: str, cache_dir: Path) -> SparseAutoencoder:
    return SparseAutoencoder.from_hf(run=run, cache_dir=cache_dir, device=device)


def load_texts(max_texts: int, cache_dir: Path, dataset_name: str = DEFAULT_DATASET_NAME) -> list[str]:
    dataset = load_dataset(dataset_name, split="train", cache_dir=str(cache_dir))
    return list(dataset.select(range(min(max_texts, len(dataset))))["text"])


def batched(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def tokenize_batch(model, texts: list[str], seq_len: int, device: str) -> torch.Tensor:
    tokens = model.to_tokens(texts, prepend_bos=True, truncate=True, move_to_device=True)
    tokens = tokens[:, :seq_len]
    return tokens.to(device)
