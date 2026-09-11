"""Synthetic data and deterministic text tokenization for examples and tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def tokenize_context(
    texts: Sequence[str], vocab_size: int = 256, max_tokens: int = 8
) -> Tensor:
    """Map whitespace tokens to stable hashed IDs; zero is reserved for padding."""

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    output = np.zeros((len(texts), max_tokens), dtype=np.int64)
    for row, text in enumerate(texts):
        for column, token in enumerate(text.lower().split()[:max_tokens]):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            output[row, column] = int.from_bytes(digest, "little") % (vocab_size - 1) + 1
    return torch.from_numpy(output)


def make_synthetic_batch(
    batch_size: int = 64,
    context_length: int = 24,
    horizon: int = 6,
    seed: int = 7,
) -> tuple[Tensor, Tensor, list[str]]:
    """Create seasonal signals whose future level is described by text metadata."""

    rng = np.random.default_rng(seed)
    total = context_length + horizon
    time = np.arange(total, dtype=np.float32)
    contexts = rng.integers(0, 2, size=batch_size)
    level = np.where(contexts == 1, 0.7, -0.7).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=batch_size).astype(np.float32)
    signal = np.sin(time[None, :] * 2.0 * np.pi / 12.0 + phase[:, None])
    signal += level[:, None]
    signal += rng.normal(0.0, 0.05, size=signal.shape)
    descriptions = ["high demand" if value else "low demand" for value in contexts]
    history = torch.tensor(signal[:, :context_length], dtype=torch.float32)
    target = torch.tensor(signal[:, context_length:], dtype=torch.float32)
    return history, target, descriptions

