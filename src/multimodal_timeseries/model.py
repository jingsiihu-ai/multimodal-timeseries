"""Small, dependency-light modules for multimodal forecasting research."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank residual update.

    The base projection is frozen by default. The low-rank B matrix starts at
    zero, so the module initially matches the base layer exactly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        if freeze_base:
            self.base.weight.requires_grad_(False)
            self.base.bias.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + self.scale * update


class SeriesPatchEncoder(nn.Module):
    """Encode a fixed-length univariate sequence as pooled patch tokens."""

    def __init__(self, context_length: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        if context_length % patch_size:
            raise ValueError("context_length must be divisible by patch_size")
        self.context_length = context_length
        self.patch_size = patch_size
        self.patch_projection = nn.Linear(patch_size, d_model)
        self.position = nn.Parameter(
            torch.zeros(1, context_length // patch_size, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, series: Tensor) -> Tensor:
        if series.ndim != 2 or series.shape[1] != self.context_length:
            raise ValueError(
                f"expected [batch, {self.context_length}], got {tuple(series.shape)}"
            )
        patches = series.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        tokens = self.patch_projection(patches) + self.position
        return self.norm(tokens.mean(dim=1))


class TextContextEncoder(nn.Module):
    """Encode token IDs with masked mean pooling."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.projection = LoRALinear(d_model, d_model, rank=4, alpha=8.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        mask = token_ids.ne(self.padding_idx).unsqueeze(-1)
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.norm(torch.tanh(self.projection(pooled)))


class MultimodalForecaster(nn.Module):
    """Fuse temporal and textual representations for multi-step forecasting."""

    def __init__(
        self,
        context_length: int = 24,
        patch_size: int = 4,
        horizon: int = 6,
        vocab_size: int = 256,
        d_model: int = 32,
    ) -> None:
        super().__init__()
        self.series_encoder = SeriesPatchEncoder(context_length, patch_size, d_model)
        self.text_encoder = TextContextEncoder(vocab_size, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, series: Tensor, token_ids: Tensor) -> dict[str, Tensor]:
        series_embedding = self.series_encoder(series)
        text_embedding = self.text_encoder(token_ids)
        gate = torch.sigmoid(
            self.gate(torch.cat([series_embedding, text_embedding], dim=-1))
        )
        fused = gate * series_embedding + (1.0 - gate) * text_embedding
        return {
            "forecast": self.forecast_head(fused),
            "series_embedding": series_embedding,
            "text_embedding": text_embedding,
            "gate": gate,
        }


def symmetric_info_nce(
    series_embedding: Tensor,
    text_embedding: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Symmetric in-batch contrastive loss for paired representations."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if series_embedding.shape != text_embedding.shape:
        raise ValueError("paired embeddings must have the same shape")
    series_embedding = F.normalize(series_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    logits = series_embedding @ text_embedding.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )

