"""Train the compact forecaster on deterministic synthetic data."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from multimodal_timeseries import (
    MultimodalForecaster,
    make_synthetic_batch,
    symmetric_info_nce,
    tokenize_context,
)


def main() -> None:
    torch.manual_seed(7)
    history, target, descriptions = make_synthetic_batch(batch_size=96, seed=7)
    token_ids = tokenize_context(descriptions)
    train = slice(0, 72)
    validation = slice(72, None)

    model = MultimodalForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    with torch.no_grad():
        initial = F.mse_loss(
            model(history[validation], token_ids[validation])["forecast"],
            target[validation],
        ).item()

    for _ in range(120):
        output = model(history[train], token_ids[train])
        forecast_loss = F.mse_loss(output["forecast"], target[train])
        alignment_loss = symmetric_info_nce(
            output["series_embedding"], output["text_embedding"]
        )
        loss = forecast_loss + 0.02 * alignment_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = F.mse_loss(
            model(history[validation], token_ids[validation])["forecast"],
            target[validation],
        ).item()

    print(f"validation MSE before training: {initial:.4f}")
    print(f"validation MSE after training:  {final:.4f}")


if __name__ == "__main__":
    main()

