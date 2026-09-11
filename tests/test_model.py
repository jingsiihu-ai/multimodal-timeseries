import torch

from multimodal_timeseries import (
    LoRALinear,
    MultimodalForecaster,
    make_synthetic_batch,
    symmetric_info_nce,
    tokenize_context,
)


def test_forecaster_shapes() -> None:
    history, _, descriptions = make_synthetic_batch(batch_size=5)
    output = MultimodalForecaster()(history, tokenize_context(descriptions))
    assert output["forecast"].shape == (5, 6)
    assert output["series_embedding"].shape == output["text_embedding"].shape
    assert torch.all((output["gate"] >= 0) & (output["gate"] <= 1))


def test_tokenization_is_deterministic() -> None:
    first = tokenize_context(["high demand", "low demand"])
    second = tokenize_context(["high demand", "low demand"])
    assert torch.equal(first, second)
    assert torch.all(first[:, :2] > 0)


def test_lora_initially_matches_base_projection() -> None:
    layer = LoRALinear(4, 3, rank=2)
    inputs = torch.randn(6, 4)
    assert torch.allclose(layer(inputs), layer.base(inputs))
    assert not layer.base.weight.requires_grad


def test_info_nce_is_finite_and_differentiable() -> None:
    left = torch.randn(4, 8, requires_grad=True)
    right = torch.randn(4, 8, requires_grad=True)
    loss = symmetric_info_nce(left, right)
    loss.backward()
    assert torch.isfinite(loss)
    assert left.grad is not None and right.grad is not None


def test_invalid_temperature_is_rejected() -> None:
    embeddings = torch.randn(2, 4)
    try:
        symmetric_info_nce(embeddings, embeddings, temperature=0)
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("expected ValueError")

