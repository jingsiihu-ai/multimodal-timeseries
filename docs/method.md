# Method notes

## Objective

Given a numerical context window `x` and short text metadata `t`, predict the next `H` values. The reference model uses two encoders and a learned element-wise fusion gate:

`z = g * z_series + (1 - g) * z_text`

The forecast head maps `z` to the requested horizon. A symmetric in-batch InfoNCE term encourages paired series and text representations to agree while keeping the forecasting objective primary.

## Why a compact text encoder?

The public demo is intentionally CPU-friendly and offline. `TextContextEncoder` can be replaced by a frozen pretrained language model while keeping the fusion and training interfaces unchanged. `LoRALinear` demonstrates the low-rank adaptation mechanism without downloading model weights.

## References

- Jin et al., [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728), 2023.
- Zhou et al., [One Fits All: Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939), NeurIPS 2023.
- Liu et al., [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://aclanthology.org/2022.acl-short.8/), ACL 2022.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.

