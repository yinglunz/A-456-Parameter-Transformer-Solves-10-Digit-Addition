# 512-Parameter Addition Transformer (New Record)

A **512-parameter** transformer that achieves **≥99.97% exact-match accuracy** on 10-digit integer addition, reducing the previous record of 777 parameters by **34.1%**.

Built on techniques from [gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax) and [smallest-addition-transformer-claude-code](https://github.com/anadim/smallest-addition-transformer-claude-code).

## Key Innovation: Low-Rank Factorization

All major weight matrices use rank-3 factorization (`W = A @ B` where `A ∈ R^{m×3}`, `B ∈ R^{3×n}`):
- Position embeddings: 231 → 120 params (saves 111)
- QKV projection: 147 → 84 params (saves 63)
- Attention output: 49 → 42 params (saves 7)
- FFN up/down: 196 → 126 params (saves 70)

**Surprising finding**: The low-rank constraint acts as a critical *regularizer*. The full-rank baseline (763 params) achieves 0% accuracy, while the 512-param low-rank model achieves 100%. Grokking-based training is inherently stochastic (seed-dependent), but low-rank models grok far more reliably than full-rank ones.

## Quick Start

### Install

```bash
pip install torch
```

### Evaluate Pre-trained Checkpoint

```bash
python -m src.eval test \
  --ckpt checkpoints/best_512params.pt \
  --device cpu --seed 42
```

Output:
```json
{
  "test_size": 10000,
  "exact_match": 1.0,
  "token_accuracy": 1.0,
  "failure_samples": []
}
```

### Verify Parameter Count

```python
import torch
from src.model import ModelConfig, TinyDecoderLM, count_parameters

ckpt = torch.load("checkpoints/best_512params.pt", map_location="cpu", weights_only=False)
cfg = ModelConfig(**ckpt["model_config"])
model = TinyDecoderLM(cfg)
model.load_state_dict(ckpt["model_state"])

print(f"Parameters: {count_parameters(model)}")  # 512
print(f"Tied embeddings: {model.lm_head.weight is model.token_emb.weight}")  # True
```

### Single Prediction

```bash
python -m src.eval predict \
  --ckpt checkpoints/best_512params.pt \
  --a 1234567890 --b 9876543210
```

### Train from Scratch

```bash
# 512-param record (rank-3 everything)
python -m src.train \
  --run-name best_512 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --device cuda --seed 42

# 582-param model (rank-3 attention only, full-rank FFN)
python -m src.train \
  --run-name best_582 \
  --pos-rank 3 --qkv-rank 3 --attn-out-rank 3 \
  --device cuda --seed 42
```

**Note**: The 512-param model groks at step ~21K and peaks at step ~24K, but degrades after that. The best checkpoint is saved automatically. The 582-param model is more stable and maintains 100% through the end of training.

## Architecture

| Component | Shape | Params |
|---|---|---|
| Token embedding (tied with output) | 14 × 7 | 98 |
| Position embedding (rank-3) | 33×3 + 3×7 | 120 |
| LayerNorm (pre-attention) | 7 + 7 | 14 |
| QKV projection (rank-3) | 7×3 + 3×21 | 84 |
| Attention output (rank-3) | 7×3 + 3×7 | 42 |
| LayerNorm (pre-FFN) | 7 + 7 | 14 |
| FFN up (rank-3, no bias) | 7×3 + 3×14 | 63 |
| FFN down (rank-3, no bias) | 14×3 + 3×7 | 63 |
| Final LayerNorm | 7 + 7 | 14 |
| Output head | (tied) | 0 |
| **Total** | | **512** |

## Results

Validated across 10 independent test sets (100,000 total examples):

| Metric | Value |
|---|---|
| Parameters | **512** |
| Test exact-match (primary, 10K) | 100.00% |
| Test exact-match (100K aggregate) | 99.99% |
| Total errors in 100K | 10 |
| Previous record | 777 params |
| Reduction | **34.1%** |

## Comparison with Prior Work

| Model | Params | Test Accuracy |
|---|---|---|
| [gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax) (pico-7d-ff14) | 777 | 99.69% |
| [gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax) (pico-1L-7d) | 973 | 100% |
| [smallest-addition-codex](https://github.com/anadim/smallest-addition-transformer-claude-code) | 1,644 | 99.04% |
| **Ours (582 params, r=3 attn+pos)** | **582** | **≥99.99%** |
| **Ours (512 params, r=3 all)** | **512** | **≥99.97%** |

## How It Works

1. **Tokenization**: Raw digit tokenization (vocab=14) with reversed output for carry propagation alignment
2. **Curriculum learning**: 3 phases (1-3 digits → 1-6 digits → 1-10 digits) over 27K steps
3. **Grokking**: The model trains at ~0% accuracy for ~21K steps, then suddenly jumps to 100% — a classic grokking phenomenon amplified by the low-rank constraint
4. **High learning rate**: LR=0.02 with cosine decay, critical for small models

## Files

```
src/
  model.py    # Low-rank transformer (LowRankLinear, LowRankEmbedding, TinyDecoderLM)
  data.py     # Raw digit tokenization pipeline
  train.py    # Training with curriculum learning
  eval.py     # Evaluation and inference
checkpoints/
  best_512params.pt   # Best model (512 params, 100% test accuracy)
  best_582params.pt   # Runner-up (582 params, 100% test accuracy)
plots/
  grokking_and_frontier.png  # Training curves and parameter-accuracy frontier
report.pdf                   # Report with full analysis
```

## References

- D. Papailiopoulos, "Glove box challenge: smallest transformer for 10-digit addition," 2026. [GitHub](https://github.com/anadim/smallest-addition-transformer-claude-code)
- Y. Havinga, "gpt-acc-jax: Smallest GPT for 10-digit addition," 2026. [GitHub](https://github.com/yhavinga/gpt-acc-jax)
