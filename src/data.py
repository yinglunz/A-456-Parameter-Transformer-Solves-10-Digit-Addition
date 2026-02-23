"""Data pipeline for 10-digit addition using raw digit tokenization.

Tokenization follows gpt-acc-jax: each character is a separate token.
Format: "0000000005+0000000007=" -> reversed sum "21000000000" + <EOS>

Vocab (14 tokens): 0-9, +, =, <PAD>, <EOS>
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Vocabulary  (same as gpt-acc-jax)
# ---------------------------------------------------------------------------
NUM_DIGITS = 10
SUM_DIGITS = 11
MAX_OPERAND = 10**NUM_DIGITS  # 10^10

TOKENS = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "+": 10, "=": 11, "<PAD>": 12, "<EOS>": 13,
}
ITOS = {v: k for k, v in TOKENS.items()}

VOCAB_SIZE = len(TOKENS)  # 14
PAD_ID = TOKENS["<PAD>"]
EOS_ID = TOKENS["<EOS>"]
PLUS_ID = TOKENS["+"]
EQUALS_ID = TOKENS["="]

# Input: 10 digits + '+' + 10 digits + '=' = 22 tokens  (prompt)
# Target: 11 reversed sum digits + EOS = 12 tokens
# Model input x: 22 prompt tokens + 11 target digits = 33 tokens
# Model target y: shifted by 1, prompt masked = 33 tokens
PROMPT_LEN = NUM_DIGITS + 1 + NUM_DIGITS + 1  # 22
TARGET_LEN = SUM_DIGITS + 1  # 12 (11 digits + EOS)
INPUT_LEN = PROMPT_LEN + SUM_DIGITS  # 33
FULL_LEN = PROMPT_LEN + TARGET_LEN  # 34

POW10_10 = torch.tensor([10**i for i in range(NUM_DIGITS)], dtype=torch.int64)
POW10_11 = torch.tensor([10**i for i in range(SUM_DIGITS)], dtype=torch.int64)


# ---------------------------------------------------------------------------
# Scalar preprocessing
# ---------------------------------------------------------------------------
def preprocess(a: int, b: int) -> List[int]:
    """Convert (a, b) -> prompt token IDs.

    Format: d9 d8 ... d0 + d9 d8 ... d0 =   (MSD first, zero-padded)
    """
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    prompt_str = a_str + "+" + b_str + "="
    return [TOKENS[ch] for ch in prompt_str]


def target_tokens(a: int, b: int) -> List[int]:
    """Reversed sum digits + EOS."""
    c = a + b
    c_str = str(c).zfill(SUM_DIGITS)
    c_rev = c_str[::-1]
    return [TOKENS[ch] for ch in c_rev] + [EOS_ID]


def postprocess(generated: Sequence[int]) -> int:
    """Convert reversed digit tokens back to integer."""
    digits: List[str] = []
    for tok in generated:
        tid = int(tok)
        if tid == EOS_ID:
            break
        if 0 <= tid <= 9:
            digits.append(str(tid))
        else:
            break
    if not digits:
        return 0
    while len(digits) < SUM_DIGITS:
        digits.append("0")
    digits = digits[:SUM_DIGITS]
    return int("".join(digits)[::-1])


# ---------------------------------------------------------------------------
# Vectorized batch encoding
# ---------------------------------------------------------------------------
def preprocess_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized: (a, b) int64 tensors -> prompt token tensor [bsz, PROMPT_LEN=22]."""
    bsz = a.shape[0]
    # Extract digits MSD-first: position j -> digit at 10^(NUM_DIGITS-1-j)
    powers = torch.tensor(
        [10 ** (NUM_DIGITS - 1 - j) for j in range(NUM_DIGITS)], dtype=torch.int64
    )
    a_digits = ((a[:, None] // powers[None, :]) % 10).to(torch.long)
    b_digits = ((b[:, None] // powers[None, :]) % 10).to(torch.long)

    plus = torch.full((bsz, 1), PLUS_ID, dtype=torch.long)
    eq = torch.full((bsz, 1), EQUALS_ID, dtype=torch.long)
    return torch.cat([a_digits, plus, b_digits, eq], dim=1)  # [bsz, 22]


def encode_batch(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build supervised LM tensors (x, y) with prompt labels masked."""
    prompt = preprocess_batch(a, b)  # [bsz, 22]
    sums = a + b
    # Reversed sum digits (LSD first)
    sum_digits = ((sums[:, None] // POW10_11[None, :]) % 10).to(torch.long)

    bsz = a.shape[0]
    eos = torch.full((bsz, 1), EOS_ID, dtype=torch.long)
    target = torch.cat([sum_digits, eos], dim=1)  # [bsz, 12]

    full = torch.cat([prompt, target], dim=1)  # [bsz, 34]
    x = full[:, :-1].clone()  # [bsz, 33]
    y = full[:, 1:].clone()   # [bsz, 33]
    y[:, : PROMPT_LEN - 1] = -100  # mask prompt tokens in labels
    return x, y


# ---------------------------------------------------------------------------
# Curriculum-aware batch sampling
# ---------------------------------------------------------------------------
def encode_curriculum_batch(
    a: torch.Tensor, b: torch.Tensor,
    min_digits: int, max_digits: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample operands within digit range and encode.

    For curriculum learning: operands have between min_digits and max_digits digits.
    """
    bsz = a.shape[0]
    # Resample within range
    for i in range(bsz):
        n_dig = torch.randint(min_digits, max_digits + 1, (1,), generator=generator).item()
        max_val = 10**n_dig
        a[i] = torch.randint(0, max_val, (1,), generator=generator, dtype=torch.int64).item()
        b[i] = torch.randint(0, max_val, (1,), generator=generator, dtype=torch.int64).item()
    return encode_batch(a, b)


# ---------------------------------------------------------------------------
# Holdout splits  (same logic as original)
# ---------------------------------------------------------------------------
def pair_hash(a: int, b: int) -> int:
    return a * MAX_OPERAND + b


def build_holdout_splits(
    val_size: int, test_size: int, seed: int, out_path: Path
) -> Dict[str, torch.Tensor]:
    if out_path.exists():
        data = torch.load(out_path, map_location="cpu", weights_only=False)
        if int(data["val_a"].numel()) == val_size and int(data["test_a"].numel()) == test_size:
            return data

    g = torch.Generator().manual_seed(seed)
    total = val_size + test_size
    pairs: List[Tuple[int, int]] = []
    seen = set()

    while len(pairs) < total:
        sample_n = max((total - len(pairs)) * 2, 4096)
        aa = torch.randint(0, MAX_OPERAND, (sample_n,), generator=g, dtype=torch.int64)
        bb = torch.randint(0, MAX_OPERAND, (sample_n,), generator=g, dtype=torch.int64)
        for ai, bi in zip(aa.tolist(), bb.tolist()):
            h = pair_hash(ai, bi)
            if h in seen:
                continue
            seen.add(h)
            pairs.append((ai, bi))
            if len(pairs) >= total:
                break

    data = {
        "val_a": torch.tensor([p[0] for p in pairs[:val_size]], dtype=torch.int64),
        "val_b": torch.tensor([p[1] for p in pairs[:val_size]], dtype=torch.int64),
        "test_a": torch.tensor([p[0] for p in pairs[val_size:]], dtype=torch.int64),
        "test_b": torch.tensor([p[1] for p in pairs[val_size:]], dtype=torch.int64),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return data
