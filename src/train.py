"""Training entrypoint for low-rank addition transformer.

Incorporates gpt-acc-jax techniques:
  - Curriculum learning (3 phases: 1-3 digits, 1-6 digits, 1-10 digits)
  - High learning rate (0.02) with cosine decay
  - 27K total training steps
  - AdamW optimizer with gradient clipping

Usage:
  python -m src.train --run-name lowrank_pos3 --pos-rank 3
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import (
    MAX_OPERAND,
    INPUT_LEN,
    VOCAB_SIZE,
    build_holdout_splits,
    encode_batch,
    pair_hash,
)
from src.eval import evaluate_exact_match
from src.model import ModelConfig, TinyDecoderLM, count_parameters


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class TrainConfig:
    seed: int
    train_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    grad_clip: float
    eval_interval: int
    val_size: int
    test_size: int
    eval_batch_size: int
    run_name: str
    run_dir: str
    split_dir: str
    best_ckpt_out: str
    last_ckpt_out: str
    device: str
    dtype: str = "fp32"  # fp32, fp16, bf16
    # Curriculum phases: list of (min_digits, max_digits, steps)
    curriculum: str = ""  # serialized; parsed below


CURRICULUM_PHASES = [
    (1, 3, 2000),    # Phase 1: easy
    (1, 6, 5000),    # Phase 2: medium
    (1, 10, 20000),  # Phase 3: full range
]
TOTAL_CURRICULUM_STEPS = sum(s for _, _, s in CURRICULUM_PHASES)  # 27000


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class CurriculumBatchSampler:
    """Samples training batches with curriculum learning and holdout avoidance."""

    def __init__(self, batch_size: int, seed: int, reserved_hashes: set,
                 curriculum_phases: List[Tuple[int, int, int]]):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes
        self.phases = curriculum_phases
        # Build cumulative step boundaries
        self.boundaries = []
        cum = 0
        for _, _, steps in self.phases:
            cum += steps
            self.boundaries.append(cum)

    def _phase_for_step(self, step: int) -> Tuple[int, int]:
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return self.phases[i][0], self.phases[i][1]
        return self.phases[-1][0], self.phases[-1][1]

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        min_dig, max_dig = self._phase_for_step(step)

        a = torch.zeros(self.batch_size, dtype=torch.int64)
        b = torch.zeros(self.batch_size, dtype=torch.int64)

        for i in range(self.batch_size):
            n_dig = int(torch.randint(min_dig, max_dig + 1, (1,), generator=self.g).item())
            max_val = 10 ** n_dig
            ai = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            bi = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, max_val, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi

        return encode_batch(a, b)


def cosine_lr(step: int, max_steps: int, base_lr: float,
              warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv_header(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    device = torch.device(train_cfg.device)
    dtype = DTYPE_MAP.get(train_cfg.dtype, torch.float32)
    run_dir = Path(train_cfg.run_dir)
    split_dir = Path(train_cfg.split_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    set_seed(train_cfg.seed)

    split_path = split_dir / f"holdout_v{train_cfg.val_size}_t{train_cfg.test_size}_seed{train_cfg.seed}.pt"
    splits = build_holdout_splits(train_cfg.val_size, train_cfg.test_size, train_cfg.seed, split_path)

    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))

    val_a, val_b = splits["val_a"], splits["val_b"]

    # Mixed precision: model stays fp32, autocast handles fp16/bf16 in forward pass
    # This keeps optimizer states (AdamW moments) in fp32 for numerical stability
    use_amp = (dtype != torch.float32 and device.type == "cuda")
    model = TinyDecoderLM(model_cfg).to(device=device)  # always fp32
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    sampler = CurriculumBatchSampler(
        train_cfg.batch_size, train_cfg.seed + 1337,
        reserved_hashes, CURRICULUM_PHASES,
    )

    save_csv_header(metrics_path,
                    ["step", "train_loss", "val_exact", "val_token_acc", "lr", "elapsed_sec"])

    best_val = -1.0
    best_step = -1
    t0 = time.time()

    # GradScaler for fp16 (not needed for bf16)
    use_scaler = (dtype == torch.float16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    print(f"Run: {train_cfg.run_name}")
    print(f"Params: {params}")
    print(f"Model config: {model_cfg}")
    print(f"Device: {device}, amp_dtype: {dtype}, use_amp: {use_amp}, scaler: {use_scaler}")
    print(f"Curriculum: {CURRICULUM_PHASES}")

    for step in range(train_cfg.train_steps):
        model.train()
        x, y = sampler.sample_batch(step)
        x, y = x.to(device), y.to(device)

        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr,
                           train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss = model(x, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
        else:
            _, loss = model(x, y)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
            val_exact, val_tok = evaluate_exact_match(
                model, val_a, val_b, train_cfg.eval_batch_size, device
            )
            elapsed = time.time() - t0
            train_loss = float(loss.item())
            append_csv(metrics_path, [step, train_loss, val_exact, val_tok, lr_now, elapsed])
            print(
                f"step={step:6d} loss={train_loss:.4f} val_exact={val_exact:.4f} "
                f"val_tok={val_tok:.5f} lr={lr_now:.2e} t={elapsed:.1f}s"
            )

            if val_exact > best_val:
                best_val = val_exact
                best_step = step
                Path(train_cfg.best_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "train_config": asdict(train_cfg),
                        "step": step,
                        "val_exact": val_exact,
                        "params": params,
                    },
                    train_cfg.best_ckpt_out,
                )

    Path(train_cfg.last_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "step": train_cfg.train_steps - 1,
            "val_exact": best_val,
            "params": params,
        },
        train_cfg.last_ckpt_out,
    )

    summary = {
        "run_name": train_cfg.run_name,
        "params": params,
        "best_val_exact": best_val,
        "best_step": best_step,
        "train_steps": train_cfg.train_steps,
        "elapsed_sec": time.time() - t0,
        "model_config": asdict(model_cfg),
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "config.json", {"model": asdict(model_cfg), "train": asdict(train_cfg)})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train low-rank addition transformer")

    # run/output
    p.add_argument("--run-name", type=str, default="lowrank_baseline")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="Training precision (fp32, fp16, bf16)")
    p.add_argument("--seed", type=int, default=42)

    # model (gpt-acc-jax defaults)
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=7)
    p.add_argument("--n-head", type=int, default=1)
    p.add_argument("--d-ff", type=int, default=14)
    p.add_argument("--dropout", type=float, default=0.0)
    # low-rank options
    p.add_argument("--pos-rank", type=int, default=0, help="Position embedding rank (0=full)")
    p.add_argument("--qkv-rank", type=int, default=0, help="QKV projection rank (0=full)")
    p.add_argument("--attn-out-rank", type=int, default=0, help="Attn output rank (0=full)")
    p.add_argument("--ffn-rank", type=int, default=0, help="FFN rank (0=full)")
    # normalization / tying
    p.add_argument("--use-rmsnorm", action="store_true", default=False,
                   help="Use RMSNorm instead of LayerNorm (saves d_model params per norm)")
    p.add_argument("--tie-qkv", type=str, default="none",
                   choices=["none", "all", "qk", "kv", "shareA", "shareB",
                            "shareB_tieQK", "shareB_tieKV", "shareA_tieKV", "shareA_tieQK"],
                   help="QKV tying mode")


    # optimization (gpt-acc-jax defaults)
    p.add_argument("--train-steps", type=int, default=27000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1350)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=1000)

    # eval
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--eval-batch-size", type=int, default=512)

    args = p.parse_args()

    if args.run_dir is None:
        args.run_dir = Path(f"results/runs/{args.run_name}")

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=INPUT_LEN,
        vocab_size=VOCAB_SIZE,
        pos_rank=args.pos_rank,
        qkv_rank=args.qkv_rank,
        attn_out_rank=args.attn_out_rank,
        ffn_rank=args.ffn_rank,
        use_rmsnorm=args.use_rmsnorm,
        tie_qkv=args.tie_qkv,
    )

    run_dir = Path(args.run_dir)
    train_cfg = TrainConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        val_size=args.val_size,
        test_size=args.test_size,
        eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        run_dir=str(run_dir),
        split_dir=str(args.split_dir),
        best_ckpt_out=str(run_dir / "checkpoints" / "best.pt"),
        last_ckpt_out=str(run_dir / "checkpoints" / "last.pt"),
        device=args.device,
        dtype=args.dtype,
    )

    summary = train(model_cfg, train_cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
