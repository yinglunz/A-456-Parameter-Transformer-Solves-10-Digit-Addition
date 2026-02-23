"""Evaluation for the low-rank addition transformer."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import (
    MAX_OPERAND,
    SUM_DIGITS,
    TARGET_LEN,
    POW10_11,
    ITOS,
    build_holdout_splits,
    postprocess,
    preprocess,
    preprocess_batch,
)
from src.model import ModelConfig, TinyDecoderLM


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> TinyDecoderLM:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**blob["model_config"])
    model = TinyDecoderLM(mcfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_exact_match(
    model: TinyDecoderLM,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    n = int(a.numel())
    exact = 0
    token_correct = 0
    token_total = n * SUM_DIGITS

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        aa = a[start:end]
        bb = b[start:end]

        prompt_t = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt_t, max_new_tokens=TARGET_LEN)
        pred_digits = gen[:, -TARGET_LEN:-1].to("cpu")  # 11 sum digits

        tgt_digits = ((aa + bb)[:, None] // POW10_11[None, :]) % 10
        tgt_digits = tgt_digits.to(torch.long)

        matches = pred_digits.eq(tgt_digits)
        token_correct += int(matches.sum().item())
        exact += int(matches.all(dim=1).sum().item())

    return exact / n, token_correct / token_total


@torch.no_grad()
def collect_failures(
    model: TinyDecoderLM,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
    limit: int = 10,
) -> List[Dict]:
    model.eval()
    fails: List[Dict] = []
    n = int(a.numel())

    for start in range(0, n, batch_size):
        if len(fails) >= limit:
            break
        end = min(start + batch_size, n)
        aa = a[start:end]
        bb = b[start:end]

        prompt_t = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt_t, max_new_tokens=TARGET_LEN)
        pred_tail = gen[:, -TARGET_LEN:].to("cpu")
        pred_digits = pred_tail[:, :SUM_DIGITS]

        tgt_digits = ((aa + bb)[:, None] // POW10_11[None, :]) % 10
        tgt_digits = tgt_digits.to(torch.long)

        mismatch = ~pred_digits.eq(tgt_digits).all(dim=1)
        for bi in torch.nonzero(mismatch, as_tuple=False).flatten().tolist():
            ai_val = int(aa[bi].item())
            bi_val = int(bb[bi].item())
            pred_num = postprocess(pred_tail[bi].tolist())
            fails.append({
                "A": str(ai_val),
                "B": str(bi_val),
                "prediction": str(pred_num),
                "ground_truth": str(ai_val + bi_val),
            })
            if len(fails) >= limit:
                break

    return fails


def run_test(
    ckpt_path: Path,
    split_dir: Path,
    seed: int,
    val_size: int,
    test_size: int,
    eval_batch: int,
    device: str,
    out_json: Path,
) -> Dict:
    dev = torch.device(device)
    splits_path = split_dir / f"holdout_v{val_size}_t{test_size}_seed{seed}.pt"
    splits = build_holdout_splits(val_size, test_size, seed, splits_path)

    test_a = splits["test_a"]
    test_b = splits["test_b"]

    model = load_model_from_ckpt(ckpt_path, dev)
    em, tok_acc = evaluate_exact_match(model, test_a, test_b, eval_batch, dev)

    results = {
        "checkpoint": str(ckpt_path),
        "test_size": int(test_a.numel()),
        "exact_match": em,
        "token_accuracy": tok_acc,
        "failure_samples": collect_failures(model, test_a, test_b, eval_batch, dev),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("test")
    pt.add_argument("--ckpt", type=Path, required=True)
    pt.add_argument("--split-dir", type=Path, default=Path("results/data"))
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--val-size", type=int, default=5000)
    pt.add_argument("--test-size", type=int, default=10000)
    pt.add_argument("--eval-batch-size", type=int, default=512)
    pt.add_argument("--device", type=str, default="cpu")
    pt.add_argument("--out-json", type=Path, default=Path("results/final_results.json"))

    pp = sub.add_parser("predict")
    pp.add_argument("--ckpt", type=Path, required=True)
    pp.add_argument("--a", type=int, required=True)
    pp.add_argument("--b", type=int, required=True)
    pp.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.cmd == "test":
        result = run_test(
            args.ckpt, args.split_dir, args.seed, args.val_size, args.test_size,
            args.eval_batch_size, args.device, args.out_json,
        )
        print(json.dumps(result, indent=2))
    elif args.cmd == "predict":
        dev = torch.device(args.device)
        model = load_model_from_ckpt(args.ckpt, dev)
        prompt = torch.tensor([preprocess(args.a, args.b)], dtype=torch.long, device=dev)
        gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
        pred = postprocess(gen[0, -TARGET_LEN:].tolist())
        print(json.dumps({
            "A": args.a, "B": args.b,
            "prediction": pred, "ground_truth": args.a + args.b,
            "correct": int(pred == args.a + args.b),
        }, indent=2))


if __name__ == "__main__":
    main()
