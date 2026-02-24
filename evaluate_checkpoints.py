"""Generate final results JSON for a checkpoint.

Usage:
    python generate_results.py checkpoints/best_512params.pt
    python generate_results.py checkpoints/best_582params.pt
"""

import argparse
import json
from pathlib import Path

import torch

from src.data import build_holdout_splits
from src.eval import evaluate_exact_match
from src.model import ModelConfig, TinyDecoderLM, count_parameters

SEEDS = [41, 100, 200, 300, 400, 500, 999, 1234, 7777, 31415]
TEST_SIZE = 10000
VAL_SIZE = 5000


def main():
    parser = argparse.ArgumentParser(description="Generate final results for a checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: results/final_<params>params.json)")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = TinyDecoderLM(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params = count_parameters(model)
    tied = model.lm_head.weight is model.token_emb.weight
    print(f"Model: {params} parameters, weight_tying={tied}")
    print(f"Config: {cfg}")
    print()

    # Multi-seed validation (seed=42 serves as both primary and part of multi-seed)
    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed validation ({len(SEEDS)} seeds x {TEST_SIZE} examples = {len(SEEDS) * TEST_SIZE} total):")
    per_seed = {}
    total_errors = 0
    for seed in SEEDS:
        p = data_dir / f"holdout_v{VAL_SIZE}_t{TEST_SIZE}_seed{seed}.pt"
        sp = build_holdout_splits(VAL_SIZE, TEST_SIZE, seed, p)
        em_s, _ = evaluate_exact_match(model, sp["test_a"], sp["test_b"], args.batch_size, args.device)
        errs = int(round((1 - em_s) * TEST_SIZE))
        total_errors += errs
        per_seed[str(seed)] = {"exact_match": em_s, "errors": errs}
        status = "PASS" if errs == 0 else f"FAIL ({errs} errors)"
        print(f"  seed={seed:>5}: exact_match={em_s:.6f}  {status}")

    total_examples = len(SEEDS) * TEST_SIZE
    aggregate_em = 1 - total_errors / total_examples
    print(f"\nAggregate: {aggregate_em:.6f} ({total_errors} errors in {total_examples} examples)")

    # Build result dict
    result = {
        "model": f"{params}-parameter low-rank transformer",
        "checkpoint": args.checkpoint,
        "parameters": params,
        "config": {
            "n_layer": cfg.n_layer,
            "d_model": cfg.d_model,
            "n_head": cfg.n_head,
            "d_ff": cfg.d_ff,
            "pos_rank": cfg.pos_rank,
            "qkv_rank": cfg.qkv_rank,
            "attn_out_rank": cfg.attn_out_rank,
            "ffn_rank": cfg.ffn_rank,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "weight_tying": tied,
            "precision": "fp32",
        },
        "training": {
            "seed": ckpt.get("train_config", {}).get("seed", 42),
            "total_steps": ckpt.get("step", "unknown"),
        },
        "multi_seed_validation": {
            "total_examples": total_examples,
            "num_test_sets": len(SEEDS),
            "aggregate_exact_match": round(aggregate_em, 6),
            "aggregate_errors": total_errors,
            "per_seed": per_seed,
        },
    }

    # Write output
    out_path = args.output or f"results/final_{params}params.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
