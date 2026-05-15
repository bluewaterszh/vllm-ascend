#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def fp32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def build_rank_case(rank: int, args: argparse.Namespace, out_dir: Path) -> None:
    total_experts = args.world_size * args.experts
    x = fp32_to_bf16_bits(np.arange(args.m * args.k, dtype=np.float32).reshape(args.m, args.k) / 256.0)
    weight1 = np.zeros((args.experts, args.k, args.n), dtype=np.int8)
    weight2 = np.zeros((args.experts, args.n // 2, args.k), dtype=np.int8)
    expert_idx = (np.arange(args.m * args.topk, dtype=np.int32).reshape(args.m, args.topk) + rank) % total_experts
    probs = np.zeros((args.m, args.topk), dtype=np.float32)
    probs[:, 0] = 1.0
    scale1 = np.zeros((args.experts, args.n), dtype=np.int64)
    scale2 = np.zeros((args.experts, args.k), dtype=np.int64)
    x_active_mask = np.ones((args.m,), dtype=np.uint8)
    expected_out = np.zeros((args.m, args.k), dtype=np.uint16)

    write(out_dir / f"rank{rank}_x.bin", x)
    write(out_dir / f"rank{rank}_weight1.bin", weight1)
    write(out_dir / f"rank{rank}_weight2.bin", weight2)
    write(out_dir / f"rank{rank}_expert_idx.bin", expert_idx)
    write(out_dir / f"rank{rank}_scale1.bin", scale1)
    write(out_dir / f"rank{rank}_scale2.bin", scale2)
    write(out_dir / f"rank{rank}_probs.bin", probs)
    write(out_dir / f"rank{rank}_x_active_mask.bin", x_active_mask)
    write(out_dir / f"rank{rank}_expected_out.bin", expected_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--max-output-size", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_json = {
        "world_size": args.world_size,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "topk": args.topk,
        "expert_per_rank": args.experts,
        "max_output_size": args.max_output_size,
    }
    (out_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")
    for rank in range(args.world_size):
        build_rank_case(rank, args, out_dir)


if __name__ == "__main__":
    main()
