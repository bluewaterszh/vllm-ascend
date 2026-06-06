#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def fp32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def fp32_to_fp16_bits(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).view(np.uint16)


def fp32_to_bf16_value(arr: np.ndarray) -> np.ndarray:
    bits = fp32_to_bf16_bits(arr).astype(np.uint32)
    return (bits << 16).view(np.float32)


def fp32_to_fp16_value(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).astype(np.float32)


def cast_fp16_value(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).astype(np.float32)


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)


def pack_scale_fp32_to_int64(scale_origin: np.ndarray) -> np.ndarray:
    rows, cols = scale_origin.shape
    packed = np.zeros((rows, cols), dtype=np.int64)
    packed_view = packed.view(np.float32).reshape(rows, cols * 2)
    packed_view[:, ::2] = scale_origin.astype(np.float32)
    return packed


def pack_weight_to_zn_int8(weight: np.ndarray) -> np.ndarray:
    k, n = weight.shape
    c0_k = 16
    c0_n = 32
    k_align = ((k + c0_k - 1) // c0_k) * c0_k
    n_align = ((n + c0_n - 1) // c0_n) * c0_n
    n_loop = n_align // c0_n
    padded = np.zeros((k_align, n_align), dtype=np.int8)
    padded[:k, :n] = weight.astype(np.int8)
    return padded.reshape(k_align, n_loop, c0_n).transpose(1, 0, 2).copy().reshape(-1)


def pack_expert_weights_to_zn(weights: np.ndarray) -> np.ndarray:
    packed = [pack_weight_to_zn_int8(weights[expert_idx]) for expert_idx in range(weights.shape[0])]
    return np.concatenate(packed, axis=0).astype(np.int8)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray) -> np.ndarray:
    x0, gate = np.split(x, 2, axis=-1)
    return (x0 * sigmoid(x0)) * gate


def quantize_init_routing_to_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    row_max = np.max(np.abs(x), axis=-1, keepdims=True)
    safe_row_max = np.where(row_max == 0.0, 1.0, row_max)
    normalized = x * 127.0 / safe_row_max
    quant = np.rint(cast_fp16_value(normalized))
    quant = np.where(row_max == 0.0, 0.0, quant)
    quant = np.clip(quant, -128.0, 127.0).astype(np.int8)
    scale = np.where(row_max[:, 0] == 0.0, 0.0, row_max[:, 0] / 127.0).astype(np.float32)
    return quant, scale


def quantize_swiglu_to_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    row_max = np.max(np.abs(x), axis=-1, keepdims=True)
    safe_row_max = np.where(row_max == 0.0, 1.0, row_max)
    normalized = x * 127.0 / safe_row_max
    quant = np.rint(normalized)
    quant = np.where(row_max == 0.0, 0.0, quant)
    quant = np.clip(quant, -128.0, 127.0).astype(np.int8)
    scale = np.where(row_max[:, 0] == 0.0, 0.0, row_max[:, 0] / 127.0).astype(np.float32)
    return quant, scale


def make_x(rank: int, args: argparse.Namespace) -> np.ndarray:
    base = np.arange(args.m * args.k, dtype=np.float32).reshape(args.m, args.k)
    x = 0.75 * np.sin((base + rank * 17.0) / 23.0) + 0.05 * np.cos((base % 29.0) / 11.0)
    return x.astype(np.float32)


def make_probs(rank: int, args: argparse.Namespace) -> np.ndarray:
    topk_weights = np.arange(args.topk, 0, -1, dtype=np.float32)[None, :]
    token_offset = (np.arange(args.m, dtype=np.float32)[:, None] % max(args.topk, 1)) * 0.05
    probs = topk_weights + token_offset + rank * 0.01
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def make_expert_idx(rank: int, args: argparse.Namespace) -> np.ndarray:
    total_experts = args.world_size * args.experts
    if total_experts <= 0:
        raise ValueError("total_experts must be positive")
    base = (np.arange(args.m, dtype=np.int32)[:, None] * args.topk + rank + np.arange(args.topk, dtype=np.int32)[None, :])
    return (base % total_experts).astype(np.int32)


def make_scale_origin(experts: int, channels: int, offset: int, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((experts, channels), dtype=np.float32)
    idx = np.arange(experts * channels, dtype=np.float32).reshape(experts, channels)
    return ((1.0 / 16.0) * (1.0 + ((idx + offset) % 5.0) / 8.0)).astype(np.float32)


def make_weight1(rank: int, args: argparse.Namespace, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((args.experts, args.k, args.n), dtype=np.int8)
    base = np.arange(args.experts * args.k * args.n, dtype=np.int32).reshape(args.experts, args.k, args.n)
    weight = ((base + rank * 13 + 7) % 7) - 3
    return weight.astype(np.int8)


def make_weight2(rank: int, args: argparse.Namespace, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((args.experts, args.n // 2, args.k), dtype=np.int8)
    base = np.arange(args.experts * (args.n // 2) * args.k, dtype=np.int32).reshape(args.experts, args.n // 2, args.k)
    weight = ((base + rank * 19 + 11) % 9) - 4
    return weight.astype(np.int8)


def compute_outputs_and_workload(
    xs: list[np.ndarray],
    weight1_nd: list[np.ndarray],
    weight2_nd: list[np.ndarray],
    scale1_origin: list[np.ndarray],
    scale2_origin: list[np.ndarray],
    expert_idx_list: list[np.ndarray],
    probs_list: list[np.ndarray],
    x_active_mask_list: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], dict[str, float]]:
    outputs = [np.zeros((args.m, args.k), dtype=np.float32) for _ in range(args.world_size)]
    total_routed_tokens = 0.0
    total_remote_routed_tokens = 0.0
    total_input_tokens = float(sum(int(mask.sum()) for mask in x_active_mask_list))

    for dst_rank in range(args.world_size):
        kept_tokens = 0
        for local_expert in range(args.experts):
            global_expert = dst_rank * args.experts + local_expert
            for src_rank in range(args.world_size):
                for token_idx in range(args.m):
                    if x_active_mask_list[src_rank][token_idx] == 0:
                        continue
                    for topk_idx in range(args.topk):
                        if int(expert_idx_list[src_rank][token_idx, topk_idx]) != global_expert:
                            continue
                        if args.max_output_size > 0 and kept_tokens >= args.max_output_size:
                            continue

                        x_token = xs[src_rank][token_idx:token_idx + 1, :]
                        qx, per_token_scale1 = quantize_init_routing_to_int8(x_token)
                        product1 = qx.astype(np.int32) @ weight1_nd[dst_rank][local_expert].astype(np.int32)
                        gmm1_half = cast_fp16_value(
                            product1.astype(np.float32) * scale1_origin[dst_rank][local_expert][None, :]
                        )
                        dequant1 = gmm1_half * per_token_scale1[:, None]

                        swiglu_out = swiglu(dequant1)[0]
                        qswiglu, per_token_scale2 = quantize_swiglu_to_int8(swiglu_out[None, :])
                        product2 = qswiglu.astype(np.int32) @ weight2_nd[dst_rank][local_expert].astype(np.int32)
                        gmm2_half = cast_fp16_value(
                            product2.astype(np.float32) * scale2_origin[dst_rank][local_expert][None, :]
                        )
                        result = cast_fp16_value(gmm2_half * per_token_scale2[:, None])[0]

                        outputs[src_rank][token_idx, :] += probs_list[src_rank][token_idx, topk_idx] * result
                        kept_tokens += 1
                        total_routed_tokens += 1.0
                        if src_rank != dst_rank:
                            total_remote_routed_tokens += 1.0

    workload = {
        "input_tokens_all_ranks": total_input_tokens,
        "routed_tokens_all_ranks": total_routed_tokens,
        "remote_routed_tokens_all_ranks": total_remote_routed_tokens,
        "compute_flops_all_ranks": total_routed_tokens * 3.0 * args.k * args.n,
        "comm_bytes_all_ranks": total_remote_routed_tokens
        * (args.k * (np.dtype(np.int8).itemsize + np.dtype(np.float16).itemsize) + np.dtype(np.float32).itemsize),
    }
    return outputs, workload


def compute_workload_only(
    expert_idx_list: list[np.ndarray],
    x_active_mask_list: list[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float]:
    total_routed_tokens = 0.0
    total_remote_routed_tokens = 0.0
    total_input_tokens = float(sum(int(mask.sum()) for mask in x_active_mask_list))

    for dst_rank in range(args.world_size):
        kept_tokens = 0
        for local_expert in range(args.experts):
            global_expert = dst_rank * args.experts + local_expert
            for src_rank in range(args.world_size):
                active_mask = x_active_mask_list[src_rank]
                expert_idx = expert_idx_list[src_rank]
                for token_idx in range(args.m):
                    if active_mask[token_idx] == 0:
                        continue
                    for topk_idx in range(args.topk):
                        if int(expert_idx[token_idx, topk_idx]) != global_expert:
                            continue
                        if args.max_output_size > 0 and kept_tokens >= args.max_output_size:
                            continue
                        kept_tokens += 1
                        total_routed_tokens += 1.0
                        if src_rank != dst_rank:
                            total_remote_routed_tokens += 1.0

    return {
        "input_tokens_all_ranks": total_input_tokens,
        "routed_tokens_all_ranks": total_routed_tokens,
        "remote_routed_tokens_all_ranks": total_remote_routed_tokens,
        "compute_flops_all_ranks": total_routed_tokens * 3.0 * args.k * args.n,
        "comm_bytes_all_ranks": total_remote_routed_tokens
        * (args.k * (np.dtype(np.int8).itemsize + np.dtype(np.float16).itemsize) + np.dtype(np.float32).itemsize),
    }


def build_rank_case(
    rank: int,
    args: argparse.Namespace,
    out_dir: Path,
    xs: list[np.ndarray],
    weight1_nd: list[np.ndarray],
    weight2_nd: list[np.ndarray],
    scale1_origin: list[np.ndarray],
    scale2_origin: list[np.ndarray],
    expert_idx_list: list[np.ndarray],
    probs_list: list[np.ndarray],
    x_active_mask_list: list[np.ndarray],
    expected_out_list: Optional[list[np.ndarray]],
) -> None:
    weight1_packed = pack_expert_weights_to_zn(weight1_nd[rank])
    weight2_packed = pack_expert_weights_to_zn(weight2_nd[rank])
    scale1_packed = pack_scale_fp32_to_int64(scale1_origin[rank])
    scale2_packed = pack_scale_fp32_to_int64(scale2_origin[rank])

    write(out_dir / f"rank{rank}_x.bin", fp32_to_fp16_bits(xs[rank]))
    write(out_dir / f"rank{rank}_weight1.bin", weight1_packed)
    write(out_dir / f"rank{rank}_weight2.bin", weight2_packed)
    write(out_dir / f"rank{rank}_expert_idx.bin", expert_idx_list[rank].astype(np.int32))
    write(out_dir / f"rank{rank}_scale1.bin", scale1_packed)
    write(out_dir / f"rank{rank}_scale2.bin", scale2_packed)
    write(out_dir / f"rank{rank}_probs.bin", probs_list[rank].astype(np.float32))
    write(out_dir / f"rank{rank}_x_active_mask.bin", x_active_mask_list[rank].astype(np.uint8))
    expected_out_path = out_dir / f"rank{rank}_expected_out.bin"
    if expected_out_list is not None:
        write(expected_out_path, fp32_to_fp16_bits(expected_out_list[rank]))
    elif expected_out_path.exists():
        expected_out_path.unlink()


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
    parser.add_argument("--case-mode", choices=["zero", "cpu-golden"], default="cpu-golden")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_golden:
        for stale_path in list(out_dir.glob("rank*_expected_out.bin")) + list(out_dir.glob("output_rank*.bin")):
            stale_path.unlink()

    xs = [make_x(rank, args) for rank in range(args.world_size)]
    probs_list = [make_probs(rank, args) for rank in range(args.world_size)]
    expert_idx_list = [make_expert_idx(rank, args) for rank in range(args.world_size)]
    x_active_mask_list = [np.ones((args.m,), dtype=np.uint8) for _ in range(args.world_size)]
    weight1_nd = [make_weight1(rank, args, args.case_mode) for rank in range(args.world_size)]
    weight2_nd = [make_weight2(rank, args, args.case_mode) for rank in range(args.world_size)]
    scale1_origin = [make_scale_origin(args.experts, args.n, rank * 17, args.case_mode) for rank in range(args.world_size)]
    scale2_origin = [make_scale_origin(args.experts, args.k, rank * 23, args.case_mode) for rank in range(args.world_size)]

    xs = [fp32_to_fp16_value(x) for x in xs]
    if args.skip_golden:
        expected_out_list = None
        workload = compute_workload_only(expert_idx_list, x_active_mask_list, args)
    else:
        expected_out_list, workload = compute_outputs_and_workload(
            xs,
            weight1_nd,
            weight2_nd,
            scale1_origin,
            scale2_origin,
            expert_idx_list,
            probs_list,
            x_active_mask_list,
            args,
        )

    case_json = {
        "world_size": args.world_size,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "topk": args.topk,
        "expert_per_rank": args.experts,
        "max_output_size": args.max_output_size,
        "case_mode": args.case_mode,
        "skip_golden": args.skip_golden,
        "seed": args.seed,
        "compare_atol": args.atol,
        "compare_rtol": args.rtol,
        **workload,
    }
    (out_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")

    for rank in range(args.world_size):
        build_rank_case(
            rank,
            args,
            out_dir,
            xs,
            weight1_nd,
            weight2_nd,
            scale1_origin,
            scale2_origin,
            expert_idx_list,
            probs_list,
            x_active_mask_list,
            expected_out_list,
        )


if __name__ == "__main__":
    main()
