#!/usr/bin/env python3
import argparse
import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import gen_data  # noqa: E402


def make_args(world_size: int, m: int, topk: int = 8, experts: int = 16) -> argparse.Namespace:
    return argparse.Namespace(
        world_size=world_size,
        m=m,
        k=7168,
        n=4096,
        topk=topk,
        experts=experts,
        max_output_size=81940,
        case_mode="cpu-golden",
        seed=20260515,
        atol=1e-3,
        rtol=1e-3,
    )


def route_counts(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    expert_counts: Counter[int] = Counter()
    dst_rank_counts = [0 for _ in range(args.world_size)]
    for rank in range(args.world_size):
        expert_idx = gen_data.make_expert_idx(rank, args)
        for expert in expert_idx.reshape(-1):
            expert = int(expert)
            expert_counts[expert] += 1
            dst_rank_counts[expert // args.experts] += 1
    total_experts = args.world_size * args.experts
    return [expert_counts[i] for i in range(total_experts)], dst_rank_counts


class GenDataDistributionTest(unittest.TestCase):
    def test_global_token_round_robin_assigns_consecutive_topk_experts(self) -> None:
        args = make_args(world_size=16, m=16)

        rank0 = gen_data.make_expert_idx(0, args)
        rank1 = gen_data.make_expert_idx(1, args)

        self.assertEqual(rank0[0].tolist(), list(range(0, 8)))
        self.assertEqual(rank0[1].tolist(), list(range(8, 16)))
        self.assertEqual(rank0[2].tolist(), list(range(16, 24)))
        self.assertEqual(rank1[0].tolist(), list(range(128, 136)))

    def test_global_token_round_robin_balances_16_rank_m16_case(self) -> None:
        args = make_args(world_size=16, m=16)

        expert_counts, dst_rank_counts = route_counts(args)

        self.assertEqual(set(expert_counts), {8})
        self.assertEqual(dst_rank_counts, [128] * 16)

    def test_global_token_round_robin_balances_target_8_rank_cases(self) -> None:
        for m in (16, 128, 2048):
            with self.subTest(m=m):
                args = make_args(world_size=8, m=m)

                expert_counts, dst_rank_counts = route_counts(args)

                self.assertEqual(len(set(expert_counts)), 1)
                self.assertEqual(len(set(dst_rank_counts)), 1)


if __name__ == "__main__":
    unittest.main()
