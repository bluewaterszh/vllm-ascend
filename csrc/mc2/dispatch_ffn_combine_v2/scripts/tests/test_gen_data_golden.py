#!/usr/bin/env python3
import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import gen_data  # noqa: E402


def make_args(case_mode: str = "cpu-golden") -> argparse.Namespace:
    return argparse.Namespace(
        world_size=2,
        m=16,
        k=128,
        n=128,
        topk=2,
        experts=2,
        max_output_size=32,
        case_mode=case_mode,
        seed=20260515,
        atol=1e-3,
        rtol=1e-3,
    )


def make_inputs(args: argparse.Namespace):
    xs = [gen_data.fp32_to_fp16_value(gen_data.make_x(rank, args)) for rank in range(args.world_size)]
    probs = [gen_data.make_probs(rank, args) for rank in range(args.world_size)]
    expert_idx = [gen_data.make_expert_idx(rank, args) for rank in range(args.world_size)]
    active_mask = [np.ones((args.m,), dtype=np.uint8) for _ in range(args.world_size)]
    weight1 = [gen_data.make_weight1(rank, args, args.case_mode) for rank in range(args.world_size)]
    weight2 = [gen_data.make_weight2(rank, args, args.case_mode) for rank in range(args.world_size)]
    scale1 = [
        gen_data.make_scale_origin(args.experts, args.n, rank * 17, args.case_mode)
        for rank in range(args.world_size)
    ]
    scale2 = [
        gen_data.make_scale_origin(args.experts, args.k, rank * 23, args.case_mode)
        for rank in range(args.world_size)
    ]
    return xs, weight1, weight2, scale1, scale2, expert_idx, probs, active_mask


class GenDataGoldenTest(unittest.TestCase):
    def test_batch_golden_matches_naive_for_cpu_golden_case(self) -> None:
        args = make_args("cpu-golden")

        naive_out, naive_workload = gen_data.compute_outputs_naive_and_workload(*make_inputs(args), args)
        batch_out, batch_workload = gen_data.compute_outputs_and_workload(*make_inputs(args), args)

        self.assertEqual(batch_workload, naive_workload)
        for rank in range(args.world_size):
            np.testing.assert_array_equal(batch_out[rank], naive_out[rank])

    def test_batch_golden_matches_naive_for_zero_case(self) -> None:
        args = make_args("zero")

        naive_out, naive_workload = gen_data.compute_outputs_naive_and_workload(*make_inputs(args), args)
        batch_out, batch_workload = gen_data.compute_outputs_and_workload(*make_inputs(args), args)

        self.assertEqual(batch_workload, naive_workload)
        for rank in range(args.world_size):
            np.testing.assert_array_equal(batch_out[rank], naive_out[rank])


if __name__ == "__main__":
    unittest.main()
