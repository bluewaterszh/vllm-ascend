# dispatch_ffn_combine_v2

## Scope

This directory contains a standalone multi-rank executable for the int8 `dispatch_ffn_combine` path.

Current validation path:
- standalone ACL + HCCL + MPI direct launch
- deterministic non-zero `cpu-golden` data
- structured fp16 accuracy report
- warmup / measured perf statistics
- equivalent throughput / TFLOPS / GB/s derived from logical workload

Not covered:
- `bf16`
- `w4_a8`
- hardware-counter-based bandwidth or FLOPS accounting

## Build and run

Run from the repo root:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export MPI_ENV_BIN=/home/ntlab/miniconda3/envs/ltr_pto/bin
export MPI_ENV_LIB=/home/ntlab/miniconda3/envs/ltr_pto/lib
export MPI_LIB_PATH=${MPI_ENV_LIB}/libmpi.so
export MPI_RUNNER=mpirun

bash csrc/mc2/dispatch_ffn_combine_v2/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

## Exposed parameters

`run.sh` keeps only the case-shape and parallelism parameters:
- `--soc`
- `--world-size`
- `--m --k --n`
- `--topk`
- `--experts`
- `--max-output-size`

`run.sh` always uses the deterministic non-zero `cpu-golden` data path.

Internal fixed defaults used by `run.sh`:
- seed: `20260515`
- warmup iterations: `3`
- measure iterations: `5`
- compare `atol`: `1e-3`
- compare `rtol`: `1e-3`

`max-output-size` is the per-destination-rank routed-token capacity used by the standalone runner. It affects both truncation behavior and workspace sizing.

## Generated artifacts

- `out/case.json`
- `out/rank{rank}_expected_out.bin`
- `out/output_rank{rank}.bin`

## Runtime output

Per rank:
- structured accuracy summary (`mismatch_count`, `max_abs_err`, `max_rel_err`, `mean_abs_err`, `rmse`)
- final `PASS` / `FAIL`

Rank 0 summary:
- kernel time stats (`avg/min/max/std`)
- e2e time stats (`avg/min/max/std`)
- input / routed tokens per second
- equivalent TFLOPS
- equivalent GB/s

## Metric note

The perf summary uses logical workload stored in `case.json`:
- `input_tokens_all_ranks`
- `routed_tokens_all_ranks`
- `remote_routed_tokens_all_ranks`
- `compute_flops_all_ranks`
- `comm_bytes_all_ranks`

These are equivalent metrics derived from the routed workload. They are useful for steady-state comparison across cases, but they are not hardware counters and should not be interpreted as literal on-chip compute or link bandwidth measurements.
