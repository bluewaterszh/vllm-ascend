# dispatch_ffn_combine_v3

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

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
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

## Stage 1 reference A/B

Reference commits:
- baseline before PTO-style HCCL context: `f7bf62cb`
- Stage 1 HCCL context / shmem migration: `5db0ebc7`

Reference case:

```bash
bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

If the baseline commit is checked out in a detached worktree, initialize its third-party submodules first.

Observed results:

| version | run | kernel avg (us) | e2e avg (us) | accuracy |
| --- | --- | ---: | ---: | --- |
| baseline `f7bf62cb` | 1 | 32.08 | 117.14 | PASS |
| baseline `f7bf62cb` | 2 | 35.79 | 118.61 | PASS |
| Stage 1 `5db0ebc7` | 1 | 42.31 | 134.34 | PASS |
| Stage 1 `5db0ebc7` | 2 | 42.00 | 132.98 | PASS |

Averages across the two runs:
- baseline: kernel `33.94 us`, e2e `117.88 us`
- Stage 1: kernel `42.16 us`, e2e `133.66 us`
- delta vs baseline: kernel `+24.2%`, e2e `+13.4%`

Use this as the current performance reference before Stage 2 communication-primitive migration.

## Current v2/v3 reference on the small case

Reference command:

```bash
bash csrc/mc2/dispatch_ffn_combine_v2/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

Observed results:

| version | kernel avg (us) | e2e avg (us) | input tokens/s | routed tokens/s | eq compute (TFLOPS) | eq comm (GB/s) | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v2 | 31.07 | 138.24 | 1029998.69 | 2059997.39 | 0.10 | 0.37 | PASS |
| current v3 | 27.26 | 97.78 | 1174053.41 | 2348106.82 | 0.12 | 0.42 | PASS |

Delta of current v3 vs v2:
- kernel: `-12.3%`
- e2e: `-29.3%`
- input throughput: `+14.0%`
- routed throughput: `+14.0%`

This is the latest same-shape reference after the AICORE-only cleanup.

## Stage 3a bring-up reference

Current status:
- Stage 3a has switched the core L0 matmul primitive in `BlockMmad` to PTO `TMATMUL / TMATMUL_ACC`.
- The current goal is functional bring-up first: keep accuracy and existing output contracts stable before tuning performance.
- Both the small case and the `CombineV1` large case still pass after the Stage 3a matmul swap.

Reference commands:

```bash
bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 2049 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 4098
```

Observed results vs the pre-Stage-3a v3 baseline:

| case | version | kernel avg (us) | e2e avg (us) | accuracy |
| --- | --- | ---: | ---: | --- |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | pre-Stage-3a v3 | 27.26 | 97.78 | PASS |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | Stage 3a bring-up | 43.79 | 145.53 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | pre-Stage-3a v3 | 29.60 | 119.29 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | Stage 3a bring-up | 49.01 | 153.74 | PASS |

Delta of Stage 3a bring-up vs the pre-Stage-3a v3 baseline:
- small case: kernel `+60.6%`, e2e `+48.8%`
- large case: kernel `+65.6%`, e2e `+28.9%`

Interpretation:
- Functionality is already open: the PTO matmul seam produces correct outputs on both validation cases.
- Performance is not yet acceptable; the current regression is expected follow-up work for the next tuning step.

## Stage 3a store-shell follow-up

Current status:
- Stage 3a now also aligns the per-channel scale staging and accumulator store path with the PTO-style shell.
- The matmul seam and output contracts remain unchanged: both reference cases still PASS.
- This follow-up recovers the large bring-up regression while keeping the Stage 3a functionality open.

Reference commands:

```bash
bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 2049 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 4098
```

Observed results:

| case | version | kernel avg (us) | e2e avg (us) | accuracy |
| --- | --- | ---: | ---: | --- |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | Stage 3a bring-up | 43.79 | 145.53 | PASS |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | Stage 3a store-shell follow-up | 23.68 | 101.54 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | Stage 3a bring-up | 49.01 | 153.74 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | Stage 3a store-shell follow-up | 28.10 | 117.65 | PASS |

Delta of the store-shell follow-up vs the Stage 3a bring-up:
- small case: kernel `-45.9%`, e2e `-30.2%`
- large case: kernel `-42.7%`, e2e `-23.5%`

Delta of the store-shell follow-up vs the pre-Stage-3a v3 baseline:
- small case: kernel `-13.1%`, e2e `+3.8%`
- large case: kernel `-5.1%`, e2e `-1.4%`

Interpretation:
- The Stage 3a bring-up regression was mainly in the scale/fixpipe/store shell, not in the PTO matmul seam alone.
- After aligning that shell with the PTO-style path, the large case is back to essentially baseline-level performance and the small case kernel time is now lower than the pre-Stage-3a v3 baseline.

## Stage 3b first safe seam

Current status:
- A first Stage 3b attempt in the SwiGLU epilogue path was rolled back because that seam caused a clear regression.
- The current kept seam is smaller: `BlockEpilogue3` now loads the per-token scale vector with PTO `TLOAD`, while the main C/D data path stays on the existing shell.
- Both reference cases still PASS after this change.

Reference commands:

```bash
bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 2049 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 4098
```

Observed results:

| case | version | kernel avg (us) | e2e avg (us) | accuracy |
| --- | --- | ---: | ---: | --- |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | Stage 3a store-shell follow-up | 23.68 | 101.54 | PASS |
| small `m=16, k=128, n=128, topk=2, max_output_size=32` | Stage 3b first safe seam | 31.53 | 124.67 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | Stage 3a store-shell follow-up | 28.10 | 117.65 | PASS |
| large `m=2049, k=128, n=128, topk=2, max_output_size=4098` | Stage 3b first safe seam | 30.04 | 116.40 | PASS |

Delta of the first safe seam vs the Stage 3a store-shell follow-up:
- small case: kernel `+33.2%`, e2e `+22.8%`
- large case: kernel `+6.9%`, e2e `-1.1%`

Interpretation:
- The per-token scale vector load is a small enough PTO seam to keep moving Stage 3b forward without reopening the large-case regression.
- The small case remains noisier, so the large case is the better stability reference for the next Stage 3b steps.

## Stage 3b regression trend notes

Current policy for the remaining Stage 3b seams:
- keep moving the functional PTO migration forward;
- record every observed regression trend as follow-up optimization input;
- avoid treating any single exploratory run as a new baseline unless it is measured on the same loop.
- keep the functionally-correct seam bundles enabled unless they break output contracts or the build.

Stability loop used for the entries below:
- large case only: `m=2049, k=128, n=128, topk=2, max_output_size=4098`
- no rebuild between runs when only the runtime path is being compared
- `warmup=5`, `measure=20`

Reference on the kept seam set at the start of this log:
- current kept seam set: `BlockEpilogue3` scale-vector `TLOAD` only
- observed large-case result on the stability loop: kernel `25.98 us`, e2e `85.60 us`, `PASS`

Observed regression entries on the same stability loop:

| seam | scope | kernel avg (us) | e2e avg (us) | delta vs kept seam set | accuracy | status |
| --- | --- | ---: | ---: | --- | --- | --- |
| SwiGLU final dequant-scale writeback PTO | `block_epilogue_pertoken_swiglu.hpp` tail writeback | 38.84 | 108.51 | kernel `+49.5%`, e2e `+26.8%` | PASS | reverted |
| SwiGLU input per-token-scale preload PTO | `block_epilogue_pertoken_swiglu.hpp` per-token scale ingress | 60.93 | 148.65 | kernel `+134.5%`, e2e `+73.7%` | PASS | reverted |
| Remote scratch-store PTO bundle | `BlockEpilogue2/3` remote-only local-scratch store before `TPUT` | 47.07 | 126.48 | kernel `+81.2%`, e2e `+47.8%` | PASS | kept for function-first |
| Full Stage 3b function-first bundle | all remaining epilogue GM↔UB shells plus `dispatch_ffn_combine_kernel.hpp` copy helpers | 35.67 | 106.14 | kernel `+37.3%`, e2e `+24.0%` | PASS | kept |

Related historical trend from earlier Stage 3b attempts:
- PTO-izing the remote scratch-store shell in `BlockEpilogue2/3` repeatedly interacted badly with the large-case path.
- The symptom pattern is consistent across those attempts: functional `PASS`, but either build friction at the helper/seam boundary or renewed large-case performance regression after the seam is enabled.
- Treat the remote scratch-store shell and the SwiGLU scale I/O shell as the two main regression clusters for the later optimization pass.

Current function-first state after the latest validation pass:
- kept seam set first expanded to include the earlier `BlockEpilogue3` scale-vector `TLOAD` plus the `BlockEpilogue2/3` remote scratch-store PTO bundle.
- reference functional regression pass at that intermediate point: small case `29.07 us / 108.95 us`, large case `40.10 us / 131.68 us`, both `PASS`.

## Stage 3b function-first completion

Current status:
- all remaining epilogue GM↔UB shells are now on PTO `TLOAD/TSTORE`, including:
  - `block_epilogue_pertoken_row.hpp` local row load/store shell
  - `block_epilogue_pertoken_v2.hpp` main 2D tile load/store shell
  - `block_epilogue_pertoken_swiglu.hpp` input/output shell and tail dequant-scale writeback
- the remaining generic copy helpers in `dispatch_ffn_combine_kernel.hpp` are also on PTO vector load/store.
- Stage 3b functionality is considered complete: both reference cases still `PASS`.
- Performance is intentionally not re-tuned here; use the stability-loop row above as the starting point for the later optimization pass.

Reference functional regression pass after the full Stage 3b bundle:
- small case: kernel `27.90 us`, e2e `121.41 us`, `PASS`
- large case: kernel `51.60 us`, e2e `149.54 us`, `PASS`

## Catlass cleanup checkpoint

Current status:
- `dispatch_ffn_combine_v3` source and CMake no longer depend on Catlass include paths or `CATLASS_ARCH`.
- The live int8/zN/A2 path now builds against the local compat surface in `op_kernel/utils/dispatch_policy_custom.hpp`.
- A fresh rebuild after removing the Catlass tail still keeps both reference cases at `PASS`.

Reference commands:

```bash
bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32

bash csrc/mc2/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 2049 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 4098
```

Observed results after the Catlass cleanup checkpoint:
- small case: kernel `21.79 us`, e2e `94.08 us`, `PASS`
- large case: kernel `28.89 us`, e2e `125.03 us`, `PASS`

Performance state relative to the earlier full Stage 3b function-first checkpoint:
- small case: kernel `-21.9%`, e2e `-22.5%`
- large case: kernel `-44.0%`, e2e `-16.4%`

Interpretation:
- The Catlass deshelling did not break the output contract or standalone build flow.
- On this checkpoint, performance is also better than the earlier full Stage 3b function-first reference instead of merely holding steady.
- The remaining build noise is the PTO macro redefinition warning, not a Catlass dependency.
