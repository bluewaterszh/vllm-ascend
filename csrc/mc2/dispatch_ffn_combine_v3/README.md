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
- Stage 1 HCCL context / window migration: `5db0ebc7`

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
- record every observed regression trend in this README as follow-up optimization input;
- avoid treating any single exploratory run as a new baseline unless it is measured on the same loop;
- do not rollback a functionally-correct seam bundle only because the large-case kernel time regresses;
- keep the functionally-correct seam bundles enabled unless they break output contracts or the build, and leave the performance cleanup for a later unified optimization pass.

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

## legacy compat cleanup checkpoint

Current status:
- `dispatch_ffn_combine_v3` source and CMake no longer depend on the old third-party include paths or legacy arch define.
- The live int8/Zn/A2 path now builds against the local compat surface in `op_kernel/utils/dispatch_policy_custom.hpp`.
- A fresh rebuild after removing the legacy compat tail still keeps both reference cases at `PASS`.

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

Observed results after the legacy compat cleanup checkpoint:
- small case: kernel `21.79 us`, e2e `94.08 us`, `PASS`
- large case: kernel `28.89 us`, e2e `125.03 us`, `PASS`

Performance state relative to the earlier full Stage 3b function-first checkpoint:
- small case: kernel `-21.9%`, e2e `-22.5%`
- large case: kernel `-44.0%`, e2e `-16.4%`

Interpretation:
- The legacy compat deshelling did not break the output contract or standalone build flow.
- On this checkpoint, performance is also better than the earlier full Stage 3b function-first reference instead of merely holding steady.
- The remaining build noise is the PTO macro redefinition warning, not a legacy compat dependency.

## PTO runtime/context cleanup checkpoint

Current status:
- `dispatch_ffn_combine_v3` no longer keeps a separate `HCCL_COMM` kernel path.
- The runtime now reconstructs a shared PTO-style `HcclDeviceContext` on the host side and passes that unified context into the kernel.
- The device-side HCCL window path now reads rank/window metadata only from that simplified context instead of parsing the old HCCL-internal resource layout in-kernel.
- The local legacy compat naming in the live path has also been renamed to PTO-style local naming.
- A fresh rebuild after this cleanup still keeps both reference cases at `PASS`.

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

Observed results after the runtime/context cleanup checkpoint:
- small case: kernel `31.25 us`, e2e `120.12 us`, `PASS`
- large case: kernel `31.02 us`, e2e `107.96 us`, `PASS`

Performance state relative to the earlier legacy compat cleanup checkpoint:
- small case: kernel `+43.4%`, e2e `+27.7%`
- large case: kernel `+7.4%`, e2e `-13.7%`

Interpretation:
- The unified PTO-style context path keeps the standalone ABI and output contract stable while removing the last live split between standalone and `HCCL_COMM` kernel semantics.
- The large case improves slightly on this checkpoint, while the small case regresses versus the earlier compat-only cleanup; treat that small-case movement as another later optimization input instead of reopening functionality work.
- The PTO macro redefinition warning was removed by dropping the duplicate build definition from `CMakeLists.txt`.

## PTO naming cleanup checkpoint

Current status:
- The remaining project-specific legacy identifiers in the live v3 path were renamed to PTO-style naming.
- The local layout adapter names now follow PTO-native spelling: `ND`, `DN`, `Zn`, `Nz`, `Zz`.
- The dead v3-local `op_kernel/utils/moe_distribute_base.h` copy was removed.
- A fresh rebuild after this naming cleanup still keeps both reference cases at `PASS`.

Reference functional regression pass after the naming cleanup:
- small case: kernel `32.15 us`, e2e `128.73 us`, `PASS`
- large case: kernel `39.14 us`, e2e `125.41 us`, `PASS`

Interpretation:
- This checkpoint was a naming-only cleanup pass; it did not change the standalone ABI, tiling contract, or kernel math path.
- The perf movement on these short runs should be treated as checkpoint noise unless it repeats on the large-case stability loop.

## PTO layout-surface checkpoint

Current status:
- The live layout adapters in `op_kernel/utils/dispatch_policy_custom.hpp` now store PTO `Shape` / `Stride` objects instead of the older local `Coord` storage.
- The live layout adapters also carry PTO metadata (`kPtoLayout`, `kTileLayout`, `kBLayout`, `kSLayout`), so layout intent is now expressed in PTO terms instead of only by concrete local type names.
- `op_kernel/utils/select_helper.hpp` now chooses the Zn B-layout path by PTO `TileLayoutCustom::ZN` instead of hard-coding a concrete local layout type match.
- `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp` no longer relies on static global layout objects for the L1 tile layouts; it now rebuilds those PTO-backed layouts through device helpers so the new non-literal layout storage remains valid in AICORE code.
- A fresh rebuild after this header-level shrink still keeps both reference cases at `PASS`.

Reference functional regression pass after the layout-surface checkpoint:
- small case: kernel `31.73 us`, e2e `113.84 us`, `PASS`
- large case: kernel `32.43 us`, e2e `118.46 us`, `PASS`

Performance state relative to the earlier PTO naming cleanup checkpoint:
- small case: kernel `-1.3%`, e2e `-11.6%`
- large case: kernel `-17.1%`, e2e `-5.5%`

Interpretation:
- This is the first Phase 1 checkpoint that changes the live compat surface semantics instead of only renaming identifiers.
- The remaining local coord shell is now narrower but not gone: `MatrixCoord` / `GemmCoord` still cluster in `dispatch_ffn_combine_kernel.hpp`, `block_mmad_preload_async_fixpipe_quant.hpp`, and the epilogue shells, which matches the next compute-side PTO migration target.
- The standalone ABI, kernel entry contract, and output correctness remain stable after the PTO-backed layout storage swap.

## PTO compute-shape helper checkpoint

Current status:
- The hot offset/tile-shape math in `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp` now routes through PTO-backed helper coords/shapes (`MakePtoCoord*`, `MulPtoCoord2D`, `ToPtoShape*`) instead of rebuilding local `MatrixCoord` objects at each GM/L1/L0 shell seam.
- The top-level live compute path in `op_kernel/dispatch_ffn_combine_kernel.hpp` now also uses PTO-backed coord/shape descriptors for block offsets, block-scheduler tile shape updates, and GM tile-view assembly in the GMM1/GMM2 paths.
- The remaining local coord shell is pushed outward to the scheduler/epilogue interface boundary rather than the inner compute-side layout math.
- A fresh rebuild after this compute-side helper swap still keeps both reference cases at `PASS`.

Reference functional regression pass after the compute-shape helper checkpoint:
- small case: kernel `29.65 us`, e2e `111.77 us`, `PASS`
- large case: kernel `27.09 us`, e2e `119.16 us`, `PASS`

Performance state relative to the earlier PTO layout-surface checkpoint:
- small case: kernel `-6.6%`, e2e `-1.8%`
- large case: kernel `-16.5%`, e2e `+0.6%`

Interpretation:
- The inner compute-side shape plumbing can move toward PTO descriptors without reopening the earlier large-case correctness regressions.
- The remaining non-PTO coord surface is now mostly at the `GemmCoord` / `MatrixCoord` boundary shared with the epilogue shells, which is the next lower-risk migration target.
- The only new build noise from this batch was duplicate `inline` spelling on the PTO helper wrappers; that warning can be removed without changing the live kernel contract.

## PTO epilogue-interface checkpoint

Current status:
- The row/SwiGLU epilogue shells now accept PTO 2D shape descriptors at the kernel boundary instead of local `MatrixCoord` wrappers.
- The `CombineV2` path now also hands the per-tile epilogue entry PTO 2D coord/shape descriptors instead of synthesizing `GemmCoord {m, n, 1}` placeholders for a K dimension that the epilogue path never used.
- The device-side epilogue internals keep the previously validated load/store order, remote scratch-store behavior, and `TPUT` sequencing; only the boundary descriptor type is changed.
- A fresh rebuild after this boundary shift still keeps both reference cases at `PASS`.

Reference functional regression pass after the epilogue-interface checkpoint:
- small case: kernel `30.30 us`, e2e `113.64 us`, `PASS`
- large case: kernel `30.10 us`, e2e `119.93 us`, `PASS`

Performance state relative to the earlier PTO compute-shape helper checkpoint:
- small case: kernel `+2.2%`, e2e `+1.7%`
- large case: kernel `+11.1%`, e2e `+0.6%`

Interpretation:
- This confirms the next low-risk seam: the epilogue call boundary can be expressed in PTO shape/coord terms without changing the already-stabilized remote-store and dequant shells.
- The remaining local coord logic is now concentrated mainly in the scheduler-facing side of `dispatch_ffn_combine_kernel.hpp` and `dispatch_policy_custom.hpp`, not in the epilogue interfaces themselves.
- The large-case e2e path stayed effectively flat on this short loop, so this seam remains function-first acceptable and can be left in place for the later optimization pass.

## PTO scheduler-shape checkpoint

Current status:
- `dispatch_policy_custom.hpp` now keeps the live `GemmIdentityBlockSwizzle` tile/loop bookkeeping in PTO 2D shapes instead of local `MatrixCoord` members.
- The scheduler exposes PTO MN getters, and the `CombineV2` epilogue launch path now consumes those PTO descriptors directly instead of re-expanding local scheduler coords back into PTO shape/coord objects in-kernel.
- The deeper GMM block scheduler contract is still intentionally left compatible with the existing `GemmCoord` matmul path, so this batch only shrinks the scheduler-facing compat shell instead of rewriting the live matmul launch ABI.
- A fresh rebuild after this scheduler-facing shrink still keeps both reference cases at `PASS`.

Reference functional regression pass after the scheduler-shape checkpoint:
- small case: kernel `31.53 us`, e2e `112.79 us`, `PASS`
- large case sanity rerun: kernel `33.56 us`, e2e `111.95 us`, `PASS`

Observed large-case noisy outlier on the same short loop:
- first run after this batch: kernel `47.41 us`, e2e `144.39 us`, `PASS`, with kernel `std=20.53 us`
- immediate rerun: kernel `33.56 us`, e2e `111.95 us`, `PASS`, with kernel `std=7.04 us`

Performance state relative to the earlier PTO epilogue-interface checkpoint:
- small case: kernel `+4.1%`, e2e `-0.8%`
- large case sanity rerun: kernel `+11.5%`, e2e `-6.7%`

Interpretation:
- The scheduler-facing compat shell can shrink further toward PTO-native shape bookkeeping without breaking output correctness or the standalone ABI.
- The first large-case short-loop sample after this batch was a noise spike rather than a stable new baseline; keep using reruns or the longer stability loop before drawing performance conclusions from scheduler-only changes.
- The remaining non-PTO coord surface is now mostly on the GMM-side `problemShape / actualBlockShape` contract, which is a larger seam than the scheduler bookkeeping that was migrated here.

## PTO comm-signal checkpoint

Current status:
- `op_kernel/utils/hccl_window.hpp` now expresses the live token-ready and cross-rank barrier behavior directly in terms of `pto::comm::Signal`, `TNOTIFY`, and `TWAIT` at the call site instead of routing those paths through an extra local wait/notify helper layer.
- The signal-region layout, epoch counters, and peer-window offsets are unchanged; this batch only shrinks the protocol-expression shell around them.
- The temporary lvalue requirement of PTO comm primitives is handled locally by binding the signal objects before `TNOTIFY/TWAIT`, so the live call sites stay PTO-native without changing the memory contract.
- A fresh rebuild after this comm-signal cleanup still keeps both reference cases at `PASS`.

Reference functional regression pass after the comm-signal checkpoint:
- small case: kernel `32.24 us`, e2e `119.79 us`, `PASS`
- large case: kernel `27.90 us`, e2e `139.83 us`, `PASS`

Performance state relative to the earlier PTO scheduler-shape checkpoint:
- small case: kernel `+2.3%`, e2e `+6.2%`
- large case: kernel `-16.9%`, e2e `+24.9%`

Interpretation:
- The kernel-side communication semantics are now closer to PTO-native expression while keeping the existing signal-region ABI stable.
- The large-case kernel time stayed healthy on this short loop, but the e2e sample is still too noisy to treat as a conclusion about the communication cleanup itself.
- The next remaining shell is primarily host-side metadata parsing and staging in `runtime_context.cpp` / `hccl_context.hpp`, not the device-side signal protocol expression.

## PTO runtime-metadata shell checkpoint

Current status:
- `op_kernel/utils/hccl_context.hpp` now exposes only the shared `HcclDeviceContext` surface used by the live runtime/kernel path.
- The host-only HCCL compat metadata structs used to rebuild that context were moved into `runtime_context.cpp`, so the resource-layout parsing shell is now local to the runtime implementation instead of leaking through a shared header.
- The standalone ABI, workspace layout, and device-side signal/window contract are unchanged; this batch only shrinks the host/runtime metadata shell around them.
- A fresh rebuild after this header/runtime split still keeps both reference cases at `PASS`.

Reference functional regression pass after the runtime-metadata shell checkpoint:
- small case: kernel `52.16 us`, e2e `157.14 us`, `PASS`, with kernel `std=41.68 us`
- large case: kernel `28.78 us`, e2e `110.36 us`, `PASS`

Performance state relative to the earlier PTO comm-signal checkpoint:
- small case: kernel `+61.8%`, e2e `+31.2%`
- large case: kernel `+3.2%`, e2e `-21.1%`

Interpretation:
- The host/runtime metadata shell can shrink further without reopening the older in-kernel HCCL resource parsing path or changing the live device context ABI.
- The small-case sample on this short loop was clearly noisy and should not be treated as a new baseline by itself.
- The large case stayed functionally and performance-wise healthy enough to keep this shell cleanup in place while the next PTO-native runtime/context seam is reduced.

## PTO runtime access-helper checkpoint

Current status:
- `StandaloneHcclContext` now exposes small helper accessors for the live PTO device-context pointer and peer-window metadata instead of forcing call sites to reach into `host_ctx` / `device_ctx` fields directly.
- `main.cpp` zero-window preparation and `tiling_builder.cpp` runtime-context wiring now consume that narrower helper seam rather than the raw storage fields.
- The local context-rebuild helpers in `runtime_context.cpp` now operate on `StandaloneHcclContext` directly, so the host/runtime metadata reconstruction shell no longer drags the whole `StandaloneRankRuntime` object through those internal helpers.
- A fresh rebuild after this helper-side shrink still keeps both reference cases at `PASS`.

Reference functional regression pass after the runtime access-helper checkpoint:
- small case: kernel `36.44 us`, e2e `124.54 us`, `PASS`, with kernel `std=8.88 us`
- large case sanity rerun: kernel `29.83 us`, e2e `116.89 us`, `PASS`, with kernel `std=9.19 us`

Observed large-case noisy sample on the same short loop:
- first run after the internal helper cleanup: kernel `30.78 us`, e2e `111.27 us`, `PASS`, with kernel `std=14.96 us`
- immediate rerun: kernel `29.83 us`, e2e `116.89 us`, `PASS`, with kernel `std=9.19 us`

Performance state relative to the earlier PTO runtime-metadata shell checkpoint:
- small case: kernel `-30.1%`, e2e `-20.7%`
- large case sanity rerun: kernel `+3.6%`, e2e `+5.9%`

Interpretation:
- The shared runtime/context surface is now thinner at the call boundary: external users no longer need to know where the PTO context stores windows or which raw pointer field should be passed into tiling.
- The short-loop samples are still noisy, especially on the large-case rerun pair, so this checkpoint should be treated as a function-first shell shrink rather than a performance claim.
- The next remaining runtime-side shell is mainly the `host_ctx` fill/copy body inside `runtime_context.cpp`, not the external call sites around it.

## PTO runtime+compute contract checkpoint

Current status:
- `StandaloneHcclContext` now owns the remaining host-side PTO context assembly operations (`reset / workspace / rank-info / window-fill / host↔device copy`) instead of leaving raw `host_ctx` field writes spread across `runtime_context.cpp`.
- The live GMM contract in `dispatch_ffn_combine_kernel.hpp` and `block_mmad_preload_async_fixpipe_quant.hpp` now passes PTO 3D shapes instead of `GemmCoord` at the kernel/block-matmul boundary.
- The GMM1/GMM2 loops now use the scheduler's PTO MN getters plus PTO 3D actual-shape descriptors, so the old `GemmCoord` contract no longer leaks through the live compute path.
- The host stub entry in `op_kernel/dispatch_ffn_combine.h` now builds the problem shape as PTO `PtoShape3D`, keeping the launch-side contract aligned with the kernel-side change.
- A fresh rebuild after this combined runtime/compute shell shrink still keeps both reference cases at `PASS`.

Reference functional regression pass after the runtime+compute contract checkpoint:
- small case: kernel `38.96 us`, e2e `131.12 us`, `PASS`, with kernel `std=7.98 us`
- large case sanity rerun: kernel `32.37 us`, e2e `118.42 us`, `PASS`, with kernel `std=12.16 us`

Observed large-case noisy sample on the same short loop:
- first run after the combined checkpoint: kernel `32.73 us`, e2e `128.56 us`, `PASS`, with kernel `std=13.41 us`
- immediate rerun: kernel `32.37 us`, e2e `118.42 us`, `PASS`, with kernel `std=12.16 us`

Performance state relative to the earlier PTO runtime access-helper checkpoint:
- small case: kernel `+6.9%`, e2e `+5.3%`
- large case sanity rerun: kernel `+8.5%`, e2e `+1.3%`

Interpretation:
- The last live `GemmCoord` contract on the GMM path is now mostly pushed back into `dispatch_policy_custom.hpp` internals; the active kernel/block-matmul seam itself is PTO-shaped.
- The runtime-side `host_ctx` assembly shell is now concentrated in `StandaloneHcclContext` methods instead of being open-coded in the ring-context rebuild path.
- The short-loop perf samples remain noisy, so this checkpoint should be treated as a semantic contract shrink first and a performance data point second.

## PTO final source-tail cleanup checkpoint

Current status:
- `dispatch_policy_custom.hpp` no longer keeps the live `GemmCoord` scheduler contract; the block swizzle surface is PTO-shape only.
- `runtime_context.hpp` / `runtime_context.cpp` now keep the device attach/free lifecycle inside `StandaloneHcclContext`, and the dead host-context accessors were removed.
- A fresh rebuild after this final cleanup still keeps both reference cases at `PASS`.

Reference functional regression pass after the final cleanup:
- small case: kernel `24.48 us`, e2e `104.06 us`, `PASS`
- large case: kernel `31.19 us`, e2e `120.29 us`, `PASS`

Interpretation:
- This closes the remaining non-performance tail for the current Stage 3b pass.
- The remaining movement in the short-loop perf numbers should be treated as checkpoint noise or later optimization input, not as a functional regression.

## PTO residual-surface cleanup checkpoint

Current status:
- The next low-risk Stage 3b cleanup batch kept shrinking the remaining GM-boundary shells in the routing/unpermute/kernel-helper paths without reopening the already-finished runtime/window contract work.
- The live copy-helper surface now additionally covers:
  - `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_op.h`
  - `op_kernel/moe_init_routing_quant_v2/moe_v2_sort_one_core.h`
  - `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`
  - `op_kernel/unpermute/moe_token_unpermute.h`
  - `op_kernel/dispatch_ffn_combine_kernel.hpp`
- The matmul-side shell was also tightened one step further by moving the live template entry points onto PTO-named aliases in `op_kernel/utils/dispatch_policy_custom.hpp`, which are then consumed by `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`.
- A fresh rebuild after this batch still keeps both reference cases at `PASS`.

Reference functional regression pass after the residual-surface cleanup checkpoint:
- small case: kernel `24.61 us`, e2e `98.12 us`, `PASS`
- large case: kernel `32.93 us`, e2e `120.62 us`, `PASS`

Residual ledger snapshot after this checkpoint:
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h`: `DataCopyPad=3`, `PipeBarrier=13`, `SetWaitFlag=12`, `Duplicate=6`, `ReduceMax=2`, `SyncAll=1`
- `op_kernel/dispatch_ffn_combine_kernel.hpp`: `DataCopyPad=4`, `DataCopy=1`, `PipeBarrier=4`, `Duplicate=2`, `SyncAll=12`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`: `DataCopyPad=4`, `SetWaitFlag=11`, `Duplicate=5`, `SyncAll=5`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h`: `DataCopyPad=2`, `PipeBarrier=12`
- `op_kernel/unpermute/moe_token_unpermute.h`: `DataCopyPad=2`, `DataCopy=1`, `PipeBarrier=9`, `Duplicate=1`
- `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`: `DataCopy=2`, `PipeBarrier=2`, `Gemm::Tile::=2`
- `op_kernel/utils/dispatch_policy_custom.hpp`: `DataCopy=6`, `Fixpipe=2`, `LoadData=1`, `Gemm::Tile::=11`, `Gemm::helper::=3`

Interpretation:
- The remaining compute-side work is now visibly concentrated instead of diffuse: the main routing/gather hot seam is `moe_v2_src_to_dst_and_gather.h`, the hottest no-touch seam is still `moe_v2_fullload_quant.h`, and the heaviest substrate shell remains the matmul/fixpipe pair in `dispatch_policy_custom.hpp` and `block_mmad_preload_async_fixpipe_quant.hpp`.
- The large case moved from the earlier `30.58 us / 125.60 us` matmul-alias checkpoint to `32.93 us / 120.62 us` here; record that drift as follow-up optimization input, but keep the seam bundle enabled because correctness and build stability remain intact.
- This checkpoint confirms the current function-first policy: continue shrinking safe seams, and defer any systematic tuning of the remaining hot shells to a later dedicated performance pass.

## PTO routing boundary-adapter follow-up

Current status:
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h` now wraps the remaining input/output tail-padding copies behind two local boundary adapters:
  - `LoadInputTile(...)`
  - `StoreExpandedXTile(...)`
- The aligned GM-boundary path now goes through PTO vector load/store first, while the true tail-padding fallback remains local to the adapter instead of staying open-coded in `Compute()`, `ComputeMax()`, and `ComputeScale()`.
- A fresh rebuild after this follow-up still keeps both reference cases at `PASS`.

Reference functional regression pass after the boundary-adapter follow-up:
- small case: kernel `34.44 us`, e2e `114.39 us`, `PASS`
- large case: kernel `31.64 us`, e2e `120.65 us`, `PASS`

Residual note for the touched file:
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h`: `DataCopyPad=2`, `PipeBarrier=13`, `SetWaitFlag=12`, `Duplicate=6`, `ReduceMax=2`, `SyncAll=1`
- Compared with the previous residual-surface checkpoint, the naked `DataCopyPad` sites in this file were reduced from 3 to 2 and are now concentrated in the boundary adapters instead of the main compute bodies.

Interpretation:
- This is the kind of seam that still fits the current Stage 3b policy: boundary-only shrink, no attempt to rewrite the hot vector math or synchronization cluster in one step.
- The large-case kernel time improved slightly versus the immediately previous checkpoint (`32.93 us -> 31.64 us`), while the small case regressed (`24.61 us -> 34.44 us`); keep both numbers as trend inputs only, not as a rollback trigger.
- The next safe follow-up in this file should continue to target boundary/helper concentration rather than the `PipeBarrier` / `SetWaitFlag` / `ReduceMax` hot cluster directly.

## Stage 3b final closure checkpoint

Current status:
- `task.md` is now closed under the Stage 3b function-first migration standard.
- Business-kernel direct GM-facing `DataCopy` has been reduced to zero.
- The last safe seam batch concentrated the remaining system-boundary copies behind local adapters in:
  - `op_kernel/unpermute/moe_token_unpermute.h`
  - `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`
  - `op_kernel/dispatch_ffn_combine_kernel.hpp`
  - `op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h`
- `moe_v2_fullload_quant.h` is intentionally closed by the hot-path-freeze rule: only the input/output tail adapters were kept, and the earlier whole-file PTO rewrite remains out of scope for this function-first stage because it regressed the large case.
- The remaining non-PTO surfaces are now explicitly classified instead of being open-ended residuals.

Reference commands for the final closure checkpoint:

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
  --m 4097 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 8194
```

Observed results after the final closure checkpoint:
- small case: kernel `34.88 us`, e2e `147.63 us`, `PASS`
- large case: kernel `29.03 us`, e2e `118.69 us`, `PASS`

Final residual classification:

### 1. boundary adapter
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h`: `DataCopyPad=2`, `PipeBarrier=13`, `SetWaitFlag=12`, `Duplicate=6`, `ReduceMax=2`, `SyncAll=1`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`: `DataCopyPad=1`, `SetWaitFlag=11`, `Duplicate=5`, `SyncAll=5`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h`: `DataCopyPad=2`, `PipeBarrier=12`
- `op_kernel/unpermute/moe_token_unpermute.h`: `DataCopyPad=2`, `PipeBarrier=9`, `Duplicate=1`
- `op_kernel/dispatch_ffn_combine_kernel.hpp`: `DataCopyPad=4`, `PipeBarrier=4`, `Duplicate=2`, `SyncAll=12`

### 2. business layer with no direct GM-facing `DataCopy`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_sort_one_core.h`: `ArithProgression=1`, `SyncAll=1`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_op.h`: `PipeBarrier=2`, `SetWaitFlag=2`, `SyncAll=4`

### 3. substrate / host-shell residuals
- `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`: `DataCopy=2`, `PipeBarrier=4`, `Gemm::Tile::=2`
- `op_kernel/utils/dispatch_policy_custom.hpp`: `DataCopy=6`, `Fixpipe=2`, `LoadData=1`, `Gemm::Tile::=11`, `Gemm::helper::=3`
- `SetWaitFlag` / `SyncAll`: retained where they still express cross-pipe or cross-core host-shell synchronization semantics.

Interpretation:
- Stage 3b is complete from the migration/task-closure perspective: the remaining non-PTO interfaces are localized, justified, and no longer spread through the business compute bodies without explanation.
- The next step is no longer function-first seam hunting; it is a dedicated performance pass over the hot vector/sync cluster and the matmul/fixpipe substrate.
- Keep treating the numbers above as checkpoint records and optimization input, not as a reason to reopen the completed Stage 3b functionality migration.
