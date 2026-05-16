# dispatch_ffn_combine_v3 task list

## 目标
- 完成 Stage 3b 的功能迁移，优先把已有等价面的实现收口到 PTO 风格。
- HCCL 仅保留 host 侧建链和 remote GM window 获取职责，kernel 侧只保留 PTO-neutral remote-window 语义。
- 先完成功能覆盖和依赖收口，再单独做性能优化。

## 当前结论
- Task 1 已完成：runtime -> tiling -> kernel 的 HCCL 语义已经压缩成 remote-window 壳，kernel 侧不再直接消费 host HCCL API。
- Task 6.1 已完成：`op_kernel/` 下 direct `Cast/Add/Adds/Mul/Muls/Div/Abs/Exp/ReduceMax` 已清零；向量计算统一收口到本地 PTO helper。
- Task 6.2 已完成：routing 链、epilogue 链和 kernel 本地同步链上的 direct `PipeBarrier/SetWaitFlag/SyncAll` 已收口到 PTO 风格 helper；剩余 direct `CrossCore*` 属于跨核协作壳，不再算业务层漏改。
- Task 6.3 已完成：`DataCopyPad` / `DataCopyPadExtParams` 只剩 boundary adapter，不再散落在业务 compute seam 内部。
- Task 6.4 已完成：matmul / fixpipe / Gemm 残留已经稳定收敛到 [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) 和 [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)。
- Task 6.5 已完成：v3 的非 PTO 直接依赖已经分成 boundary adapter、kernel coordination shell、substrate 三类，并同步到 [api_interface.md](api_interface.md)。

## 最新验证
- build
  - `cmake --build csrc/mc2/dispatch_ffn_combine_v3/build --target dispatch_ffn_combine_v3 -j16`
- small case
  - `bash csrc/mc2/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 16 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 32`
  - kernel `29.00 us`, e2e `114.37 us`, rank0/rank1 PASS
- large case
  - `bash csrc/mc2/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 4097 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 8194`
  - kernel `29.29 us`, e2e `116.56 us`, rank0/rank1 PASS

## 任务完成表

| 任务 | 状态 | 结果 |
| --- | --- | --- |
| Task 1 runtime/kernel 侧 HCCL 语义压成 remote-window 壳 | 已完成 | host/runtime 保留 HCCL bootstrap；kernel 只消费 remote-window + PTO comm 语义。 |
| Task 2.1 GM-facing copy 收口 | 已完成 | 业务 kernel 中 direct GM-facing `DataCopy` 已清零。 |
| Task 2.2 带 padding / atomic 的系统边界 copy 收口 | 已完成 | 剩余 `DataCopyPad` 仅保留在 boundary adapter。 |
| Task 2.3 `fullload_quant` compute seam | 已完成 | 维持 hot-path adapter，避免大 case 稳定性回退。 |
| Task 3.1 向量 helper 收敛 | 已完成 | direct 向量算子已经收口到 PTO helper。 |
| Task 3.2 pipeline/sync helper 收敛 | 已完成 | routing / epilogue / kernel local sync 已 helper 化。 |
| Task 4.1 Gemm/matmul 外围壳隔离 | 已完成 | matmul/fixpipe 残留集中到 substrate 文件。 |
| Task 4.2 Gemm 壳里的具体残留接口 | 已完成 | `Gemm::Tile::*` / `Gemm::helper::*` / `Fixpipe` 已稳定归类为 substrate。 |
| Task 5 残留非-PTO 依赖台账 | 已完成 | 当前活树的残留已经重新归档。 |
| Task 6.1 向量算子 PTO 化 | 已完成 | `op_kernel/` 下 direct 向量算子 grep 为 0。 |
| Task 6.2 pipeline/sync helper PTO 化 | 已完成 | routing 链 direct sync 已清空；kernel local `SyncAll<true>()` 已收口到 `kernel_detail::PtoSyncAll<true>()`。 |
| Task 6.3 boundary adapter 再评估 | 已完成 | 剩余 `DataCopyPad` 已确认属于 boundary adapter，不继续混入业务 seam。 |
| Task 6.4 substrate 壳精细拆分 | 已完成 | `dispatch_policy_custom.hpp` 与 `block_mmad_preload_async_fixpipe_quant.hpp` 成为唯一主集中点。 |
| Task 6.5 AscendC substrate 与宿主壳台账 | 已完成 | 当前剩余依赖的职责边界已经固定。 |

## 残留非-PTO 依赖台账

### 1. boundary adapter 暂留
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - `DataCopyPad`
  - 说明：stride/pad 读写 adapter，已经不再是普通业务 copy seam。
- [moe_v2_src_to_dst_and_gather.h](op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h)
  - `DataCopyPad`
  - 说明：输入 tail load / 输出 tail store adapter。
- [moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h)
  - `DataCopyPad`, `SetAtomicAdd`, `SetAtomicNone`
  - 说明：atomic 写回 adapter。
- [moe_v2_fullload_quant.h](op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h)
  - `DataCopyPad`
  - 说明：hot-path 尾块 adapter，本阶段不再整段强切。
- [moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h)
  - `DataCopyPad`, `DataCopyPadExtParams`
  - 说明：尾块 load/store adapter。

### 2. kernel coordination shell 暂留
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SetL2CacheHint`
  - 说明：跨核协作和 cache hint 壳；当前没有清晰 public PTO 1:1 对应面。
- [hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
  - `DataCacheCleanAndInvalid`
  - 说明：remote-window cache coherence 壳；核心 notify/wait 已是 `pto::comm::TNOTIFY/TWAIT`。
- 本地 helper 定义
  - [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h)
  - [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp)
  - [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp)
  - [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp)
  - [hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
  - 说明：这些文件内部仍用 AscendC 原语实现 wrapper，但业务调用面已经统一转到 PTO 风格 helper。

### 3. substrate / 宿主壳暂留
- [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp)
  - `DataCopy`, `LoadData`, `LoadDataWithTranspose`, `Fixpipe`, `Gemm::Tile::*`, `Gemm::helper::*`
  - 说明：matmul/fixpipe substrate 的集中实现面。
- [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)
  - `DataCopy`, `PipeBarrier`, `CrossCoreSetFlag`, matmul event shell
  - 说明：主算外围 shell，已从业务 kernel 中隔离出来。
- 通用 AscendC substrate
  - `kernel_operator.h`, `__aicore__`, `GM_ADDR`, `LocalTensor`, `GlobalTensor`, `TPipe`, `TQue`, `TBuf`
  - 说明：当前 PTO 仍依附这层 substrate 运行。

## 性能记录（只记录，不优化）

| 检查点 | small kernel / e2e | large kernel / e2e | 说明 |
| --- | --- | --- | --- |
| swiglu 向量 seam 收口后 | `54.30 us / 147.36 us` | `34.80 us / 131.66 us` | direct 向量热段转 PTO helper 的首个稳定点。 |
| 第一批 sync helper 收口后 | `24.18 us / 125.36 us` | `28.90 us / 108.74 us` | routing / epilogue / kernel helper 首轮收口。 |
| 本轮 routing + kernel sync 收口后 | `29.00 us / 114.37 us` | `29.29 us / 116.56 us` | 6.2 完成后的当前稳定点。 |

## 最终验收 checklist
- [x] HCCL/HCOM 直接 API 仅保留在 host runtime 层
- [x] 业务 kernel 中 direct GM-facing `DataCopy` 已清零
- [x] `DataCopyPad` 已收口到 boundary adapter
- [x] `op_kernel/` 下 direct 向量算子已清零
- [x] routing / epilogue / kernel local sync 已收口到 PTO 风格 helper
- [x] matmul/fixpipe 残留已集中隔离到 substrate 文件
- [x] build 成功
- [x] small case PASS
- [x] large case PASS
- [x] 剩余非-PTO 依赖均已给出明确归因
