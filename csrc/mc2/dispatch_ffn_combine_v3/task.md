# dispatch_ffn_combine_v3 task list

## 目标
- 完成 Stage 3b 的功能迁移，优先把已有等价面的实现收口到 PTO 风格。
- HCCL 仅保留 host 侧建链和 remote GM window 获取职责，kernel 侧只保留 PTO-neutral remote-window 语义。
- 先做功能覆盖和依赖收口，再单独做性能优化。

## 当前状态
- 已完成 remote-window 语义链改造：runtime -> tiling -> kernel 不再暴露 HCCL-specific 命名。
- 已完成多轮 routing / gather / dynamic-quant 的 PTO helper 收口，最近一轮继续覆盖了 `moe_v2_src_to_dst_op.h`、`moe_v2_sort_one_core.h`、`moe_v2_expert_token_out.h`、`moe_token_unpermute.h`、`dispatch_ffn_combine_kernel.hpp`。
- 已把 matmul 外围壳进一步压到 PTO 命名 alias：`dispatch_policy_custom.hpp` 新增本地 PTO alias，`block_mmad_preload_async_fixpipe_quant.hpp` 改为优先消费这些 alias。
- 已完成 `op_kernel/` 关键残留面的精确盘点与最终分类：业务 kernel 中的 direct GM-facing `DataCopy` 已清零，剩余 `DataCopyPad` 全部收口到 boundary adapter；direct `DataCopy` 仅保留在 matmul/fixpipe substrate。
- 当前 build 通过，最新 standalone 验证结果：small case kernel `34.88 us` / e2e `147.63 us`，large case kernel `29.03 us` / e2e `118.69 us`，双 rank 均 PASS。
- 已验证 `moe_v2_fullload_quant.h` 的整段 PTO 化会拉差 large case；本阶段已改为只保留 hot-path tail adapter，不再做整段硬切。

## 已完成
1. runtime/kernel 侧 HCCL 语义压缩成 PTO-neutral remote-window 壳
   - 涉及：
     - `csrc/mc2/dispatch_ffn_combine_v3/runtime_context.hpp`
     - `csrc/mc2/dispatch_ffn_combine_v3/runtime_context.cpp`
     - `csrc/mc2/dispatch_ffn_combine_v3/tiling_builder.cpp`
     - `csrc/mc2/dispatch_ffn_combine_v3/op_kernel/dispatch_ffn_combine_tiling.h`
     - `csrc/mc2/dispatch_ffn_combine_v3/op_kernel/dispatch_ffn_combine.h`
     - `csrc/mc2/dispatch_ffn_combine_v3/op_kernel/dispatch_ffn_combine_kernel.hpp`
     - `csrc/mc2/dispatch_ffn_combine_v3/op_kernel/utils/hccl_context.hpp`
     - `csrc/mc2/dispatch_ffn_combine_v3/op_kernel/utils/hccl_window.hpp`
2. 一轮低风险 GM-facing copy / vector helper 收口
   - 已纳入 PTO helper 的代表文件：
     - `moe_v2_pto_sort.h`
     - `moe_v2_mrgsort.h`
     - `moe_v2_mrgsort_out.h`
     - `moe_v2_gather_out.h`
     - `moe_v2_gather_quant.h`
     - `moe_v2_gather_dynamic_quant.h`
     - `moe_v2_fullload_dynamic_quant.h`
     - `moe_v2_fullload_quant_base.h`
     - `moe_v2_init_routing_fullload.h`
     - `moe_v2_sort_multi_core.h`
     - `moe_v2_src_to_dst_with_capacity.h`
     - `moe_v2_src_to_dst_op.h`
     - `moe_v2_sort_one_core.h`
     - `moe_token_unpermute.h`
3. 小幅继续收口的稳定 seam
   - `moe_v2_expert_token_out.h` 继续缩小 GM-boundary 写回面。
   - `dispatch_ffn_combine_kernel.hpp` 的 `expertIdx` / `flag` / `tokenPerExpert` / `preSumBeforeRank` 等向量读写继续统一到本地 PTO vector helper。
4. `moe_v2_src_to_dst_and_gather.h` 的安全局部 seam
   - `CopyIn()` 的 routing 映射读取改成 `PtoLoadVector`。
   - 非热路径的 GM-boundary copy 优先切到 PTO：`expandedRowIdx` 写回、`expandedX` 写回、`dynamicQuantScale` 写回、`quantSmooth`/`quantSrc` 的读写。
   - 保留非 float 输入路径的 `DataCopyPad` 和局部 `Duplicate/ReduceMax/PipeBarrier` 热段，不做整段强切。
5. Gemm / matmul helper 外围壳收口
   - `block_mmad_preload_async_fixpipe_quant.hpp` 增加 `detail::MatmulShell`。
   - `dispatch_policy_custom.hpp` 新增 `PtoElementAccumulatorSelector`、`PtoL1AlignHelper`、`PtoCopyGmToL1`、`PtoQuantTileCopy`、`PtoCopyL0CToGm` 等 alias。
   - `CopyGmToL1S` / `CopyL1ToFP` / `CopyL0CToGm` / `L1AlignHelper` 不再直接散落在 `BlockMmad` 模板体内，而是经本地 PTO alias / wrapper 边界汇总。
   - `dispatch_policy_custom.hpp` 继续作为底层 matmul/fixpipe substrate 的集中实现面，未直接重写主算流水。

## 按计划刷新后的任务表（含完成状态）

| 计划项 | 当前状态 | 还没做的点 | 主要文件 | 完成状态 |
| --- | --- | --- | --- | --- |
| Task 1 runtime/kernel 侧 HCCL 语义压成 remote-window 壳 | 已完成 | 无 | `runtime_context.hpp/.cpp`, `tiling_builder.cpp`, `dispatch_ffn_combine_tiling.h`, `dispatch_ffn_combine.h`, `dispatch_ffn_combine_kernel.hpp`, `hccl_context.hpp`, `hccl_window.hpp` | 已完成 |
| Task 2.1 GM-facing copy 收口 | 已完成 | 无；业务 kernel 中 direct GM-facing `DataCopy` 已清零，剩余系统边界 copy 已转入 adapter | `moe_v2_src_to_dst_and_gather.h`, `moe_v2_sort_one_core.h`, `moe_token_unpermute.h`, `dispatch_ffn_combine_kernel.hpp`, `moe_v2_fullload_quant.h` | 已完成 |
| Task 2.2 带 padding / atomic 的系统边界 copy 收口 | 已完成 | 无；剩余 `DataCopyPad` 已集中在 boundary adapter 中，不再散落在业务计算体 | `moe_v2_src_to_dst_and_gather.h`, `moe_v2_expert_token_out.h`, `dispatch_ffn_combine_kernel.hpp`, `moe_v2_fullload_quant.h`, `moe_token_unpermute.h` | 已完成 |
| Task 2.3 `fullload_quant` compute seam | 已完成（按热路径冻结策略） | 无；本阶段只保留 hot-path tail adapter，不再尝试整段 PTO 化 | `moe_v2_fullload_quant.h` | 已完成 |
| Task 3.1 向量 helper 继续收敛 | 已完成（阶段目标） | 无；非热路径向量 copy/helper 已收口，剩余 hot vector math 作为后续 perf pass 输入保留 | `moe_v2_src_to_dst_and_gather.h`, `moe_v2_fullload_quant.h`, `moe_token_unpermute.h`, `block_epilogue_pertoken_swiglu.hpp` | 已完成 |
| Task 3.2 同步/生命周期 helper 收敛 | 已完成（分类完成） | 无；`PipeBarrier / SetWaitFlag / Duplicate / ReduceMax / ArithProgression / SyncAll` 已明确划分为局部 pipeline helper 或宿主壳暂留 | `moe_v2_gather_dynamic_quant.h`, `moe_v2_fullload_dynamic_quant.h`, `moe_v2_src_to_dst_with_capacity.h`, `moe_v2_src_to_dst_and_gather.h`, `dispatch_ffn_combine_kernel.hpp` | 已完成 |
| Task 4.1 Gemm/matmul 外围壳隔离 | 已完成 | 无；matmul/fixpipe 壳已集中隔离到 substrate 文件，不再向业务 kernel 扩散 | `block_mmad_preload_async_fixpipe_quant.hpp`, `dispatch_policy_custom.hpp`, `dispatch_ffn_combine.cpp` | 已完成 |
| Task 4.2 Gemm 壳里的具体残留接口 | 已完成（集中隔离） | 无；残留 `Gemm::Tile::*` / `Gemm::helper::*` / `CopyGmToL1*` / `CopyL0CToGm` 已归拢到本地 substrate | `dispatch_ffn_combine.cpp`, `block_mmad_preload_async_fixpipe_quant.hpp`, `dispatch_policy_custom.hpp` | 已完成 |
| Task 5 残留非-PTO 依赖台账 | 已完成 | 无；已形成固定 ledger，并给出 adapter / hot path / substrate 归类 | `task.md`, `README.md`, `op_kernel/` | 已完成 |
| Task 5 + Verification 阶段化收口标准 | 已完成 | 无；已固化 grep + build + small/large PASS + 残留解释 的 checklist | `README.md`, `task.md` | 已完成 |

## 残留依赖台账（最终归档）

### 1. boundary adapter 暂留
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h`
  - 当前残留：`DataCopyPad=2`, `PipeBarrier=13`, `SetWaitFlag=12`, `Duplicate=6`, `ReduceMax=2`, `SyncAll=1`
  - 说明：只剩输入/输出 tail adapter 与热段同步壳；可对齐 GM 边界已全部收口。
- `op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h`
  - 当前残留：`DataCopyPad=1`, `SetWaitFlag=11`, `Duplicate=5`, `SyncAll=5`
  - 说明：只剩 atomic count/cumsum 写回 adapter。
- `op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h`
  - 当前残留：`DataCopyPad=2`, `PipeBarrier=12`
  - 说明：只剩 hot-path 输入/输出 tail adapter，整段 PTO 化已明确冻结。
- `op_kernel/unpermute/moe_token_unpermute.h`
  - 当前残留：`DataCopyPad=2`, `PipeBarrier=9`, `Duplicate=1`
  - 说明：只剩 load/store tail adapter 与局部向量热段。
- `op_kernel/dispatch_ffn_combine_kernel.hpp`
  - 当前残留：`DataCopyPad=4`, `PipeBarrier=4`, `Duplicate=2`, `SyncAll=12`
  - 说明：direct `DataCopy` 已清零；剩余是 stride/pad adapter 与跨 core 协作壳。

### 2. 业务层已无 direct GM-facing `DataCopy`
- `op_kernel/moe_init_routing_quant_v2/moe_v2_sort_one_core.h`
  - 当前残留：`ArithProgression=1`, `SyncAll=1`
  - 说明：GM-facing copy 已完全收口，剩余仅排序 payload 生成与同步壳。
- `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_op.h`
  - 当前残留：`PipeBarrier=2`, `SetWaitFlag=2`, `SyncAll=4`
  - 说明：GM-facing copy 已完全收口，剩余仅局部 pipeline/sync 壳。

### 3. substrate / 宿主壳暂留
- `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`
  - 当前残留：`DataCopy=2`, `PipeBarrier=4`, `Gemm::Tile::=2`
  - 说明：matmul 外围 tile/fixpipe substrate，继续集中保留在此，不向业务 kernel 扩散。
- `op_kernel/utils/dispatch_policy_custom.hpp`
  - 当前残留：`DataCopy=6`, `Fixpipe=2`, `LoadData=1`, `Gemm::Tile::=11`, `Gemm::helper::=3`
  - 说明：matmul/fixpipe substrate 的集中实现面；Stage 3b 只要求隔离，不要求此阶段重写底层。
- `SetWaitFlag` / `SyncAll`
  - 说明：当前仍承担跨 pipe / 跨 core 的宿主同步语义，已从“未分类残留”转为“明确暂留壳”。

## 本轮性能记录（只记录，不处理）
- `moe_v2_src_to_dst_and_gather.h` GM-boundary seam 后：
  - small: kernel `33.01 us`, e2e `111.40 us`
  - large: kernel `29.39 us`, e2e `109.45 us`
- `detail::MatmulShell` wrapper 边界收口后：
  - small: kernel `30.99 us`, e2e `109.74 us`
  - large: kernel `30.58 us`, e2e `125.60 us`
- `src_to_dst_op + sort_one_core + unpermute + kernel helper + matmul alias` 这一轮继续收口后：
  - small: kernel `24.61 us`, e2e `98.12 us`
  - large: kernel `32.93 us`, e2e `120.62 us`
- `moe_v2_src_to_dst_and_gather.h` boundary-adapter seam（输入 tail load / 输出 tail store 收口）后：
  - small: kernel `34.44 us`, e2e `114.39 us`
  - large: kernel `31.64 us`, e2e `120.65 us`
- `unpermute + expert_token_out + dispatch_ffn_combine_kernel + fullload_quant` adapter 收口完成后：
  - small: kernel `34.88 us`, e2e `147.63 us`
  - large: kernel `29.03 us`, e2e `118.69 us`

## 保留的性能敏感结论
- `moe_v2_fullload_quant.h` 不能整段直接切到 PTO helper；本阶段最终只保留 hot-path tail adapter。
- `moe_v2_src_to_dst_and_gather.h` 仍是当前最大的 compute-side 热段残留面，但其 direct copy 已缩到 boundary adapter。
- matmul/fixpipe 壳已经稳定收缩到 `dispatch_policy_custom.hpp` + `block_mmad_preload_async_fixpipe_quant.hpp` 两个集中点。
- 当前 large case 仍以“功能正确 + 记录性能趋势”为准，不因为单次性能波动回退功能正确的 seam。

## 最终验收 checklist
- [x] HCCL/HCOM 直接 API 仅保留在 host runtime 层
- [x] 业务 kernel 中 direct GM-facing `DataCopy` 已清零
- [x] 剩余 `DataCopyPad` 已收口到 boundary adapter / hot-path adapter
- [x] matmul/fixpipe 残留已集中隔离到 substrate 文件
- [x] build 成功
  - `cmake --build csrc/mc2/dispatch_ffn_combine_v3/build --target dispatch_ffn_combine_v3 -j16`
- [x] small case PASS
  - `bash csrc/mc2/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 16 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 32`
- [x] large case PASS
  - `bash csrc/mc2/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 4097 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 8194`
- [x] 剩余非-PTO 依赖均已给出明确归因（boundary adapter / hot path / substrate）
