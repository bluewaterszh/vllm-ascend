# dispatch_ffn_combine_v3 API interface inventory

## 目的
这份清单只回答一件事：`dispatch_ffn_combine_v3` 当前还剩哪些 **PTO 之外的直接依赖接口**，以及它们分别属于哪一类：
- boundary adapter
- kernel coordination shell
- substrate / 宿主壳

它不再重复阶段任务状态；任务完成情况见 [task.md](task.md)。

## 当前快照
- 这份清单对应当前活树，已覆盖 Task 6.1 ~ 6.5 的最终状态。
- `op_kernel/` 下 direct `Cast/Add/Adds/Mul/Muls/Div/Abs/Exp/ReduceMax` 已清零。
- routing 链 direct `PipeBarrier/SetWaitFlag/SyncAll` 已统一改走本地 PTO 风格 helper。
- 剩余 direct 非 PTO 接口主要集中在：
  1. boundary adapter；
  2. cross-core / cache-coherence / atomic 这类 kernel coordination shell；
  3. matmul / fixpipe / Gemm substrate。

## PTO 公开映射参考
已确认存在并公开的 PTO 指令族：
- 数据搬运：`TLOAD` / `TSTORE` / `TMOV`
- 类型转换：`TCVT`
- 向量算子：`TADD` / `TADDS` / `TMUL` / `TMULS` / `TDIV` / `TDIVS` / `TABS` / `TMAX` / `TMAXS`
- reduce：`TROWMAX` / `TCOLMAX`
- 同步/事件：`TASSIGN` / `TSYNC`
- matmul：`TMATMUL` / `TMATMUL_ACC`
- 通信：`TGET` / `TPUT` / `TNOTIFY` / `TWAIT`

仍需谨慎看待的 PTO 名字：
- `TSTORE_FP`
- `TMOV_FP`

它们虽然公开存在，但当前不应当成可以直接替换 v3 matmul/fixpipe substrate 的成熟 drop-in。

---

## 一、host/runtime 侧非 PTO 直接依赖

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| ACL runtime (`acl/acl.h`) | [main.cpp](main.cpp), [runtime_context.hpp](runtime_context.hpp) | No | 保留 | device/runtime 生命周期依赖，不是 PTO 替代范围。 |
| HCCL host bootstrap (`hccl/hccl_comm.h`, `hccl/hccl_types.h`) | [main.cpp](main.cpp), [runtime_context.hpp](runtime_context.hpp) | No | 保留 | host 侧建链与资源获取依赖；kernel 已不再直接触达。 |
| `HcomGetCommHandleByGroup` / `HcomGetL0TopoTypeEx` / `HcclAllocComResourceByTiling` | [runtime_context.hpp](runtime_context.hpp), [runtime_context.cpp](runtime_context.cpp) | No | 保留 | 仅允许停留在 host/runtime 层。 |

结论：host/runtime 的 HCCL/ACL 依赖仍然存在，但已经退回到正确边界。

---

## 二、AscendC substrate 依赖

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `kernel_operator.h` | [op_kernel/dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [op_kernel/utils/dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [op_kernel/unpermute/moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h) | No | 保留 | 整个 kernel substrate 总入口。 |
| `__aicore__`, `GM_ADDR` | 全部活 kernel 文件 | No | 保留 | kernel ABI / 编译属性。 |
| `LocalTensor`, `GlobalTensor` | 全部活 kernel 文件 | No | 保留 | PTO helper 仍直接消费 AscendC tensor substrate。 |
| `TPipe`, `TQue`, `TBuf` | routing / gather / unpermute / epilogue 文件 | No | 保留 | queue/buffer 生命周期仍是 AscendC substrate。 |
| `GetBlockIdx`, `GetBlockNum`, `GetTaskRation` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 保留 | device builtins。 |
| `DataCacheCleanAndInvalid` | [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 保留 | cache coherence 宿主壳。 |
| `SetL2CacheHint` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp) | No | 保留 | cache hint 调优接口。 |

结论：这些属于 PTO 当前仍依附的 AscendC substrate，不再作为 Stage 3b 未完成项处理。

---

## 三、业务/kernel 仍直接使用的非 PTO 接口

### 3.1 boundary adapter

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `DataCopyPad` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [moe_v2_src_to_dst_and_gather.h](op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h), [moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h), [moe_v2_fullload_quant.h](op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h), [moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h) | Partial | 当前保留 | `TLOAD/TSTORE/TFILLPAD` 只能覆盖部分语义；当前剩余点都已属于 tail/stride/atomic 边界 adapter。 |
| `DataCopyPadExtParams` / `DataCopyExtParams` | 同上 | Partial | 当前保留 | 没有 1:1 PTO 参数结构；要替换必须连 copy 语义一起改。 |
| `SetAtomicAdd` / `SetAtomicNone` | [moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h) | No clear 1:1 | 当前保留 | 这是 tensor atomic writeback adapter，不是 comm signal atomic 语义。 |

### 3.2 向量计算接口

当前结论：**业务层 direct 向量算子已经清零。**

已收口到 PTO helper 的主入口：
- [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h)
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
- [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp)
- [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp)
- [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp)

对应 PTO 面：
- `TCVT`
- `TADD/TADDS`
- `TMUL/TMULS`
- `TDIV`
- `TABS`
- `TROWMAX/TMAX`
- `TEXP`

结论：向量计算已不再是当前活树里的 direct 非 PTO 残留面。

### 3.3 pipeline / sync / coordination

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| helper body 内部的 `PipeBarrier` / `SetFlag` / `WaitFlag` / `SyncAll` | [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h), [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp), [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp), [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp), [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | Partial | 当前保留为 helper substrate | 业务调用面已经改成 PTO 风格 helper；剩下的是 wrapper 的底层实现。 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | No clear 1:1 | 当前保留 | 这是本地 cross-core 协作壳，不等价于 comm 侧 `TNOTIFY/TWAIT`。 |
| `DataCacheCleanAndInvalid` | [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 当前保留 | remote-window cache coherence 壳。 |

结论：routing 链 direct sync 已完成 helper 化；当前剩余 direct sync 主要是 helper body 和 cross-core coordination shell。

---

## 四、matmul / fixpipe / Gemm substrate 残留

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `lib/matmul_intf.h` | [dispatch_ffn_combine.cpp](op_kernel/dispatch_ffn_combine.cpp) | No | 保留 | AscendC/Gemm substrate 头。 |
| `DataCopy` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | Yes / Partial | 当前保留在 substrate | PTO 有 `TLOAD/TSTORE/TMOV`，但剩余场景多是 matmul shell 内部搬运，不是普通 GM↔Vec drop-in。 |
| `LoadData` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 当前保留 | L1→L0 fragment copy，不是普通 `TLOAD` 直替。 |
| `LoadDataWithTranspose` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 当前保留 | 底层转置搬运面。 |
| `Fixpipe` / `FixpipeParamsV220` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 当前保留 | `TSTORE_FP/TMOV_FP` 方向存在，但当前不适合作成熟 drop-in。 |
| `Gemm::Tile::*` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | Partial / No | 当前保留 | 已经通过本地 `Pto*` alias 收口，但底层仍是 Gemm substrate。 |
| `Gemm::helper::*` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | No | 当前保留 | traits/helper 层，不是 PTO primitive。 |

结论：matmul/fixpipe 仍然是最重的硬残留，但已经从业务 kernel seam 中隔离出来。

---

## 五、按“还能不能继续改”重新归类

### A. 如需继续推进，优先级最高
- `DataCopyPad` 的 boundary adapter 再评估
- helper body 对 `TSYNC/TASSIGN` 的进一步吸收
- matmul substrate 中少量可继续 alias 化的接口

### B. 当前已经完成收口，不再作为 Stage 3b 未完成项
- direct 向量算子
- routing 链 direct `PipeBarrier/SetWaitFlag/SyncAll`
- kernel local `SyncAll<true>()` 业务调用点

### C. 当前明确属于 substrate / 宿主壳
- `kernel_operator.h`
- `__aicore__`, `GM_ADDR`
- `LocalTensor`, `GlobalTensor`
- `TPipe`, `TQue`, `TBuf`
- `GetBlockIdx`, `GetBlockNum`, `GetTaskRation`
- `DataCacheCleanAndInvalid`
- `SetL2CacheHint`
- `CrossCoreSetFlag`, `CrossCoreWaitFlag`
- `SetAtomicAdd`, `SetAtomicNone`
- `lib/matmul_intf.h`
- `Gemm::helper::*`

---

## 六、按文件看当前还剩哪些直接接口

### [op_kernel/dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
- `DataCopyPad`
- `CrossCoreSetFlag` / `CrossCoreWaitFlag`
- `SetL2CacheHint`
- 本地 sync helper 的底层实现

### [op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h](op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h)
- `DataCopyPad`
- 其余向量/同步调用已改走 PTO helper

### [op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h)
- `DataCopyPad`
- `SetAtomicAdd` / `SetAtomicNone`
- 其余同步调用已改走 PTO helper

### [op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h](op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h)
- `DataCopyPad`
- 其余同步调用已改走 PTO helper

### [op_kernel/unpermute/moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h)
- `DataCopyPad`
- `DataCopyPadExtParams`

### [op_kernel/utils/hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
- `DataCacheCleanAndInvalid`
- local barrier/sync wrapper body
- 核心 remote signal 已是 `pto::comm::TNOTIFY/TWAIT`

### [op_kernel/utils/dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp)
- `DataCopy`
- `LoadData`
- `LoadDataWithTranspose`
- `Fixpipe`
- `Gemm::Tile::*`
- `Gemm::helper::*`

### [op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)
- `DataCopy`
- `PipeBarrier`
- `CrossCoreSetFlag`
- matmul event shell

---

## 七、直接结论
1. Stage 3b 范围内，**最该继续做的 direct 业务接口已经做完**：向量算子和 routing/local sync 都已收口。
2. 当前树上的非 PTO 直接依赖，主量已经不是业务 seam，而是：
   - boundary adapter；
   - cross-core / cache / atomic coordination shell；
   - matmul / fixpipe / Gemm substrate。
3. 如果后续还要继续收口，优先顺序应当是：
   1. `DataCopyPad` boundary adapter 再评估；
   2. helper body 对 `TSYNC/TASSIGN` 的进一步吸收；
   3. 单独处理 matmul/fixpipe substrate。
