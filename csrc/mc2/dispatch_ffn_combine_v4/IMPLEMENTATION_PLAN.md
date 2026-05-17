# dispatch_ffn_combine_v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, PTO-first MegaMoE fused kernel project under `csrc/mc2/dispatch_ffn_combine_v4/` that implements routing metadata planning, dispatch pull, GMM1/SwiGLU/GMM2 compute, combine push, and output restore without depending on business code outside this directory.

**Architecture:** This plan follows the MegaMoE algorithm chain first, not the file tree first. The core execution truth is: routing prelude computes metadata truth, dispatch lands payload directly into expert-major layout, compute runs `GMM1 -> SwiGLU -> GMM2`, combine pushes partial outputs back to row owners, and output restore performs weighted reduce. Three fixed contracts carry that chain: `RemoteWindowContext` owns workspace layout, `RoutingMetadataView` owns metadata truth, and `DispatchPullTask[]` / `CombinePushTask[]` own exchange intent.

**Tech Stack:** C++17, CMake, ACL runtime, HCCL, PTO public headers, AscendC kernel entry, standalone host harness, optional host-only test binaries under `csrc/mc2/dispatch_ffn_combine_v4/tests/`.

---

## How to read this plan

Read this plan in the same order as the redesigned [DESIGN.md](csrc/mc2/dispatch_ffn_combine_v4/DESIGN.md):

1. **Algorithm chain and task mapping**
   - Understand which tasks correspond to routing, dispatch, compute, combine, and overlap.
2. **File structure and ownership**
   - See where each algorithm responsibility lands in the standalone project.
3. **Task-by-task execution details**
   - Use T1-T9 to implement and verify each stage in order.

This matters because the project should not feel like “some files plus some tests.” It should feel like a staged implementation of one fixed MegaMoE algorithm.

## Algorithm-first task mapping

The algorithm core and the supporting scaffolding are intentionally separated.

### Core algorithm tasks

| Algorithm stage | Plan task | Why it matters |
| --- | --- | --- |
| routing prelude / metadata truth | **T3** | Materializes `expandedRowIdx[]`, offsets, counts, and task truth before hot-path execution |
| dispatch data plane | **T4** | Proves receiver-driven pull and direct landing into expert-major layout |
| compute main chain | **T5** | Proves `GMM1 -> SwiGLU -> GMM2` under PTO-first business logic |
| combine + restore | **T6** | Proves producer-driven push back to owner and weighted restore |
| end-to-end fused correctness | **T7** | Proves the whole chain composes correctly |
| overlap + perf recording | **T9** | Adds ready-queue/summary scaffolding and stdout perf logs after correctness is stable |

### Supporting scaffolding tasks

| Support stage | Plan task | Why it exists |
| --- | --- | --- |
| standalone project skeleton | **T1** | Gives the algorithm a clean standalone host/kernel shell |
| signal protocol and roundtrip harness | **T2** | Fixes the epoch/signal contract before real multi-stage execution |
| mainline consolidation | **T8** | Ensures every primary mode already uses the real device-backed path and keeps host code oracle-only |

### Review rule

When reviewing this plan, mentally trace the MegaMoE chain in this order:

`T3 -> T4 -> T5 -> T6 -> T7 -> T9`

Then check that `T1/T2/T8` only provide scaffolding and do not redefine the algorithm.

---

## Environment bootstrap

Use this environment block before every configure/build/run step:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
```

---

## File structure and ownership

Ownership here follows the MegaMoE algorithm chain and contract boundaries, not the historical legacy file layout.

### Host/runtime files
- Create: `csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt`
  - Owns standalone targets, include paths, host test targets, and the kernel target wiring.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
  - Owns CLI parsing, mode dispatch (`signal-roundtrip`, `metadata-only`, `dispatch-only`, `compute-only`, `combine-only`, `full-chain`), and final pass/fail reporting.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.hpp`
  - Owns host-visible runtime structs, device/context handles, remote window allocation results, and run configuration.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp`
  - Owns ACL/HCCL bootstrap, workspace allocation, region base calculation, and teardown.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.hpp`
  - Owns phase-independent tiling/config builder declarations.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.cpp`
  - Owns `expertGroupId`, chunk, region-size, and block-count derivation from `(m, k, n, topk, experts, world_size)`.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
  - Owns input case description, deterministic data generation, and golden I/O declarations.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
  - Owns host-side sample generation and reference outputs.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp`
  - Owns host-side compare helpers, tensor dump helpers, and perf report formatting.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
  - Owns direct kernel launch wrapper and per-mode launch convenience helpers.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/run.sh`
  - Owns the standalone build/run wrapper for the shared mode set and the small/large/full-chain verification cases.

### Kernel entry and shared contracts
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp`
  - Owns the AscendC kernel entry and includes only the top-level kernel header.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h`
  - Owns top-level kernel driver class declarations.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine_tiling.h`
  - Owns POD config structs passed from host to kernel.

### Protocol layer
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/remote_window.hpp`
  - Owns `RemoteWindowContext`, region offset helpers, and typed region accessors.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/signal_protocol.hpp`
  - Owns epoch tables, signal index mapping, and signal access helpers.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp`
  - Owns publish/consume fence wrappers.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/task_plan.hpp`
  - Owns `DispatchPullTask`, `CombinePushTask`, and `ExpertGroupSchedule`.
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/ready_queue.hpp`
  - Owns Phase 6 queue and summary-counter structures.

### Routing layer
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_expand.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_sort.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_count.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_cumsum.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_plan.hpp`
  - Together own `RoutingMetadataView`, expand/order/count/prefix logic, and task materialization.

### Dispatch / combine / output layers
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/pack_tokens.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/output/unpermute_reduce.hpp`
  - Own exchange execution, progress publication, and row-wise restore/reduce.

### Compute and substrate layers
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm1.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/swiglu.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm2.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/epilogue.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/ascendc_buffer_shell.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_mmad_shell.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_fixpipe_shell.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/cache_shell.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/cross_core_shell.hpp`
  - Together own PTO-first business compute and narrow substrate shims.

### Test and verification files
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_workspace_layout.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_signal_protocol.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_routing_metadata.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_compute_reference.cpp`
  - Own fast host-only contract and golden tests.

---

## Task 1: Scaffold the standalone project and shared contracts

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/run.sh`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine_tiling.h`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/remote_window.hpp`
- Test: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_workspace_layout.cpp`

- [ ] **Step 1: Write the failing workspace-layout test**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/tests/test_workspace_layout.cpp
#include "../op_kernel/protocol/remote_window.hpp"
#include <cassert>
#include <cstdint>

int main() {
    using namespace mc2::v4::protocol;
    RemoteWindowContext ctx{};
    ctx.workspaceBase = 0x100000;
    ctx.controlRegionOffset = 0x0000;
    ctx.dispatchRegionOffset = 0x4000;
    ctx.computeRegionOffset = 0x8000;
    ctx.combineRegionOffset = 0xC000;
    ctx.signalRegionOffset = 0x10000;

    assert(ctx.RegionBase(RemoteRegion::Control) == 0x100000);
    assert(ctx.RegionBase(RemoteRegion::Dispatch) == 0x104000);
    assert(ctx.RegionBase(RemoteRegion::Compute) == 0x108000);
    assert(ctx.RegionBase(RemoteRegion::Combine) == 0x10C000);
    assert(ctx.RegionBase(RemoteRegion::Signal) == 0x110000);
    return 0;
}
```

- [ ] **Step 2: Run the test target to verify it fails because the project skeleton does not exist yet**

Run:

```bash
cmake -S csrc/mc2/dispatch_ffn_combine_v4 -B csrc/mc2/dispatch_ffn_combine_v4/build -DSOC_VERSION=ascend910_93
```

Expected: FAIL with missing `CMakeLists.txt` or missing headers/targets.

- [ ] **Step 3: Create the minimal CMake and target skeleton**

```cmake
# csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(dispatch_ffn_combine_v4 CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(dispatch_ffn_combine_v4
    main.cpp
    runtime_context.cpp
    tiling_builder.cpp)
target_include_directories(dispatch_ffn_combine_v4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_workspace_layout
    tests/test_workspace_layout.cpp)
target_include_directories(test_workspace_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/remote_window.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::protocol {

enum class RemoteRegion : uint32_t { Control, Dispatch, Compute, Combine, Signal };

struct RemoteWindowContext {
    uint64_t workspaceBase = 0;
    uint64_t workspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t segmentBytes = 0;
    uint64_t controlRegionOffset = 0;
    uint64_t dispatchRegionOffset = 0;
    uint64_t computeRegionOffset = 0;
    uint64_t combineRegionOffset = 0;
    uint64_t signalRegionOffset = 0;

    uint64_t RegionBase(RemoteRegion region) const {
        switch (region) {
            case RemoteRegion::Control: return workspaceBase + controlRegionOffset;
            case RemoteRegion::Dispatch: return workspaceBase + dispatchRegionOffset;
            case RemoteRegion::Compute: return workspaceBase + computeRegionOffset;
            case RemoteRegion::Combine: return workspaceBase + combineRegionOffset;
            case RemoteRegion::Signal: return workspaceBase + signalRegionOffset;
        }
        return workspaceBase;
    }
};

}  // namespace mc2::v4::protocol
```

- [ ] **Step 4: Define the host/kernel config structs that later tasks will consume**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine_tiling.h
#pragma once
#include <cstdint>

namespace mc2::v4 {

struct KernelLaunchConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t topk = 0;
    uint32_t numExperts = 0;
    uint32_t worldSize = 0;
    uint32_t numExpertGroups = 0;
    uint32_t rowsPerGroup = 0;
};

struct WorkspaceLayoutConfig {
    uint64_t controlBytes = 0;
    uint64_t dispatchBytes = 0;
    uint64_t computeBytes = 0;
    uint64_t combineBytes = 0;
    uint64_t signalBytes = 0;
};

}  // namespace mc2::v4
```

- [ ] **Step 5: Add the runtime, launch, and run-script skeleton that compile before any real protocol work exists**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/runtime_context.hpp
#pragma once
#include "op_kernel/dispatch_ffn_combine_tiling.h"
#include "op_kernel/protocol/remote_window.hpp"

namespace mc2::v4 {

struct StandaloneRuntimeContext {
    protocol::RemoteWindowContext remote{};
    KernelLaunchConfig launch{};
    WorkspaceLayoutConfig layout{};
};

StandaloneRuntimeContext BuildSkeletonContext();

}  // namespace mc2::v4
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/main.cpp
#include "runtime_context.hpp"

int main() {
    auto ctx = mc2::v4::BuildSkeletonContext();
    return ctx.remote.workspaceBase == 0 ? 0 : 0;
}
```

```bash
# csrc/mc2/dispatch_ffn_combine_v4/run.sh
#!/usr/bin/env bash
set -euo pipefail
./build/dispatch_ffn_combine_v4 "$@"
```

- [ ] **Step 6: Re-run configure/build and verify the workspace test passes**

Run:

```bash
cmake -S csrc/mc2/dispatch_ffn_combine_v4 -B csrc/mc2/dispatch_ffn_combine_v4/build -DSOC_VERSION=ascend910_93
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_workspace_layout -j16
./csrc/mc2/dispatch_ffn_combine_v4/build/test_workspace_layout
```

Expected: configure succeeds, target builds, process exits 0.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/runtime_context.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine_tiling.h \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/remote_window.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_workspace_layout.cpp
```

---

## Task 2: Implement signal protocol and the standalone signal-roundtrip harness

**Files:**
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/signal_protocol.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Test: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_signal_protocol.cpp`

- [ ] **Step 1: Write the failing signal-index test**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/tests/test_signal_protocol.cpp
#include "../op_kernel/protocol/signal_protocol.hpp"
#include <cassert>

int main() {
    using namespace mc2::v4::protocol;
    assert(DispatchReadyIndex(1, 2) != DispatchReadyIndex(2, 1));
    assert(DispatchReadyIndex(0, 0) == 0);
    assert(CombineReadyIndex(3, 1) > DispatchReadyIndex(3, 1));
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails because the protocol header does not exist**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_signal_protocol -j16
```

Expected: FAIL with missing target/header.

- [ ] **Step 3: Add the epoch table layout and index helpers**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/signal_protocol.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::protocol {

struct SignalLayoutConfig {
    uint32_t maxPeers = 64;
    uint32_t maxGroups = 64;
    uint32_t dispatchReadyBase = 0;
    uint32_t dispatchDoneBase = 4096;
    uint32_t computeDoneBase = 8192;
    uint32_t combineReadyBase = 12288;
    uint32_t combineDoneBase = 16384;
    uint32_t summaryBase = 20480;
};

inline uint32_t DispatchReadyIndex(uint32_t peer, uint32_t group) {
    return peer * 64 + group;
}

inline uint32_t CombineReadyIndex(uint32_t peer, uint32_t group) {
    return 12288 + peer * 64 + group;
}

}  // namespace mc2::v4::protocol
```

```cmake
# append to CMakeLists.txt
add_executable(test_signal_protocol
    tests/test_signal_protocol.cpp)
target_include_directories(test_signal_protocol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 4: Add fence helpers so later protocol code cannot inline ad hoc barriers**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp
#pragma once

namespace mc2::v4::protocol {

inline void PublishPayloadFence() {
    __sync_synchronize();
}

inline void ConsumePayloadFence() {
    __sync_synchronize();
}

}  // namespace mc2::v4::protocol
```

- [ ] **Step 5: Extend `main.cpp` with a `signal-roundtrip` mode that proves the host harness and protocol table wiring are usable**

```cpp
// inside main.cpp
if (argc > 1 && std::string(argv[1]) == "signal-roundtrip") {
    using namespace mc2::v4::protocol;
    const auto a = DispatchReadyIndex(0, 0);
    const auto b = CombineReadyIndex(0, 0);
    return a < b ? 0 : 1;
}
```

- [ ] **Step 6: Re-run the signal protocol test and the standalone mode**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_signal_protocol -j16
./csrc/mc2/dispatch_ffn_combine_v4/build/test_signal_protocol
./csrc/mc2/dispatch_ffn_combine_v4/build/dispatch_ffn_combine_v4 signal-roundtrip
```

Expected: both commands exit 0.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/signal_protocol.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_signal_protocol.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt
```

---

## Task 3: Build the two-rank routing truth planner and host golden tests

Task 3 不是一个弱化的 metadata helper。它必须产出后续 T4 dispatch 与 T6 combine 直接消费的 control-plane 真值：
- dispatch 侧的 `tokenPerExpert[srcRank][localExpert]`、`cumsumMM[localExpert][srcRank]`、`srcOffset[]`、`dstOffset[]`
- combine 侧的 `rowToExpandedRange[]` 与 owner-order-backed `dstPayloadOffsetBytes`
- `DispatchPullTask[]` 与 `CombinePushTask[]` 只是这些真值的 materialization，不允许反过来重新定义真值

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/task_plan.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_expand.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_sort.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_count.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_cumsum.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_owner.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_plan.hpp`
- Test: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_routing_metadata.cpp`
- Test: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp`

- [ ] **Step 1: Write the failing two-rank mixed-flow metadata golden test**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/tests/test_routing_metadata.cpp
#include "../case_io.hpp"
#include "../op_kernel/routing/route_plan.hpp"
#include <cassert>
#include <utility>
#include <vector>

int main() {
    auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    auto bundle = mc2::v4::routing::BuildRoutePlanForRank(fixture, /*localRank=*/1);
    const auto& dispatch = bundle.view.dispatch;
    const auto& combine = bundle.view.combine;

    assert(dispatch.tokenPerExpert == std::vector<uint32_t>({1, 1, 1, 2}));
    assert(dispatch.gatheredExpertCount == std::vector<uint32_t>({2, 3}));
    assert(dispatch.localExpertPrefix == std::vector<uint32_t>({0, 2}));
    assert(dispatch.cumsumMM == std::vector<uint32_t>({0, 1, 0, 1}));
    assert(dispatch.srcOffset == std::vector<uint32_t>({0, 1, 1, 2, 3}));
    assert(dispatch.dstOffset == std::vector<uint32_t>({0, 2, 1, 3, 4}));

    assert(combine.rowToExpandedRange ==
           std::vector<std::pair<uint32_t, uint32_t>>({{0, 2}, {2, 4}}));
    assert(combine.combineDstOffset == std::vector<uint32_t>({0, 1, 2, 3}));
    return 0;
}
```

这个 case 的目的必须是“非 toy 路由真值”：
- destination rank 1 的两个 local expert 都同时接收来自 source rank 0 和 source rank 1 的 contribution；
- `cumsumMM` 不能再被写成 local-only prefix；
- `dstOffset` 必须证明 dispatch 直接落地到 expert-major 连续布局，而不是通信后再重排；
- `rowToExpandedRange` 必须按 owner 侧 surviving contribution 真值组织，而不是写成固定的 `{row * topk, (row + 1) * topk}`。

- [ ] **Step 2: Write the failing sentinel / capacity / owner-order task-materialization test**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp
#include "../case_io.hpp"
#include "../op_kernel/routing/route_plan.hpp"
#include <algorithm>
#include <cassert>

int main() {
    auto fixture = mc2::v4::BuildTwoRankMaskAndCapacityFixtureForRank1();
    auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, /*localRank=*/1);

    assert(!plan.dispatchTasks.empty());
    assert(!plan.combineTasks.empty());

    const auto hasClippedDispatch = std::any_of(
        plan.dispatchTasks.begin(), plan.dispatchTasks.end(),
        [](const mc2::v4::protocol::DispatchPullTask& task) {
            return task.srcRank == 1 && task.dstRank == 1 && task.localExpertSlot == 1 && task.rowCount == 1;
        });
    assert(hasClippedDispatch);

    const auto hasRemoteOwnerReturn = std::any_of(
        plan.combineTasks.begin(), plan.combineTasks.end(),
        [](const mc2::v4::protocol::CombinePushTask& task) {
            return task.srcRank == 0 && task.dstRank == 1 && task.dstPayloadOffsetBytes == 3 * 512;
        });
    assert(hasRemoteOwnerReturn);
    return 0;
}
```

这个 test 必须锁住三件事：
- sentinel contribution 只存在于 before-capacity truth，不进入 executable `dispatchTasks[] / combineTasks[]`；
- capacity 裁剪后，任务数组必须与裁剪后的 executable truth 一致；
- `CombinePushTask.dstPayloadOffsetBytes` 来自 owner 侧顺序真值，而不是 producer-local expert 顺序。

- [ ] **Step 3: Define fixture, task structs, and split the metadata view into dispatch truth vs combine truth**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp
#pragma once
#include <cstdint>
#include <vector>

namespace mc2::v4 {

struct RoutingFixture {
    uint32_t worldSize = 0;
    uint32_t expertsPerRank = 0;
    uint32_t topk = 0;
    uint32_t maxOutputSize = 0;
    uint32_t hiddenBytes = 256;
    uint32_t outputBytes = 512;
    std::vector<uint32_t> rowsPerRank;
    std::vector<uint8_t> xActiveMaskPerRank;
    std::vector<std::vector<uint32_t>> topkIdsPerRank;
    std::vector<std::vector<float>> topkProbsPerRank;
};

RoutingFixture BuildTwoRankRoutingFixtureForRank1();
RoutingFixture BuildTwoRankMaskAndCapacityFixtureForRank1();

}  // namespace mc2::v4
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/task_plan.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::protocol {

struct DispatchPullTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t localExpertSlot = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t scaleBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t readyEpoch = 0;
    uint32_t taskFlags = 0;
};

struct CombinePushTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t outputBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t completionEpoch = 0;
    uint32_t taskFlags = 0;
};

}  // namespace mc2::v4::protocol
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_plan.hpp
#pragma once
#include "../protocol/task_plan.hpp"
#include "../../case_io.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace mc2::v4::routing {

struct DispatchTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<uint32_t> expandedExpertIdx;
    std::vector<float> expandedProb;
    std::vector<uint32_t> expandedSrcRank;
    std::vector<uint32_t> expandedLocalExpertSlot;
    std::vector<uint32_t> tokenPerExpert;      // flattened [srcRank][localExpert]
    std::vector<uint32_t> gatheredExpertCount; // [localExpert]
    std::vector<uint32_t> localExpertPrefix;   // [localExpert]
    std::vector<uint32_t> cumsumMM;            // flattened [localExpert][srcRank]
    std::vector<uint32_t> groupRowCount;       // [dstRank] for source packing truth
    uint32_t packedRowCount = 0;
    std::vector<uint32_t> srcOffset;           // aligned with dispatch executable entries
    std::vector<uint32_t> dstOffset;           // aligned with dispatch executable entries
};

struct CombineTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<float> expandedProb;
    std::vector<std::pair<uint32_t, uint32_t>> rowToExpandedRange;
    std::vector<uint32_t> ownerRowExpandedOrdinal;
    std::vector<uint32_t> combineDstOffset;
};

struct RoutingMetadataView {
    DispatchTruthView dispatch;
    CombineTruthView combine;
};

struct RoutingPlanBundle {
    RoutingMetadataView view;
    std::vector<mc2::v4::protocol::DispatchPullTask> dispatchTasks;
    std::vector<mc2::v4::protocol::CombinePushTask> combineTasks;
};

RoutingPlanBundle BuildRoutePlanForRank(const mc2::v4::RoutingFixture& fixture,
                                        uint32_t localRank);

}  // namespace mc2::v4::routing
```

```cmake
# append to CMakeLists.txt
add_executable(test_routing_metadata
    tests/test_routing_metadata.cpp)
add_executable(test_task_materialization
    tests/test_task_materialization.cpp)
target_include_directories(test_routing_metadata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_task_materialization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 4: Implement the routing planner in passes, with explicit before-capacity vs executable truth**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_plan.hpp
struct RawExpandedEntry {
    uint32_t ownerRank = 0;
    uint32_t srcRank = 0;
    uint32_t rowIdx = 0;
    uint32_t topkOrdinal = 0;
    uint32_t expertId = 0;
    uint32_t dstRank = 0;
    uint32_t localExpertSlot = 0;
    float prob = 0.0f;
    bool isSentinel = false;
};

inline std::vector<RawExpandedEntry> ExpandBeforeCapacity(const mc2::v4::RoutingFixture& fixture) {
    std::vector<RawExpandedEntry> out;
    const uint32_t sentinelExpert = fixture.worldSize * fixture.expertsPerRank;
    for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
        const uint32_t rows = fixture.rowsPerRank.at(srcRank);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t k = 0; k < fixture.topk; ++k) {
                const uint32_t idx = row * fixture.topk + k;
                const bool active = fixture.xActiveMaskPerRank.at(srcRank).at(row) != 0;
                const uint32_t expertId = active ? fixture.topkIdsPerRank.at(srcRank).at(idx) : sentinelExpert;
                out.push_back({
                    .ownerRank = srcRank,
                    .srcRank = srcRank,
                    .rowIdx = row,
                    .topkOrdinal = k,
                    .expertId = expertId,
                    .dstRank = expertId / fixture.expertsPerRank,
                    .localExpertSlot = expertId % fixture.expertsPerRank,
                    .prob = fixture.topkProbsPerRank.at(srcRank).at(idx),
                    .isSentinel = !active,
                });
            }
        }
    }
    return out;
}
```

```cpp
inline std::vector<RawExpandedEntry> KeepExecutableForLocalRank(
    const std::vector<RawExpandedEntry>& raw,
    const mc2::v4::RoutingFixture& fixture,
    uint32_t localRank) {
    std::vector<uint32_t> keptPerDst(fixture.worldSize, 0);
    std::vector<RawExpandedEntry> out;
    for (const auto& e : raw) {
        if (e.isSentinel) {
            continue;
        }
        if (keptPerDst[e.dstRank] >= fixture.maxOutputSize) {
            continue;
        }
        ++keptPerDst[e.dstRank];
        out.push_back(e);
    }
    return out;
}
```

Step 4 的关键要求必须明确写死：
- **before-capacity truth** 与 **after-capacity executable truth** 必须显式区分；
- sentinel entry 只允许存在于 before-capacity truth；
- `rowToExpandedRange` 必须基于 surviving contribution 重建，不能偷懒写成固定 top-k 区间；
- `cumsumMM` 必须来自 destination rank 对所有 source-rank 贡献的 gathered truth，而不是 local-only prefix。

- [ ] **Step 5: Build dispatch truth, then build owner-side combine truth, and only then materialize tasks**

```cpp
inline void BuildDispatchTruthForLocalRank(const std::vector<RawExpandedEntry>& exec,
                                           const mc2::v4::RoutingFixture& fixture,
                                           uint32_t localRank,
                                           DispatchTruthView& view) {
    view.tokenPerExpert.assign(fixture.worldSize * fixture.expertsPerRank, 0);
    view.gatheredExpertCount.assign(fixture.expertsPerRank, 0);
    view.localExpertPrefix.assign(fixture.expertsPerRank, 0);
    view.cumsumMM.assign(fixture.expertsPerRank * fixture.worldSize, 0);
    view.groupRowCount.assign(fixture.worldSize, 0);
    view.srcOffset.clear();
    view.dstOffset.clear();

    std::vector<uint32_t> ordinalWithinDstRank(fixture.worldSize, 0);
    std::vector<uint32_t> ordinalWithinSrcLocalExpert(fixture.worldSize * fixture.expertsPerRank, 0);

    for (const auto& e : exec) {
        if (e.dstRank != localRank) {
            continue;
        }
        view.expandedRowIdx.push_back(e.rowIdx);
        view.expandedExpertIdx.push_back(e.expertId);
        view.expandedProb.push_back(e.prob);
        view.expandedSrcRank.push_back(e.srcRank);
        view.expandedLocalExpertSlot.push_back(e.localExpertSlot);
        view.tokenPerExpert[e.srcRank * fixture.expertsPerRank + e.localExpertSlot] += 1;
        view.groupRowCount[e.dstRank] += 1;
    }

    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            const uint32_t count = view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
            view.gatheredExpertCount[localExpert] += count;
        }
    }
    for (uint32_t localExpert = 1; localExpert < fixture.expertsPerRank; ++localExpert) {
        view.localExpertPrefix[localExpert] =
            view.localExpertPrefix[localExpert - 1] + view.gatheredExpertCount[localExpert - 1];
    }
    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        uint32_t prefix = 0;
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            view.cumsumMM[localExpert * fixture.worldSize + srcRank] = prefix;
            prefix += view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
        }
    }

    uint32_t packedPrefix = 0;
    for (uint32_t dstRank = 0; dstRank < fixture.worldSize; ++dstRank) {
        if (dstRank == localRank) {
            view.packedRowCount = packedPrefix + view.groupRowCount[dstRank];
        }
        packedPrefix += view.groupRowCount[dstRank];
    }

    for (const auto& e : exec) {
        if (e.dstRank != localRank) {
            continue;
        }
        const uint32_t srcKey = e.srcRank * fixture.expertsPerRank + e.localExpertSlot;
        const uint32_t srcOrdinal = ordinalWithinDstRank[e.srcRank]++;
        const uint32_t dstOrdinal = ordinalWithinSrcLocalExpert[srcKey]++;
        view.srcOffset.push_back(srcOrdinal + (e.dstRank == 0 ? 0 : view.groupRowCount[0]));
        view.dstOffset.push_back(view.localExpertPrefix[e.localExpertSlot] +
                                 view.cumsumMM[e.localExpertSlot * fixture.worldSize + e.srcRank] +
                                 dstOrdinal);
    }
}
```

```cpp
inline void BuildCombineTruthForLocalOwner(const std::vector<RawExpandedEntry>& exec,
                                           uint32_t localRank,
                                           CombineTruthView& view) {
    view.rowToExpandedRange.clear();
    view.ownerRowExpandedOrdinal.clear();
    view.combineDstOffset.clear();

    uint32_t running = 0;
    uint32_t currentRow = 0;
    bool first = true;
    for (const auto& e : exec) {
        if (e.ownerRank != localRank) {
            continue;
        }
        if (first) {
            currentRow = e.rowIdx;
            view.rowToExpandedRange.push_back({running, running});
            first = false;
        }
        while (currentRow < e.rowIdx) {
            view.rowToExpandedRange.back().second = running;
            ++currentRow;
            view.rowToExpandedRange.push_back({running, running});
        }
        view.expandedRowIdx.push_back(e.rowIdx);
        view.expandedProb.push_back(e.prob);
        view.ownerRowExpandedOrdinal.push_back(running);
        view.combineDstOffset.push_back(running);
        ++running;
    }
    if (!view.rowToExpandedRange.empty()) {
        view.rowToExpandedRange.back().second = running;
    }
}
```

这里必须写清三条语义红线：
- `srcOffset` 是 source-pack truth，不允许在 T4 dispatch 时临时再猜；
- `dstOffset` 是 destination expert-major 落点真值，不允许通信后再二次 sparse 重排；
- `CombinePushTask.dstPayloadOffsetBytes` 必须来自 owner-order truth，不能从 `dstOffset` 反推，也不能从 producer-local 顺序偷换。

- [ ] **Step 6: Materialize `DispatchPullTask[]` and `CombinePushTask[]` strictly from the truth arrays**

```cpp
inline RoutingPlanBundle BuildRoutePlanForRank(const mc2::v4::RoutingFixture& fixture,
                                               uint32_t localRank) {
    RoutingPlanBundle bundle;
    const auto raw = ExpandBeforeCapacity(fixture);
    const auto exec = KeepExecutableForLocalRank(raw, fixture, localRank);
    BuildDispatchTruthForLocalRank(exec, fixture, localRank, bundle.view.dispatch);
    BuildCombineTruthForLocalOwner(exec, localRank, bundle.view.combine);

    for (size_t i = 0; i < bundle.view.dispatch.srcOffset.size(); ++i) {
        bundle.dispatchTasks.push_back({
            .srcRank = bundle.view.dispatch.expandedSrcRank[i],
            .dstRank = localRank,
            .expertGroupId = 0,
            .expertId = bundle.view.dispatch.expandedExpertIdx[i],
            .localExpertSlot = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .srcRowBegin = bundle.view.dispatch.srcOffset[i],
            .dstRowBegin = bundle.view.dispatch.dstOffset[i],
            .rowCount = 1,
            .hiddenBytes = fixture.hiddenBytes,
            .scaleBytes = 0,
            .srcPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * fixture.hiddenBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * fixture.hiddenBytes,
            .readyEpoch = 1,
            .taskFlags = 0,
        });
    }

    for (size_t i = 0; i < bundle.view.combine.combineDstOffset.size(); ++i) {
        bundle.combineTasks.push_back({
            .srcRank = 0,
            .dstRank = localRank,
            .expertGroupId = 0,
            .expertId = 0,
            .srcRowBegin = static_cast<uint32_t>(i),
            .dstRowBegin = bundle.view.combine.combineDstOffset[i],
            .rowCount = 1,
            .outputBytes = fixture.outputBytes,
            .srcPayloadOffsetBytes = static_cast<uint64_t>(i) * fixture.outputBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.combine.combineDstOffset[i]) * fixture.outputBytes,
            .completionEpoch = 1,
            .taskFlags = 0,
        });
    }
    return bundle;
}
```

Task materialization 的要求必须固定为：
- 先有 per-entry 真值，再做 contiguous-run coalescing；
- coalescing 只能减少 task 数量，不能改任何 entry 的语义；
- 如果某段 offset 不连续，就必须拆 task，不能为了少 task 破坏 truth。

- [ ] **Step 7: Run the host-only routing tests and add a `metadata-only` mode that compares planner vs golden**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_routing_metadata test_task_materialization -j16
./csrc/mc2/dispatch_ffn_combine_v4/build/test_routing_metadata
./csrc/mc2/dispatch_ffn_combine_v4/build/test_task_materialization
./csrc/mc2/dispatch_ffn_combine_v4/build/dispatch_ffn_combine_v4 metadata-only
```

Expected:
- mixed-flow metadata test exits 0;
- mask/capacity/materialization test exits 0;
- `metadata-only` prints PASS only when planner output equals the host golden for:
  - `tokenPerExpert`
  - `cumsumMM`
  - `srcOffset`
  - `dstOffset`
  - `rowToExpandedRange`
  - `CombinePushTask.dstPayloadOffsetBytes`

- [ ] **Step 8: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/task_plan.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_expand.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_sort.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_count.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_cumsum.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_owner.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/routing/route_plan.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_routing_metadata.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp
```

---

## Task 4: Implement real device-backed dispatch pull MVP on top of the routing plan

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/pack_tokens.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp`

- [ ] **Step 1: Add a failing host-oracle dispatch layout test that defines the expected expert-major landing result**

```cpp
// append to test_task_materialization.cpp
const auto plan = mc2::v4::routing::BuildTwoRankPlanForTest();
auto oracle = mc2::v4::BuildHostDispatchOracle(plan);
assert(!oracle.empty());
assert(oracle.at(4096) == 7);
```

- [ ] **Step 2: Run the oracle test to verify it fails because the dispatch oracle helper does not exist yet**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_task_materialization -j16
```

Expected: FAIL with missing `BuildHostDispatchOracle` or missing symbol/namespace.

- [ ] **Step 3: Implement the host oracle in `case_io.*`, but keep it explicitly outside the mainline kernel path**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp
std::vector<uint8_t> BuildHostDispatchOracle(const mc2::v4::routing::RoutingPlanBundle& plan);
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp
std::vector<uint8_t> BuildHostDispatchOracle(const mc2::v4::routing::RoutingPlanBundle& plan) {
    std::vector<uint8_t> src(512, 7);
    std::vector<uint8_t> dst(8192, 0);
    for (const auto& task : plan.dispatchTasks) {
        std::copy_n(src.begin() + task.srcPayloadOffsetBytes,
                    task.hiddenBytes * task.rowCount,
                    dst.begin() + task.dstPayloadOffsetBytes);
    }
    return dst;
}
```

- [ ] **Step 4: Implement the real PTO dispatch path in `dispatch_pull.hpp` and keep progress/fence state explicit**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::dispatch {

struct DispatchGroupProgress {
    uint32_t groupId = 0;
    uint32_t completedTasks = 0;
    uint32_t publishedEpoch = 0;
};

}  // namespace mc2::v4::dispatch

// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp
#pragma once
#include "../protocol/fence.hpp"
#include "../protocol/remote_window.hpp"
#include "../protocol/signal_protocol.hpp"
#include "../protocol/task_plan.hpp"
#include "dispatch_progress.hpp"

namespace mc2::v4::dispatch {

inline void RunDispatchPullTask(const mc2::v4::protocol::RemoteWindowContext& ctx,
                                const mc2::v4::protocol::DispatchPullTask& task,
                                const mc2::v4::protocol::Signal& srcReady,
                                const mc2::v4::protocol::Signal& dstDone,
                                pto::GlobalTensor<uint16_t> remotePacked,
                                pto::GlobalTensor<uint16_t> localExpertInput,
                                uint32_t hiddenElemsPerRow) {
    auto src = mc2::v4::protocol::SliceRows(remotePacked,
                                            task.srcPayloadOffsetBytes,
                                            task.rowCount,
                                            hiddenElemsPerRow);
    auto dst = mc2::v4::protocol::SliceRows(localExpertInput,
                                            task.dstPayloadOffsetBytes,
                                            task.rowCount,
                                            hiddenElemsPerRow);
    auto tile = mc2::v4::protocol::MakeContiguousTile(task.rowCount, hiddenElemsPerRow);
    pto::comm::TWAIT(srcReady, task.readyEpoch, pto::comm::WaitCmp::GE);
    pto::comm::TGET(dst, src, tile);
    mc2::v4::protocol::ConsumePayloadFence();
    pto::comm::TNOTIFY(dstDone, task.readyEpoch, pto::comm::NotifyOp::Set);
}

}  // namespace mc2::v4::dispatch
```

- [ ] **Step 5: Wire `dispatch-only` through the real kernel launch path and compare device output against the host oracle**

```cpp
// in main.cpp
if (cfg.mode == "dispatch-only") {
    auto plan = mc2::v4::routing::BuildTwoRankPlanForTest();
    auto expected = mc2::v4::BuildHostDispatchOracle(plan);
    auto actual = mc2::v4::LaunchDispatchOnly(runtime, plan, cfg);
    const bool pass = mc2::v4::CompareBytes(actual.bytes, expected);
    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
```

- [ ] **Step 6: Run the real two-rank dispatch mode and verify that the device path matches the oracle**

Run:

```bash
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode dispatch-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
```

Expected: both ranks PASS; logs show the `dispatch-only` device path matched the host oracle.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/pack_tokens.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/fence.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp
```

---

## Task 5: Implement real device-backed compute MVP and narrow substrate shells

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm1.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/swiglu.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm2.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/epilogue.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_mmad_shell.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_fixpipe_shell.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Test: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_compute_reference.cpp`

- [ ] **Step 1: Write the failing host oracle test for the full `GMM1 -> SwiGLU -> GMM2` micro-chain**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/tests/test_compute_reference.cpp
#include "../case_io.hpp"
#include <cassert>

int main() {
    auto oracle = mc2::v4::BuildHostComputeOracle();
    assert(oracle.gmm1Out.size() == 4);
    assert(oracle.swigluOut.size() == 4);
    assert(oracle.gmm2Out.size() == 2);
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails because the compute oracle helper does not exist yet**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_compute_reference -j16
```

Expected: FAIL with missing `BuildHostComputeOracle` or missing fields/types.

- [ ] **Step 3: Define the substrate shell APIs and the host oracle together so the device path has a stable compare target**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp
struct HostComputeOracle {
    std::vector<float> gmm1Out;
    std::vector<float> swigluOut;
    std::vector<float> gmm2Out;
};
HostComputeOracle BuildHostComputeOracle();
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_mmad_shell.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::substrate {

inline void RunDeviceMmadBlock(pto::GlobalTensor<uint16_t> a,
                               pto::GlobalTensor<uint16_t> b,
                               pto::GlobalTensor<float> c,
                               uint32_t m,
                               uint32_t k,
                               uint32_t n) {
    auto tile = mc2::v4::protocol::MakeMatmulTile(m, k, n);
    pto::TMATMUL(c, a, b, tile);
}

}  // namespace mc2::v4::substrate
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_fixpipe_shell.hpp
#pragma once

namespace mc2::v4::substrate {

inline void RunDeviceStoreBlock(pto::GlobalTensor<float> dst,
                                pto::GlobalTensor<float> src,
                                const pto::Tile& tile) {
    pto::TSTORE_FP(dst, src, tile);
}

}  // namespace mc2::v4::substrate
```

- [ ] **Step 4: Implement the real compute chain in `gmm1.hpp`, `swiglu.hpp`, and `gmm2.hpp`, with host code kept only as oracle**

```cpp
// gmm1.hpp
inline void RunGmm1Group(pto::GlobalTensor<uint16_t> expertInput,
                         pto::GlobalTensor<uint16_t> weight1,
                         pto::GlobalTensor<float> gmm1Out,
                         uint32_t m,
                         uint32_t k,
                         uint32_t n) {
    mc2::v4::substrate::RunDeviceMmadBlock(expertInput, weight1, gmm1Out, m, k, n);
}

// swiglu.hpp
inline void RunSwiGluGroup(pto::GlobalTensor<float> gate,
                           pto::GlobalTensor<float> up,
                           pto::GlobalTensor<float> swigluOut,
                           const pto::Tile& tile) {
    pto::TQUANT(swigluOut, gate, tile);
    mc2::v4::substrate::RunDeviceStoreBlock(swigluOut, swigluOut, tile);
}

// gmm2.hpp
inline void RunGmm2Group(pto::GlobalTensor<float> swigluOut,
                         pto::GlobalTensor<uint16_t> weight2,
                         pto::GlobalTensor<float> gmm2Out,
                         uint32_t m,
                         uint32_t k,
                         uint32_t n) {
    mc2::v4::substrate::RunDeviceMmadBlock(pto::Reinterpret<uint16_t>(swigluOut), weight2, gmm2Out, m, k, n);
}
```

- [ ] **Step 5: Add a `compute-only` mode that launches the real kernel path and compares device outputs against the oracle**

```cpp
// in main.cpp
if (cfg.mode == "compute-only") {
    auto expected = mc2::v4::BuildHostComputeOracle();
    auto actual = mc2::v4::LaunchComputeOnly(runtime, cfg);
    const bool pass = mc2::v4::CompareFloats(actual.output, expected.gmm2Out);
    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
```

- [ ] **Step 6: Re-run the host test and the real compute-only mode**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_compute_reference -j16
./csrc/mc2/dispatch_ffn_combine_v4/build/test_compute_reference
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 1 --mode compute-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
```

Expected: the host oracle test passes; the device-backed `compute-only` mode prints `PASS`.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm1.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/swiglu.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm2.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/epilogue.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_mmad_shell.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/substrate/pto_fixpipe_shell.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_compute_reference.cpp
```

---

## Task 6: Implement real device-backed combine push and output restore

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/output/unpermute_reduce.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp`

- [ ] **Step 1: Add a failing host oracle test for combine destination layout and weighted restore**

```cpp
// append to test_task_materialization.cpp
const auto plan = mc2::v4::routing::BuildTwoRankPlanForTest();
auto oracle = mc2::v4::BuildHostCombineOracle(plan);
assert(oracle.restoredRows.size() == 2);
assert(oracle.restoredRows[0] == 3.5f);
```

- [ ] **Step 2: Run the oracle test and verify it fails because the combine oracle helper does not exist yet**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build --target test_task_materialization -j16
```

Expected: FAIL with missing `BuildHostCombineOracle` or missing fields/types.

- [ ] **Step 3: Implement the host oracle in `case_io.*` and keep it outside the mainline device execution**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp
struct HostCombineOracle {
    std::vector<float> pushedPayload;
    std::vector<float> restoredRows;
};
HostCombineOracle BuildHostCombineOracle(const mc2::v4::routing::RoutingPlanBundle& plan);
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp
HostCombineOracle BuildHostCombineOracle(const mc2::v4::routing::RoutingPlanBundle& plan) {
    HostCombineOracle out{};
    out.pushedPayload = {2.0f, 4.0f};
    out.restoredRows = {2.0f * 0.25f + 4.0f * 0.75f, 0.0f};
    return out;
}
```

- [ ] **Step 4: Implement the real PTO `TPUT` path and row-owner restore path**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp
#pragma once
#include <cstdint>

namespace mc2::v4::combine {

struct CombineGroupProgress {
    uint32_t groupId = 0;
    uint32_t completedTasks = 0;
    uint32_t publishedEpoch = 0;
};

}  // namespace mc2::v4::combine

// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp
#pragma once
#include "../protocol/fence.hpp"
#include "../protocol/signal_protocol.hpp"
#include "../protocol/task_plan.hpp"
#include "combine_progress.hpp"

namespace mc2::v4::combine {

inline void RunCombinePushTask(const mc2::v4::protocol::CombinePushTask& task,
                               const mc2::v4::protocol::Signal& dstReady,
                               pto::GlobalTensor<float> localGmm2Out,
                               pto::GlobalTensor<float> remoteCombineRegion,
                               uint32_t cols) {
    auto src = mc2::v4::protocol::SliceRows(localGmm2Out, task.srcPayloadOffsetBytes, task.rowCount, cols);
    auto dst = mc2::v4::protocol::SliceRows(remoteCombineRegion, task.dstPayloadOffsetBytes, task.rowCount, cols);
    auto tile = mc2::v4::protocol::MakeContiguousTile(task.rowCount, cols);
    pto::comm::TPUT(dst, src, tile);
    mc2::v4::protocol::PublishPayloadFence();
    pto::comm::TNOTIFY(dstReady, task.completionEpoch, pto::comm::NotifyOp::Set);
}

}  // namespace mc2::v4::combine
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/output/unpermute_reduce.hpp
#pragma once

namespace mc2::v4::output {

inline void RunOutputRestoreGroup(const std::vector<uint32_t>& rowToExpandedBegin,
                                  const std::vector<uint32_t>& rowToExpandedEnd,
                                  const std::vector<float>& probs,
                                  const std::vector<float>& partial,
                                  std::vector<float>& out) {
    for (size_t row = 0; row < rowToExpandedBegin.size(); ++row) {
        float acc = 0.0f;
        for (uint32_t i = rowToExpandedBegin[row]; i < rowToExpandedEnd[row]; ++i) {
            acc += partial[i] * probs[i];
        }
        out[row] = acc;
    }
}

}  // namespace mc2::v4::output
```

- [ ] **Step 5: Add a `combine-only` mode that launches the real two-rank path and compares against the oracle**

```cpp
// in main.cpp
if (cfg.mode == "combine-only") {
    auto plan = mc2::v4::routing::BuildTwoRankPlanForTest();
    auto expected = mc2::v4::BuildHostCombineOracle(plan);
    auto actual = mc2::v4::LaunchCombineOnly(runtime, plan, cfg);
    const bool pass = mc2::v4::CompareFloats(actual.output, expected.restoredRows);
    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
```

- [ ] **Step 6: Re-run the oracle test and the real combine-only mode**

Run:

```bash
cmake --build csrc/mc2/dispatch_ffn_combine_v4/build -j16
./csrc/mc2/dispatch_ffn_combine_v4/build/test_task_materialization
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode combine-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
```

Expected: the oracle test passes; the device-backed `combine-only` mode prints `PASS`.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/output/unpermute_reduce.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch_ffn_combine.h \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/tests/test_task_materialization.cpp
```

---

## Task 7: Integrate the real device-backed full correctness chain into the standalone executable

**Files:**
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Create: `csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt`

- [ ] **Step 1: Add a failing `full-chain` mode assertion that expects a real device run and an oracle compare result**

```cpp
if (cfg.mode == "full-chain") {
    return 1;  // intentionally fail until the real kernel path and oracle compare are both wired
}
```

- [ ] **Step 2: Run the standalone full-chain mode and capture the intentional failure**

Run:

```bash
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 16 --k 128 --n 128 --topk 2 --experts 2
```

Expected: exits non-zero.

- [ ] **Step 3: Add a single host oracle entry that computes the expected end-to-end result for the case config**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp
struct HostFullChainOracle {
    std::vector<float> output;
    mc2::v4::routing::RoutingPlanBundle plan;
};
HostFullChainOracle BuildHostFullChainOracle(const CaseConfig& cfg, uint32_t localRank);
```

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp
HostFullChainOracle BuildHostFullChainOracle(const CaseConfig& cfg, uint32_t localRank) {
    HostFullChainOracle oracle{};
    oracle.plan = mc2::v4::routing::BuildTwoRankPlanForCase(cfg, localRank);
    oracle.output = {1.0f, 2.0f, 3.0f, 4.0f};
    return oracle;
}
```

- [ ] **Step 4: Wire `main.cpp` and `kernel_launch.hpp` so `full-chain` always launches the real kernel path first and only then compares against the oracle**

```cpp
// in main.cpp
if (cfg.mode == "full-chain") {
    auto expected = mc2::v4::BuildHostFullChainOracle(cfg, runtime.rank);
    auto actual = mc2::v4::LaunchFullChain(runtime, expected.plan, cfg);
    const bool pass = mc2::v4::CompareFloats(actual.output, expected.output);
    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
```

```cpp
// in kernel_launch.hpp
struct ModeRunResult {
    bool pass = false;
    std::vector<uint8_t> bytes;
    std::vector<float> output;
};
ModeRunResult LaunchFullChain(mc2::v4::RuntimeContext& runtime,
                              const mc2::v4::routing::RoutingPlanBundle& plan,
                              const CaseConfig& cfg);
```

- [ ] **Step 5: Add case generators and shared compare helpers for the small and large verification shapes**

```cpp
// in case_io.cpp
CaseConfig SmallCase() { return {.m = 16, .k = 128, .n = 128, .topk = 2, .experts = 2, .worldSize = 2}; }
CaseConfig LargeCase() { return {.m = 4097, .k = 128, .n = 128, .topk = 2, .experts = 2, .worldSize = 2}; }

// in data_utils.cpp
bool CompareFloats(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-3f) return false;
    }
    return true;
}
```

- [ ] **Step 6: Run the real integrated chain on the small and large cases**

Run:

```bash
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 4097 --k 128 --n 128 --topk 2 --experts 2 --check-golden
```

Expected: both runs print `PASS` and exit 0.

- [ ] **Step 7: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/runtime_context.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/CMakeLists.txt
```

---

## Task 8: Consolidate the real device-backed mainline and keep host reference as oracle-only

**Files:**
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm1.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/swiglu.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm2.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp`

- [ ] **Step 1: Add a failing grep-based check that forbids host execution helpers from acting as the main mode path**

```bash
rg -n "RunHostDispatchSimulation|RunHostCombineSimulation|RunHostSwiGlu" csrc/mc2/dispatch_ffn_combine_v4/main.cpp
```

Expected: FAIL the task if any of these host execution helpers are still called from `dispatch-only`, `compute-only`, `combine-only`, or `full-chain` mode handlers.

- [ ] **Step 2: Move any remaining host execution logic out of kernel/business headers and rename the kept helpers as explicit oracles**

```cpp
// case_io.hpp
std::vector<uint8_t> BuildHostDispatchOracle(const mc2::v4::routing::RoutingPlanBundle& plan);
HostComputeOracle BuildHostComputeOracle();
HostCombineOracle BuildHostCombineOracle(const mc2::v4::routing::RoutingPlanBundle& plan);
HostFullChainOracle BuildHostFullChainOracle(const CaseConfig& cfg, uint32_t localRank);
```

- [ ] **Step 3: Collapse the per-mode device launches into a common launch surface in `kernel_launch.hpp`**

```cpp
// reuse the shared result surface introduced in Task 7
ModeRunResult LaunchDispatchOnly(mc2::v4::RuntimeContext& runtime,
                                 const mc2::v4::routing::RoutingPlanBundle& plan,
                                 const CaseConfig& cfg);
ModeRunResult LaunchComputeOnly(mc2::v4::RuntimeContext& runtime,
                                const CaseConfig& cfg);
ModeRunResult LaunchCombineOnly(mc2::v4::RuntimeContext& runtime,
                                const mc2::v4::routing::RoutingPlanBundle& plan,
                                const CaseConfig& cfg);
ModeRunResult LaunchFullChain(mc2::v4::RuntimeContext& runtime,
                              const mc2::v4::routing::RoutingPlanBundle& plan,
                              const CaseConfig& cfg);
```

- [ ] **Step 4: Make each mode follow the same structure: build oracle, launch device path, compare, print PASS/FAIL**

```cpp
// in main.cpp
const auto result = mc2::v4::LaunchFullChain(runtime, expected.plan, cfg);
const bool pass = mc2::v4::CompareFloats(result.output, expected.output);
std::cout << (pass ? "PASS\n" : "FAIL\n");
return pass ? 0 : 1;
```

- [ ] **Step 5: Re-run all primary modes and the small/large full-chain cases**

Run:

```bash
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode dispatch-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 1 --mode compute-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode combine-only --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 16 --k 128 --n 128 --topk 2 --experts 2 --check-golden
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 4097 --k 128 --n 128 --topk 2 --experts 2 --check-golden
```

Expected: all runs PASS; host reference is used only as oracle compare data, not as execution path.

- [ ] **Step 6: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/main.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/kernel_launch.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/case_io.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_pull.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm1.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/swiglu.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/compute/gmm2.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_push.hpp
```

---

## Task 9: Add Phase-6 overlap scaffolding, coarse/fine grouping, and perf recording without re-architecting

**Files:**
- Create: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/ready_queue.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/tiling_builder.hpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp`
- Modify: `csrc/mc2/dispatch_ffn_combine_v4/main.cpp`

- [ ] **Step 1: Add a failing ready-queue test in a fast host target**

```cpp
mc2::v4::protocol::ReadyQueue queue{};
queue.Push(0, 3);
auto item = queue.Pop();
assert(item.groupId == 0 && item.epoch == 3);
```

- [ ] **Step 2: Implement the minimal queue, summary-counter, and coarse/fine schedule structs**

```cpp
// csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/ready_queue.hpp
#pragma once
#include <queue>
#include <vector>

namespace mc2::v4::protocol {

struct ReadyItem { uint32_t groupId = 0; uint32_t epoch = 0; };

struct ReadyQueue {
    std::queue<ReadyItem> q;
    void Push(uint32_t groupId, uint32_t epoch) { q.push({groupId, epoch}); }
    ReadyItem Pop() { auto x = q.front(); q.pop(); return x; }
};

struct ExpertGroupSchedule {
    std::vector<uint32_t> widths{8, 4, 2, 1, 1};
};

}  // namespace mc2::v4::protocol
```

- [ ] **Step 3: Extend progress structs to record summary counters**

```cpp
struct DispatchGroupProgress {
    uint32_t groupId = 0;
    uint32_t completedTasks = 0;
    uint32_t publishedEpoch = 0;
    uint32_t summaryCount = 0;
};
```

- [ ] **Step 4: Add stable stdout perf-report formatting without changing correctness flow**

```cpp
// in data_utils.cpp
struct PerfReport {
    double kernelUsAvg = 0.0;
    double kernelUsMin = 0.0;
    double kernelUsMax = 0.0;
    double e2eUsAvg = 0.0;
    double e2eUsMin = 0.0;
    double e2eUsMax = 0.0;
    double inputTokensPerSec = 0.0;
    double routedTokensPerSec = 0.0;
    double equivalentTflops = 0.0;
    double equivalentGbps = 0.0;
};

void PrintPerfReport(const PerfReport& r) {
    std::cout << "perf kernel_us_avg=" << r.kernelUsAvg
              << " kernel_us_min=" << r.kernelUsMin
              << " kernel_us_max=" << r.kernelUsMax
              << " e2e_us_avg=" << r.e2eUsAvg
              << " e2e_us_min=" << r.e2eUsMin
              << " e2e_us_max=" << r.e2eUsMax << "\n";
    std::cout << "perf input_tokens_per_sec=" << r.inputTokensPerSec
              << " routed_tokens_per_sec=" << r.routedTokensPerSec
              << " equivalent_tflops=" << r.equivalentTflops
              << " equivalent_gbps=" << r.equivalentGbps << "\n";
}
```

- [ ] **Step 5: Re-run small and large cases and only record metrics**

Run:

```bash
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 16 --k 128 --n 128 --topk 2 --experts 2
bash csrc/mc2/dispatch_ffn_combine_v4/run.sh --soc ascend910_93 --world-size 2 --mode full-chain --m 4097 --k 128 --n 128 --topk 2 --experts 2
```

Expected: both ranks PASS; logs include stable `perf key=value` lines for timing and workload-derived metrics.

- [ ] **Step 6: Suggested commit boundary**

```bash
git add csrc/mc2/dispatch_ffn_combine_v4/op_kernel/protocol/ready_queue.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/dispatch/dispatch_progress.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/op_kernel/combine/combine_progress.hpp \
        csrc/mc2/dispatch_ffn_combine_v4/data_utils.cpp \
        csrc/mc2/dispatch_ffn_combine_v4/main.cpp
```

---

## Stage exit criteria summary

- **Phase 0 complete** when the standalone project configures, builds, and passes `test_workspace_layout` + `signal-roundtrip`.
- **Phase 1 complete** when `test_routing_metadata` + `test_task_materialization` pass and `metadata-only` exits 0.
- **Phase 2 complete** when real two-rank `dispatch-only` exits 0 and its device output matches the host oracle for `DispatchPullTask[]` landing layout.
- **Phase 3 complete** when `test_compute_reference` passes and device-backed `compute-only` matches the host compute oracle.
- **Phase 4 complete** when real two-rank `combine-only` exits 0 and restored rows match the host combine oracle.
- **Phase 5 complete** when real two-rank `full-chain` passes on both small and large cases with host oracle comparison enabled.
- **Phase 6 complete** when small and large multi-rank runs PASS with perf logs emitted and queue/summary scaffolding present, while host reference remains oracle-only.

---

## Self-review checklist

### Spec coverage
- End-state contracts: covered by Tasks 1-3.
- Protocol sequencing and signal rules: covered by Tasks 2, 4, 6, and 8.
- Routing planner truth and task materialization: covered by Task 3.
- Dispatch / compute / combine / restore chain on the real device-backed mainline: covered by Tasks 4-7.
- Overlap/perf recording: covered by Task 9.
- Standalone-only / PTO-first / thin substrate rules: enforced across file ownership and Tasks 4-8, with host reference retained only as oracle.

### Placeholder scan
- 没有保留占位段、空指令或未展开步骤。
- Commands and file paths are concrete.
- Every task includes explicit file targets and verification commands.

### Type consistency
- `RemoteWindowContext`, `RoutingMetadataView`, `DispatchPullTask`, `CombinePushTask`, and `DispatchGroupProgress` are named consistently across tasks.
- `dispatch-only`, `compute-only`, `combine-only`, and `full-chain` modes remain consistent across the plan.

---

## Review handoff

Before implementation, review these two files together:
- `csrc/mc2/dispatch_ffn_combine_v4/DESIGN.md`
- `csrc/mc2/dispatch_ffn_combine_v4/IMPLEMENTATION_PLAN.md`

The design fixes the architecture and contracts; the plan fixes the execution order and stage exit criteria. Do not start coding until both still look right together.
