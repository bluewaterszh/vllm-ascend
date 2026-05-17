#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"
#include <pto/pto-inst.hpp>

#include "../dispatch_ffn_combine_tiling.h"
#include "../protocol/remote_window.hpp"
#include "../protocol/signal_protocol.hpp"
#include "../protocol/task_plan.hpp"

namespace mc2::v4::dispatch {

using namespace AscendC;

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using PtoGlobal = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int TileElems = 1024>
V4_FORCE_INLINE_AICORE void DispatchTGetContiguous(__gm__ T* dstPtr, __gm__ T* srcPtr, int32_t elemNum)
{
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    using Global = PtoGlobal<T>;
    for (int32_t offset = 0; offset < elemNum; offset += TileElems) {
        int32_t cur = elemNum - offset;
        if (cur > TileElems) {
            cur = TileElems;
        }
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        Global dstG(dstPtr + offset, shape, stride);
        Global srcG(srcPtr + offset, shape, stride);
        TileData tile(1, cur);
        TASSIGN(tile, 0x0);
        pto::comm::TGET(dstG, srcG, tile);
        pipe_barrier(PIPE_ALL);
    }
    dsb(DSB_DDR);
}

template <typename T>
V4_FORCE_INLINE_AICORE void DispatchSelfCopyContiguous(__gm__ T* dstPtr, __gm__ T* srcPtr, int32_t elemNum)
{
    for (int32_t i = 0; i < elemNum; ++i) {
        dstPtr[i] = srcPtr[i];
    }
    dsb(DSB_DDR);
}

template <typename T, int TileElems = 1024>
V4_FORCE_INLINE_AICORE void DispatchTGetViaScratchContiguous(__gm__ T* dstPtr,
                                                             __gm__ T* srcPtr,
                                                             __gm__ T* scratchPtr,
                                                             int32_t elemNum)
{
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    using Global = PtoGlobal<T>;
    for (int32_t offset = 0; offset < elemNum; offset += TileElems) {
        int32_t cur = elemNum - offset;
        if (cur > TileElems) {
            cur = TileElems;
        }
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        Global srcG(srcPtr + offset, shape, stride);
        Global scratchG(scratchPtr + offset, shape, stride);
        TileData stagingTile(1, cur);
        TASSIGN(stagingTile, 0x0);
        pto::comm::TGET(scratchG, srcG, stagingTile);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        DispatchSelfCopyContiguous<T>(dstPtr + offset, scratchPtr + offset, cur);
    }
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunDispatchPullTask(__gm__ protocol::RemoteWindowContext* ctx,
                                                const __gm__ protocol::DispatchPullTask* task)
{
    __gm__ int32_t* signalBase = reinterpret_cast<__gm__ int32_t*>(ctx->workspaceBase + ctx->signalRegionOffset);
    auto ready = pto::comm::Signal(signalBase + protocol::DispatchReadyIndex(task->srcRank, task->srcRowBegin));
    auto done = pto::comm::Signal(signalBase + protocol::DispatchDoneIndex(task->srcRank, task->srcRowBegin));

    __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(ctx->windowIn[task->srcRank] +
                                                             ctx->dispatchRegionOffset +
                                                             task->srcPayloadOffsetBytes);
    __gm__ uint8_t* dst = reinterpret_cast<__gm__ uint8_t*>(ctx->workspaceBase +
                                                             ctx->computeRegionOffset +
                                                             task->dstPayloadOffsetBytes);
    __gm__ uint8_t* scratch = reinterpret_cast<__gm__ uint8_t*>(ctx->workspaceBase +
                                                                 ctx->controlRegionOffset +
                                                                 static_cast<uint64_t>(AscendC::GetBlockIdx()) * task->hiddenBytes);
    pto::comm::TWAIT(ready, task->readyEpoch, pto::comm::WaitCmp::GE);
    if ((task->hiddenBytes & 0x3U) == 0) {
        auto* srcWords = reinterpret_cast<__gm__ uint32_t*>(src);
        auto* dstWords = reinterpret_cast<__gm__ uint32_t*>(dst);
        auto* scratchWords = reinterpret_cast<__gm__ uint32_t*>(scratch);
        const int32_t wordCount = static_cast<int32_t>(task->hiddenBytes / sizeof(uint32_t));
        if (task->srcRank == ctx->rank) {
            DispatchSelfCopyContiguous<uint32_t>(dstWords, srcWords, wordCount);
        } else {
            DispatchTGetViaScratchContiguous<uint32_t>(dstWords, srcWords, scratchWords, wordCount);
        }
    } else {
        if (task->srcRank == ctx->rank) {
            DispatchSelfCopyContiguous<uint8_t>(dst, src, static_cast<int32_t>(task->hiddenBytes));
        } else {
            DispatchTGetViaScratchContiguous<uint8_t>(dst, src, scratch, static_cast<int32_t>(task->hiddenBytes));
        }
    }
    pto::comm::TNOTIFY(done, task->readyEpoch, pto::comm::NotifyOp::Set);
}

V4_FORCE_INLINE_AICORE void RunDispatchOnlyKernel(__gm__ protocol::RemoteWindowContext* ctx,
                                                  __gm__ protocol::DispatchPullTask* tasks,
                                                  const __gm__ StandaloneKernelTilingData* tiling)
{
    const uint32_t taskIdx = AscendC::GetBlockIdx();
    if (taskIdx >= tiling->taskCount) {
        return;
    }
    RunDispatchPullTask(ctx, tasks + taskIdx);
}

}  // namespace mc2::v4::dispatch
#endif
