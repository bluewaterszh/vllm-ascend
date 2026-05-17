#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"
#include <pto/pto-inst.hpp>

#include "combine_progress.hpp"
#include "../dispatch_ffn_combine_tiling.h"
#include "../output/unpermute_reduce.hpp"
#include "../protocol/remote_window.hpp"
#include "../protocol/task_plan.hpp"

namespace mc2::v4::combine {

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using CombineGlobal = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using CombineTile = pto::Tile<pto::TileType::Vec, float, 1, mc2::v4::protocol::kCombineTransportElems, pto::BLayout::RowMajor, -1, -1>;

V4_FORCE_INLINE_AICORE void RunCombinePutSlot(__gm__ float* dstPtr, __gm__ float* srcPtr)
{
    ShapeDyn shape(1, 1, 1, 1, mc2::v4::protocol::kCombineTransportElems);
    StrideDyn stride(mc2::v4::protocol::kCombineTransportElems,
                     mc2::v4::protocol::kCombineTransportElems,
                     mc2::v4::protocol::kCombineTransportElems,
                     mc2::v4::protocol::kCombineTransportElems,
                     1);
    CombineGlobal dstG(dstPtr, shape, stride);
    CombineGlobal srcG(srcPtr, shape, stride);
    CombineTile tile(1, mc2::v4::protocol::kCombineTransportElems);
    TASSIGN(tile, 0x0);
    pto::comm::TPUT(dstG, srcG, tile);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunCombinePushTask(__gm__ mc2::v4::protocol::RemoteWindowContext* ctx,
                                               __gm__ mc2::v4::CombineOnlyParams* params,
                                               const __gm__ mc2::v4::protocol::CombinePushTask* task)
{
    auto* src = reinterpret_cast<__gm__ float*>(params->localPartial + task->srcPayloadOffsetBytes);
    auto* dst = reinterpret_cast<__gm__ float*>(ctx->windowIn[task->dstRank] +
                                                ctx->combineRegionOffset +
                                                task->dstPayloadOffsetBytes);
    RunCombinePutSlot(dst, src);
}

V4_FORCE_INLINE_AICORE void RunCombineOnlyKernel(__gm__ mc2::v4::protocol::RemoteWindowContext* ctx,
                                                 __gm__ mc2::v4::CombineOnlyParams* params,
                                                 const __gm__ mc2::v4::StandaloneKernelTilingData* tiling)
{
    if (tiling->mode != static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly)) {
        return;
    }
    if (params->phase == 1) {
        const uint32_t taskIdx = AscendC::GetBlockIdx();
        if (taskIdx >= params->taskCount) {
            return;
        }
        auto* tasks = reinterpret_cast<__gm__ mc2::v4::protocol::CombinePushTask*>(params->combineTasks);
        RunCombinePushTask(ctx, params, tasks + taskIdx);
        return;
    }
    if (AscendC::GetBlockIdx() == 0) {
        mc2::v4::output::RunCombineRestorePhase(ctx, params);
    }
}

}  // namespace mc2::v4::combine
#endif
