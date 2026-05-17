#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "pto/comm/pto_comm_inst.hpp"
#include <pto/pto-inst.hpp>

#include "combine_progress.hpp"
#include "../dispatch_ffn_combine_tiling.h"
#include "../output/unpermute_reduce.hpp"
#include "../protocol/remote_window.hpp"

namespace mc2::v4::combine {

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

using CombineShape = pto::Shape<1, 1, 1, 1, 8>;
using CombineStride = pto::Stride<8, 8, 8, 8, 1>;
using CombineGlobal = pto::GlobalTensor<float, CombineShape, CombineStride, pto::Layout::ND>;
using CombineTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;

V4_FORCE_INLINE_AICORE void RunCombinePut8(__gm__ float* dstPtr, __gm__ float* srcPtr)
{
    CombineGlobal dstG(dstPtr);
    CombineGlobal srcG(srcPtr);
    CombineTile tile(1, 8);
    TASSIGN(tile, 0x0);
    pto::comm::TPUT(dstG, srcG, tile);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunCombinePushPhase(__gm__ mc2::v4::protocol::RemoteWindowContext* ctx,
                                                __gm__ mc2::v4::CombineOnlyParams* params)
{
    auto* localPartial = reinterpret_cast<__gm__ float*>(params->localPartial);
    auto* localCombine = reinterpret_cast<__gm__ float*>(ctx->workspaceBase + ctx->combineRegionOffset);
    if (ctx->rank == 0) {
        for (uint32_t i = 0; i < 8; ++i) {
            localCombine[i] = localPartial[i];
        }
        dsb(DSB_DDR);
        return;
    }
    if (ctx->rank == 1) {
        auto* remoteCombine = reinterpret_cast<__gm__ float*>(ctx->windowIn[0] + ctx->combineRegionOffset +
                                                               static_cast<uint64_t>(8 * sizeof(float)));
        RunCombinePut8(remoteCombine, localPartial);
    }
}

V4_FORCE_INLINE_AICORE void RunCombineOnlyKernel(__gm__ mc2::v4::protocol::RemoteWindowContext* ctx,
                                                 __gm__ mc2::v4::CombineOnlyParams* params,
                                                 const __gm__ mc2::v4::StandaloneKernelTilingData* tiling)
{
    if (AscendC::GetBlockIdx() != 0 || tiling->mode != static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly)) {
        return;
    }
    if (params->phase == 1) {
        RunCombinePushPhase(ctx, params);
    } else {
        mc2::v4::output::RunCombineRestorePhase(ctx, params);
    }
}

}  // namespace mc2::v4::combine
#endif
