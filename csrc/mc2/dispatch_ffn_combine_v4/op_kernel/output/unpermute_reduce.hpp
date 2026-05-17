#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "../dispatch_ffn_combine_tiling.h"
#include "../protocol/remote_window.hpp"

namespace mc2::v4::output {

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

V4_FORCE_INLINE_AICORE void RunCombineRestorePhase(__gm__ mc2::v4::protocol::RemoteWindowContext* ctx,
                                                   __gm__ mc2::v4::CombineOnlyParams* params)
{
    auto* localCombine = reinterpret_cast<__gm__ float*>(ctx->workspaceBase + ctx->combineRegionOffset);
    auto* restoredRows = reinterpret_cast<__gm__ float*>(params->restoredRows);
    if (ctx->rank == 0) {
        restoredRows[0] = localCombine[0] * 0.25f + localCombine[8] * 0.75f;
    } else {
        restoredRows[0] = 0.0f;
    }
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::output
#endif
