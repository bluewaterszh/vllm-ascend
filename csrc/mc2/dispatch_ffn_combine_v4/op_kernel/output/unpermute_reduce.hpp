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
    auto* expandedProb = reinterpret_cast<__gm__ float*>(params->expandedProb);
    auto* rowRanges = reinterpret_cast<__gm__ mc2::v4::protocol::CombineRowRange*>(params->rowRanges);
    for (uint32_t row = 0; row < params->localRowCount; ++row) {
        float acc = 0.0f;
        const uint32_t begin = rowRanges[row].begin;
        const uint32_t end = rowRanges[row].end;
        for (uint32_t idx = begin; idx < end; ++idx) {
            acc += localCombine[idx * mc2::v4::protocol::kCombineTransportElems] * expandedProb[idx];
        }
        restoredRows[row] = acc;
    }
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::output
#endif
