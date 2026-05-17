#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "../dispatch_ffn_combine_tiling.h"
#include "gmm1.hpp"
#include "gmm2.hpp"
#include "swiglu.hpp"

namespace mc2::v4::compute {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

V4_FORCE_INLINE_AICORE void RunComputeOnlyKernel(__gm__ mc2::v4::ComputeOnlyParams* params,
                                                 const __gm__ mc2::v4::StandaloneKernelTilingData* tiling)
{
    if (AscendC::GetBlockIdx() != 0 || tiling->mode != static_cast<uint32_t>(mc2::v4::KernelMode::ComputeOnly)) {
        return;
    }
    auto* input = reinterpret_cast<__gm__ float*>(params->input);
    auto* weight1 = reinterpret_cast<__gm__ float*>(params->weight1);
    auto* gmm1Out = reinterpret_cast<__gm__ float*>(params->gmm1Out);
    auto* swigluOut = reinterpret_cast<__gm__ float*>(params->swigluOut);
    auto* weight2 = reinterpret_cast<__gm__ float*>(params->weight2);
    auto* gmm2Out = reinterpret_cast<__gm__ float*>(params->gmm2Out);
    RunGmm1Group(input, weight1, gmm1Out);
    RunSwiGluGroup(gmm1Out, swigluOut);
    RunGmm2Group(swigluOut, weight2, gmm2Out);
}

}  // namespace mc2::v4::compute
#endif
