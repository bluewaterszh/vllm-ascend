#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

namespace mc2::v4::compute {

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

V4_FORCE_INLINE_AICORE void RunSwiGluGroup(__gm__ float* gmm1Out, __gm__ float* swigluOut)
{
    for (uint32_t i = 0; i < 2; ++i) {
        const float gate = gmm1Out[i];
        const float up = gmm1Out[i + 2];
        swigluOut[i] = gate * 0.5f * up;
    }
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::compute
#endif
