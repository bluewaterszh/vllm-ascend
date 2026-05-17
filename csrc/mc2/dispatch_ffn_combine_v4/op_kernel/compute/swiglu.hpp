#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

namespace mc2::v4::compute {

#ifndef V4_FORCE_INLINE_AICORE
#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif

V4_FORCE_INLINE_AICORE void RunSwiGluGroupRange(__gm__ float* gmm1Out,
                                                __gm__ float* swigluOut,
                                                uint32_t begin,
                                                uint32_t width)
{
    const uint32_t end = begin + width;
    for (uint32_t i = begin; i < end; ++i) {
        const float gate = gmm1Out[i];
        const float up = gmm1Out[i + 2];
        swigluOut[i] = gate * 0.5f * up;
    }
    dsb(DSB_DDR);
}

V4_FORCE_INLINE_AICORE void RunSwiGluGroup(__gm__ float* gmm1Out, __gm__ float* swigluOut)
{
    RunSwiGluGroupRange(gmm1Out, swigluOut, 0, 2);
}

}  // namespace mc2::v4::compute
#endif
