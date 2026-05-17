#pragma once

#if defined(__CCE_AICORE__)
#include "../substrate/pto_mmad_shell.hpp"

namespace mc2::v4::compute {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

V4_FORCE_INLINE_AICORE void RunGmm1Group(const float* input, __gm__ float* weight1, __gm__ float* gmm1Out)
{
    for (uint32_t n = 0; n < 4; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 2; ++k) {
            acc += input[k] * weight1[k * 4 + n];
        }
        gmm1Out[n] = acc;
    }
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::compute
#endif
