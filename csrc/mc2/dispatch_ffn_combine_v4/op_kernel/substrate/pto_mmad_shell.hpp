#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

namespace mc2::v4::substrate {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <uint32_t M, uint32_t K, uint32_t N>
V4_FORCE_INLINE_AICORE void RunDeviceMmadBlock(__gm__ float* dst, __gm__ float* lhs, __gm__ float* rhs)
{
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += lhs[m * K + k] * rhs[k * N + n];
            }
            dst[m * N + n] = acc;
        }
    }
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::substrate
#endif
