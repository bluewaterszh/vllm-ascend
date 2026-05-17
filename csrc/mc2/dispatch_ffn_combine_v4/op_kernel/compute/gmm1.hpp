#pragma once

#if defined(__CCE_AICORE__)
#include "../substrate/pto_mmad_shell.hpp"

namespace mc2::v4::compute {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

V4_FORCE_INLINE_AICORE void RunGmm1Group(__gm__ float* input, __gm__ float* weight1, __gm__ float* gmm1Out)
{
    substrate::RunDeviceMmadBlock<1, 2, 4>(gmm1Out, input, weight1);
}

}  // namespace mc2::v4::compute
#endif
