#pragma once

#if defined(__CCE_AICORE__)
#include "../substrate/pto_mmad_shell.hpp"

namespace mc2::v4::compute {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

V4_FORCE_INLINE_AICORE void RunGmm2Group(__gm__ float* swigluOut, __gm__ float* weight2, __gm__ float* gmm2Out)
{
    substrate::RunDeviceMmadBlock<1, 2, 2>(gmm2Out, swigluOut, weight2);
}

}  // namespace mc2::v4::compute
#endif
