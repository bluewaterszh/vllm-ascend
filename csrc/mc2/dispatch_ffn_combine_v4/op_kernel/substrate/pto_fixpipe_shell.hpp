#pragma once

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

namespace mc2::v4::substrate {

#define V4_FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__

template <typename Global, typename TileData>
V4_FORCE_INLINE_AICORE void RunDeviceStoreBlock(Global& dst, TileData& src)
{
    TSTORE(dst, src);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

}  // namespace mc2::v4::substrate
#endif
