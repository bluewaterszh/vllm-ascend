#pragma once

#include <cstdint>

static constexpr uint32_t PTO_HCCL_MAX_RANKS = 64;

struct HcclDeviceContext {
    uint64_t workSpace = 0;
    uint64_t workSpaceSize = 0;
    uint32_t rankId = 0;
    uint32_t rankNum = 0;
    uint64_t winSize = 0;
    uint64_t windowsIn[PTO_HCCL_MAX_RANKS] = {};
    uint64_t windowsOut[PTO_HCCL_MAX_RANKS] = {};
};

