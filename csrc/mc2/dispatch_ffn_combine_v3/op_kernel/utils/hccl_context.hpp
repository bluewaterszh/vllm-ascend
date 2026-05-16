#pragma once

#include <cstdint>

static constexpr uint32_t PTO_HCCL_MAX_RANKS = 64;

struct PtoRemoteWindowContext {
    uint64_t workspaceBase = 0;
    uint64_t workspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t windowBytes = 0;
    uint64_t windowIn[PTO_HCCL_MAX_RANKS] = {};
    uint64_t windowOut[PTO_HCCL_MAX_RANKS] = {};
};
