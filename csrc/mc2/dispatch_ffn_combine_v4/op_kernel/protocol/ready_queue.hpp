#pragma once

#include <cstdint>

namespace mc2::v4::protocol {

struct ReadyQueueEntry {
    uint32_t phase = 0;
    uint32_t epoch = 0;
    uint32_t taskCount = 0;
    uint32_t reserved = 0;
};

struct ReadyQueueSummary {
    uint32_t dispatchReady = 0;
    uint32_t computeReady = 0;
    uint32_t combineReady = 0;
    uint32_t completed = 0;
};

}  // namespace mc2::v4::protocol
