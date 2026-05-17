#pragma once

#include <cstdint>

namespace mc2::v4::dispatch {

struct DispatchGroupProgress {
    uint32_t expectedTasks = 0;
    uint32_t completedTasks = 0;
    uint32_t readyEpochBase = 0;
    uint32_t doneEpochBase = 0;
};

}  // namespace mc2::v4::dispatch
