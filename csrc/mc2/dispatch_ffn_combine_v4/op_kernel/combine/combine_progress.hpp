#pragma once

#include <cstdint>

namespace mc2::v4::combine {

struct CombineGroupProgress {
    uint32_t groupId = 0;
    uint32_t tileId = 0;
    uint32_t phase = 0;
    uint32_t completedTasks = 0;
    uint32_t publishedEpoch = 0;
};

}  // namespace mc2::v4::combine
