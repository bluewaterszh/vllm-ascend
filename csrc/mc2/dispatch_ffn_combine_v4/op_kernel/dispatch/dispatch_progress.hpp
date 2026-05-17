#pragma once

#include <cstdint>

#include "../protocol/ready_queue.hpp"

namespace mc2::v4::dispatch {

struct DispatchGroupProgress {
    uint32_t groupId = 0;
    uint32_t expectedTasks = 0;
    uint32_t completedTasks = 0;
    uint32_t expectedSourceCount = 0;
    uint32_t completedSourceCount = 0;
    uint32_t readyEpochBase = 0;
    uint32_t doneEpochBase = 0;
    uint32_t publishedEpoch = 0;
    uint32_t summaryCount = 0;
};

inline mc2::v4::protocol::DispatchSummaryView BuildDispatchSummaryView(const DispatchGroupProgress& progress)
{
    return {
        .minReadyEpoch = progress.publishedEpoch,
        .visibleSourceCount = progress.completedSourceCount,
    };
}

inline bool CanLaunchCompute(const DispatchGroupProgress& progress,
                             const mc2::v4::protocol::DispatchSummaryView& summary)
{
    return progress.expectedSourceCount != 0 &&
           summary.visibleSourceCount >= progress.expectedSourceCount &&
           summary.minReadyEpoch >= progress.publishedEpoch;
}

}  // namespace mc2::v4::dispatch
