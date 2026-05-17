#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace mc2::v4::protocol {

struct ReadyQueueEntry {
    uint32_t groupId = 0;
    uint32_t readyEpoch = 0;
    uint32_t expectedSourceCount = 0;
    uint32_t completedSourceCount = 0;
};

struct ReadyQueueSummary {
    uint32_t dispatchReady = 0;
    uint32_t computeReady = 0;
    uint32_t combineReady = 0;
    uint32_t completed = 0;
};

struct DispatchSummaryView {
    uint32_t minReadyEpoch = 0;
    uint32_t visibleSourceCount = 0;
};

struct ExpertGroupSchedule {
    std::vector<uint32_t> widths;

    uint32_t WidthForStep(uint32_t step) const
    {
        return step < widths.size() ? widths[step] : 1;
    }
};

struct ReadyQueue {
    std::vector<ReadyQueueEntry> dispatchEntries;
    std::vector<ReadyQueueEntry> computeEntries;
    std::vector<ReadyQueueEntry> combineEntries;

    void PushDispatchReady(uint32_t groupId, uint32_t readyEpoch, uint32_t expectedSourceCount)
    {
        dispatchEntries.push_back({groupId, readyEpoch, expectedSourceCount, 0});
    }

    void PushComputeReady(uint32_t groupId, uint32_t readyEpoch)
    {
        computeEntries.push_back({groupId, readyEpoch, 0, 0});
    }

    void PushCombineReady(uint32_t groupId, uint32_t readyEpoch)
    {
        combineEntries.push_back({groupId, readyEpoch, 0, 0});
    }
};

inline ExpertGroupSchedule BuildExpertGroupSchedule(uint32_t experts)
{
    ExpertGroupSchedule schedule;
    if (experts == 0) {
        return schedule;
    }
    const uint32_t seed[] = {8, 4, 2, 1, 1};
    uint32_t remaining = experts;
    for (uint32_t width : seed) {
        if (remaining == 0) {
            break;
        }
        const uint32_t clamped = width < remaining ? width : remaining;
        schedule.widths.push_back(clamped);
        remaining -= clamped;
    }
    while (remaining > 0) {
        schedule.widths.push_back(1);
        --remaining;
    }
    return schedule;
}

}  // namespace mc2::v4::protocol
