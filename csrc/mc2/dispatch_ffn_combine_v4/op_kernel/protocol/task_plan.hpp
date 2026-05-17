#pragma once

#include <cstdint>

namespace mc2::v4::protocol {

struct DispatchPullTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t localExpertSlot = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t scaleBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t readyEpoch = 0;
    uint32_t taskFlags = 0;
};

struct CombinePushTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t outputBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t completionEpoch = 0;
    uint32_t taskFlags = 0;
};

struct ExpertGroupSchedule {
    uint32_t expertGroupId = 0;
    uint32_t dispatchTaskBegin = 0;
    uint32_t dispatchTaskEnd = 0;
    uint32_t combineTaskBegin = 0;
    uint32_t combineTaskEnd = 0;
};

}  // namespace mc2::v4::protocol
