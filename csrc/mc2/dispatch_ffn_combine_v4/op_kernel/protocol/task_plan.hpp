#pragma once

#include <cstdint>

namespace mc2::v4::protocol {

static constexpr uint32_t kCombineTransportElems = 8;
static constexpr uint32_t kCombineTransportBytes = kCombineTransportElems * sizeof(float);

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
    uint64_t srcScaleOffsetBytes = 0;
    uint64_t dstScaleOffsetBytes = 0;
    uint32_t readyEpoch = 0;
    uint32_t taskFlags = 0;
};

struct CombinePushTask {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t expertGroupId = 0;
    uint32_t expertId = 0;
    uint32_t ownerRow = 0;
    uint32_t tileId = 0;
    uint32_t srcRowBegin = 0;
    uint32_t dstRowBegin = 0;
    uint32_t rowCount = 0;
    uint32_t outputBytes = 0;
    uint64_t srcPayloadOffsetBytes = 0;
    uint64_t dstPayloadOffsetBytes = 0;
    uint32_t completionEpoch = 0;
    uint32_t taskFlags = 0;
};

struct CombineRowRange {
    uint32_t begin = 0;
    uint32_t end = 0;
};

struct ExpertGroupTaskRange {
    uint32_t expertGroupId = 0;
    uint32_t dispatchTaskBegin = 0;
    uint32_t dispatchTaskEnd = 0;
    uint32_t combineTaskBegin = 0;
    uint32_t combineTaskEnd = 0;
};

}  // namespace mc2::v4::protocol
