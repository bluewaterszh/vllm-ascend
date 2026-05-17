#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "../../case_io.hpp"
#include "../protocol/task_plan.hpp"
#include "route_count.hpp"
#include "route_cumsum.hpp"
#include "route_expand.hpp"
#include "route_owner.hpp"
#include "route_sort.hpp"

namespace mc2::v4::routing {

struct RawExpandedEntry {
    uint32_t ownerRank = 0;
    uint32_t srcRank = 0;
    uint32_t rowIdx = 0;
    uint32_t topkOrdinal = 0;
    uint32_t expertId = 0;
    uint32_t dstRank = 0;
    uint32_t localExpertSlot = 0;
    float prob = 0.0f;
    bool isSentinel = false;
};

struct DispatchTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<uint32_t> expandedExpertIdx;
    std::vector<float> expandedProb;
    std::vector<uint32_t> expandedSrcRank;
    std::vector<uint32_t> expandedLocalExpertSlot;
    std::vector<uint32_t> tokenPerExpert;
    std::vector<uint32_t> gatheredExpertCount;
    std::vector<uint32_t> localExpertPrefix;
    std::vector<uint32_t> cumsumMM;
    std::vector<uint32_t> groupRowCount;
    uint32_t packedRowCount = 0;
    std::vector<uint32_t> srcOffset;
    std::vector<uint32_t> dstOffset;
};

struct CombineTruthView {
    std::vector<uint32_t> expandedRowIdx;
    std::vector<float> expandedProb;
    std::vector<std::pair<uint32_t, uint32_t>> rowToExpandedRange;
    std::vector<uint32_t> ownerRowExpandedOrdinal;
    std::vector<uint32_t> combineDstOffset;
};

struct RoutingMetadataView {
    DispatchTruthView dispatch;
    CombineTruthView combine;
};

struct RoutingPlanBundle {
    RoutingMetadataView view;
    std::vector<mc2::v4::protocol::DispatchPullTask> dispatchTasks;
    std::vector<mc2::v4::protocol::CombinePushTask> combineTasks;
};

inline uint32_t SentinelExpertId(const mc2::v4::RoutingFixture& fixture)
{
    return fixture.worldSize * fixture.expertsPerRank;
}

inline std::vector<RawExpandedEntry> ExpandBeforeCapacity(const mc2::v4::RoutingFixture& fixture)
{
    std::vector<RawExpandedEntry> out;
    const uint32_t sentinelExpert = SentinelExpertId(fixture);
    for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
        const uint32_t rows = fixture.rowsPerRank.at(srcRank);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t k = 0; k < fixture.topk; ++k) {
                const uint32_t idx = row * fixture.topk + k;
                const bool active = fixture.xActiveMaskPerRank.at(srcRank).at(row) != 0;
                const uint32_t expertId = active ? fixture.topkIdsPerRank.at(srcRank).at(idx) : sentinelExpert;
                out.push_back({
                    .ownerRank = srcRank,
                    .srcRank = srcRank,
                    .rowIdx = row,
                    .topkOrdinal = k,
                    .expertId = expertId,
                    .dstRank = active ? (expertId / fixture.expertsPerRank) : fixture.worldSize,
                    .localExpertSlot = active ? (expertId % fixture.expertsPerRank) : fixture.expertsPerRank,
                    .prob = fixture.topkProbsPerRank.at(srcRank).at(idx),
                    .isSentinel = !active,
                });
            }
        }
    }
    return out;
}

inline std::vector<RawExpandedEntry> KeepExecutableEntries(const std::vector<RawExpandedEntry>& raw,
                                                           const mc2::v4::RoutingFixture& fixture)
{
    std::vector<uint32_t> keptPerDst(fixture.worldSize, 0);
    std::vector<RawExpandedEntry> out;
    out.reserve(raw.size());
    for (const auto& entry : raw) {
        if (entry.isSentinel || entry.dstRank >= fixture.worldSize) {
            continue;
        }
        if (keptPerDst[entry.dstRank] >= fixture.maxOutputSize) {
            continue;
        }
        ++keptPerDst[entry.dstRank];
        out.push_back(entry);
    }
    return out;
}

inline void BuildDispatchTruthForLocalRank(const std::vector<RawExpandedEntry>& executable,
                                           const mc2::v4::RoutingFixture& fixture,
                                           uint32_t localRank,
                                           DispatchTruthView& view)
{
    view = {};
    view.tokenPerExpert.assign(fixture.worldSize * fixture.expertsPerRank, 0);
    view.gatheredExpertCount.assign(fixture.expertsPerRank, 0);
    view.localExpertPrefix.assign(fixture.expertsPerRank, 0);
    view.cumsumMM.assign(fixture.expertsPerRank * fixture.worldSize, 0);
    view.groupRowCount.assign(fixture.expertsPerRank, 0);

    std::vector<RawExpandedEntry> localEntries;
    localEntries.reserve(executable.size());
    for (const auto& entry : executable) {
        if (entry.dstRank != localRank) {
            continue;
        }
        localEntries.push_back(entry);
        view.expandedRowIdx.push_back(entry.rowIdx);
        view.expandedExpertIdx.push_back(entry.expertId);
        view.expandedProb.push_back(entry.prob);
        view.expandedSrcRank.push_back(entry.srcRank);
        view.expandedLocalExpertSlot.push_back(entry.localExpertSlot);
        ++view.tokenPerExpert[entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot];
        ++view.groupRowCount[entry.localExpertSlot];
    }

    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            const uint32_t count = view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
            view.gatheredExpertCount[localExpert] += count;
        }
    }
    view.localExpertPrefix = ExclusiveScan(view.gatheredExpertCount);
    for (uint32_t localExpert = 0; localExpert < fixture.expertsPerRank; ++localExpert) {
        uint32_t prefix = 0;
        for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
            const size_t index = static_cast<size_t>(localExpert) * fixture.worldSize + srcRank;
            view.cumsumMM[index] = prefix;
            prefix += view.tokenPerExpert[srcRank * fixture.expertsPerRank + localExpert];
        }
    }

    std::vector<uint32_t> srcOrdinalPerRank(fixture.worldSize, 0);
    std::vector<uint32_t> dstOrdinalPerSrcLocalExpert(fixture.worldSize * fixture.expertsPerRank, 0);
    for (const auto& entry : localEntries) {
        view.srcOffset.push_back(srcOrdinalPerRank[entry.srcRank]++);
        const uint32_t localKey = entry.srcRank * fixture.expertsPerRank + entry.localExpertSlot;
        const uint32_t localOrdinal = dstOrdinalPerSrcLocalExpert[localKey]++;
        view.dstOffset.push_back(
            view.localExpertPrefix[entry.localExpertSlot] +
            view.cumsumMM[entry.localExpertSlot * fixture.worldSize + entry.srcRank] +
            localOrdinal);
    }
    view.packedRowCount = static_cast<uint32_t>(localEntries.size());
}

inline void BuildCombineTruthForLocalOwner(const std::vector<RawExpandedEntry>& executable,
                                           const mc2::v4::RoutingFixture& fixture,
                                           uint32_t localRank,
                                           CombineTruthView& view)
{
    view = {};
    const uint32_t localRows = fixture.rowsPerRank.at(localRank);
    std::vector<std::vector<RawExpandedEntry>> rowBuckets(localRows);
    for (const auto& entry : executable) {
        if (entry.ownerRank != localRank) {
            continue;
        }
        rowBuckets.at(entry.rowIdx).push_back(entry);
    }

    uint32_t running = 0;
    for (uint32_t row = 0; row < localRows; ++row) {
        const uint32_t begin = running;
        for (const auto& entry : rowBuckets[row]) {
            view.expandedRowIdx.push_back(entry.rowIdx);
            view.expandedProb.push_back(entry.prob);
            view.ownerRowExpandedOrdinal.push_back(running);
            view.combineDstOffset.push_back(running);
            ++running;
        }
        view.rowToExpandedRange.push_back({begin, running});
    }
}

inline std::vector<uint32_t> BuildOwnerCombineDstOffsetForExecutable(const std::vector<RawExpandedEntry>& executable,
                                                                  const mc2::v4::RoutingFixture& fixture)
{
    std::vector<std::vector<uint32_t>> rowCounts(fixture.worldSize);
    std::vector<std::vector<uint32_t>> rowBases(fixture.worldSize);
    std::vector<std::vector<uint32_t>> rowOrdinals(fixture.worldSize);
    for (uint32_t ownerRank = 0; ownerRank < fixture.worldSize; ++ownerRank) {
        const uint32_t rows = fixture.rowsPerRank.at(ownerRank);
        rowCounts[ownerRank].assign(rows, 0);
        rowBases[ownerRank].assign(rows, 0);
        rowOrdinals[ownerRank].assign(rows, 0);
    }

    for (const auto& entry : executable) {
        ++rowCounts.at(entry.ownerRank).at(entry.rowIdx);
    }
    for (uint32_t ownerRank = 0; ownerRank < fixture.worldSize; ++ownerRank) {
        uint32_t running = 0;
        for (uint32_t row = 0; row < rowCounts[ownerRank].size(); ++row) {
            rowBases[ownerRank][row] = running;
            running += rowCounts[ownerRank][row];
        }
    }

    std::vector<uint32_t> dstOffsets;
    dstOffsets.reserve(executable.size());
    for (const auto& entry : executable) {
        auto& ownerRowOrdinal = rowOrdinals.at(entry.ownerRank).at(entry.rowIdx);
        dstOffsets.push_back(rowBases.at(entry.ownerRank).at(entry.rowIdx) + ownerRowOrdinal);
        ++ownerRowOrdinal;
    }
    return dstOffsets;
}

inline RoutingPlanBundle BuildRoutePlanForRank(const mc2::v4::RoutingFixture& fixture,
                                               uint32_t localRank)
{
    RoutingPlanBundle bundle;
    const auto raw = ExpandBeforeCapacity(fixture);
    const auto executable = KeepExecutableEntries(raw, fixture);
    BuildDispatchTruthForLocalRank(executable, fixture, localRank, bundle.view.dispatch);
    BuildCombineTruthForLocalOwner(executable, fixture, localRank, bundle.view.combine);
    const auto ownerCombineDstOffset = BuildOwnerCombineDstOffsetForExecutable(executable, fixture);

    const uint64_t dispatchSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    const uint64_t dispatchScaleSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * sizeof(float);
    for (size_t i = 0; i < bundle.view.dispatch.srcOffset.size(); ++i) {
        bundle.dispatchTasks.push_back({
            .srcRank = bundle.view.dispatch.expandedSrcRank[i],
            .dstRank = localRank,
            .expertGroupId = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .expertId = bundle.view.dispatch.expandedExpertIdx[i],
            .localExpertSlot = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .srcRowBegin = bundle.view.dispatch.srcOffset[i],
            .dstRowBegin = bundle.view.dispatch.dstOffset[i],
            .rowCount = 1,
            .hiddenBytes = fixture.hiddenBytes,
            .scaleBytes = sizeof(float),
            .srcPayloadOffsetBytes = dispatchSegmentBytes * localRank +
                                     static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * fixture.hiddenBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * fixture.hiddenBytes,
            .srcScaleOffsetBytes = dispatchScaleSegmentBytes * localRank +
                                   static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * sizeof(float),
            .dstScaleOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * sizeof(float),
            .readyEpoch = 1,
            .taskFlags = 0,
        });
    }

    uint32_t tileId = 0;
    uint32_t sourceOrdinal = 0;
    for (size_t entryIdx = 0; entryIdx < executable.size(); ++entryIdx) {
        const auto& entry = executable[entryIdx];
        if (entry.dstRank != localRank) {
            continue;
        }
        bundle.combineTasks.push_back({
            .srcRank = localRank,
            .dstRank = entry.ownerRank,
            .expertGroupId = entry.localExpertSlot,
            .expertId = entry.expertId,
            .ownerRow = entry.rowIdx,
            .tileId = tileId++,
            .srcRowBegin = sourceOrdinal,
            .dstRowBegin = ownerCombineDstOffset[entryIdx],
            .rowCount = 1,
            .outputBytes = mc2::v4::protocol::kCombineTransportBytes,
            .srcPayloadOffsetBytes = static_cast<uint64_t>(sourceOrdinal) * mc2::v4::protocol::kCombineTransportBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(ownerCombineDstOffset[entryIdx]) * mc2::v4::protocol::kCombineTransportBytes,
            .completionEpoch = 1,
            .taskFlags = 0,
        });
        ++sourceOrdinal;
    }

    return bundle;
}

}  // namespace mc2::v4::routing
