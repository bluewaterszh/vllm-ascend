#pragma once

#include "../../case_io.hpp"
#include "../protocol/task_plan.hpp"
#include "route_count.hpp"
#include "route_cumsum.hpp"
#include "route_expand.hpp"
#include "route_owner.hpp"
#include "route_sort.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

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
    std::vector<uint32_t> tokenPerExpert;      // flattened [srcRank][localExpert]
    std::vector<uint32_t> gatheredExpertCount; // [localExpert]
    std::vector<uint32_t> localExpertPrefix;   // [localExpert]
    std::vector<uint32_t> cumsumMM;            // flattened [localExpert][srcRank]
    std::vector<uint32_t> groupRowCount;       // [srcRank]
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

inline uint32_t SentinelExpertId(const mc2::v4::RoutingFixture& fixture) {
    return fixture.worldSize * fixture.expertsPerRank;
}

inline std::vector<RawExpandedEntry> ExpandBeforeCapacity(const mc2::v4::RoutingFixture& fixture) {
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
                                                           const mc2::v4::RoutingFixture& fixture) {
    std::vector<uint32_t> keptPerDst(fixture.worldSize, 0);
    std::vector<RawExpandedEntry> out;
    out.reserve(raw.size());
    for (const auto& entry : raw) {
        if (entry.isSentinel) {
            continue;
        }
        if (entry.dstRank >= fixture.worldSize) {
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
                                           DispatchTruthView& view) {
    view = {};
    view.tokenPerExpert.assign(fixture.worldSize * fixture.expertsPerRank, 0);
    view.gatheredExpertCount.assign(fixture.expertsPerRank, 0);
    view.localExpertPrefix.assign(fixture.expertsPerRank, 0);
    view.cumsumMM.assign(fixture.expertsPerRank * fixture.worldSize, 0);
    view.groupRowCount.assign(fixture.worldSize, 0);

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
        ++view.groupRowCount[entry.srcRank];
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
                                           CombineTruthView& view) {
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

inline RoutingPlanBundle BuildRoutePlanForRank(const mc2::v4::RoutingFixture& fixture,
                                               uint32_t localRank) {
    RoutingPlanBundle bundle;
    const auto raw = ExpandBeforeCapacity(fixture);
    const auto executable = KeepExecutableEntries(raw, fixture);
    BuildDispatchTruthForLocalRank(executable, fixture, localRank, bundle.view.dispatch);
    BuildCombineTruthForLocalOwner(executable, fixture, localRank, bundle.view.combine);

    const uint64_t dispatchSegmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    for (size_t i = 0; i < bundle.view.dispatch.srcOffset.size(); ++i) {
        bundle.dispatchTasks.push_back({
            .srcRank = bundle.view.dispatch.expandedSrcRank[i],
            .dstRank = localRank,
            .expertGroupId = 0,
            .expertId = bundle.view.dispatch.expandedExpertIdx[i],
            .localExpertSlot = bundle.view.dispatch.expandedLocalExpertSlot[i],
            .srcRowBegin = bundle.view.dispatch.srcOffset[i],
            .dstRowBegin = bundle.view.dispatch.dstOffset[i],
            .rowCount = 1,
            .hiddenBytes = fixture.hiddenBytes,
            .scaleBytes = 0,
            .srcPayloadOffsetBytes = dispatchSegmentBytes * localRank +
                                     static_cast<uint64_t>(bundle.view.dispatch.srcOffset[i]) * fixture.hiddenBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.dispatch.dstOffset[i]) * fixture.hiddenBytes,
            .readyEpoch = 1,
            .taskFlags = 0,
        });
    }

    size_t combineOrdinal = 0;
    for (const auto& entry : executable) {
        if (entry.ownerRank != localRank) {
            continue;
        }
        bundle.combineTasks.push_back({
            .srcRank = entry.dstRank,
            .dstRank = localRank,
            .expertGroupId = 0,
            .expertId = entry.expertId,
            .srcRowBegin = entry.rowIdx,
            .dstRowBegin = bundle.view.combine.combineDstOffset[combineOrdinal],
            .rowCount = 1,
            .outputBytes = fixture.outputBytes,
            .srcPayloadOffsetBytes = static_cast<uint64_t>(entry.rowIdx) * fixture.outputBytes,
            .dstPayloadOffsetBytes = static_cast<uint64_t>(bundle.view.combine.combineDstOffset[combineOrdinal]) * fixture.outputBytes,
            .completionEpoch = 1,
            .taskFlags = 0,
        });
        ++combineOrdinal;
    }

    return bundle;
}

}  // namespace mc2::v4::routing
