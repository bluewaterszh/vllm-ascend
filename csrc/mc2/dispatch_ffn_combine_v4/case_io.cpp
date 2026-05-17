#include "case_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "op_kernel/routing/route_plan.hpp"

namespace mc2::v4 {
namespace {

std::vector<uint8_t> MakeDispatchRowPayload(const RoutingFixture& fixture,
                                            uint32_t srcRank,
                                            uint32_t rowIdx)
{
    std::vector<uint8_t> row(fixture.hiddenBytes, 0);
    const uint8_t value = static_cast<uint8_t>(srcRank * 16 + rowIdx + 1);
    std::fill(row.begin(), row.end(), value);
    return row;
}

ComputeFixture BuildComputeFixture(uint32_t localRank)
{
    ComputeFixture fixture;
    fixture.input = {1.0f, 2.0f};
    fixture.quantInput = {2, 4};
    fixture.inputScale = {0.5f};
    fixture.weight1 = {
        localRank == 0 ? 2.0f : 4.0f, 0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    fixture.weight2 = {
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    return fixture;
}

float ProducerScalarForRank(uint32_t rank)
{
    return BuildHostComputeOracle(rank).gmm2Out.empty() ? 0.0f : BuildHostComputeOracle(rank).gmm2Out[0];
}

}  // namespace

RoutingFixture BuildTwoRankRoutingFixtureForRank1()
{
    RoutingFixture fixture;
    fixture.worldSize = 2;
    fixture.expertsPerRank = 2;
    fixture.topk = 2;
    fixture.maxOutputSize = 8;
    fixture.hiddenBytes = 256;
    fixture.outputBytes = sizeof(float);
    fixture.rowsPerRank = {2, 2};
    fixture.xActiveMaskPerRank = {
        {1, 1},
        {1, 1},
    };
    fixture.topkIdsPerRank = {
        {2, 0,
         3, 1},
        {2, 3,
         0, 3},
    };
    fixture.topkProbsPerRank = {
        {0.6f, 0.4f,
         0.7f, 0.3f},
        {0.8f, 0.2f,
         0.55f, 0.45f},
    };
    return fixture;
}

RoutingFixture BuildTwoRankMaskAndCapacityFixtureForRank1()
{
    RoutingFixture fixture;
    fixture.worldSize = 2;
    fixture.expertsPerRank = 2;
    fixture.topk = 2;
    fixture.maxOutputSize = 3;
    fixture.hiddenBytes = 256;
    fixture.outputBytes = sizeof(float);
    fixture.rowsPerRank = {2, 2};
    fixture.xActiveMaskPerRank = {
        {1, 1},
        {0, 1},
    };
    fixture.topkIdsPerRank = {
        {2, 3,
         2, 0},
        {2, 3,
         0, 3},
    };
    fixture.topkProbsPerRank = {
        {0.7f, 0.3f,
         0.65f, 0.35f},
        {0.9f, 0.1f,
         0.6f, 0.4f},
    };
    return fixture;
}

std::vector<uint8_t> BuildDispatchPublicationForRank(const RoutingFixture& fixture, uint32_t srcRank)
{
    const auto raw = routing::ExpandBeforeCapacity(fixture);
    const auto executable = routing::KeepExecutableEntries(raw, fixture);
    const uint64_t segmentBytes = static_cast<uint64_t>(fixture.maxOutputSize) * fixture.hiddenBytes;
    std::vector<uint8_t> publication(static_cast<size_t>(fixture.worldSize) * segmentBytes, 0);
    std::vector<uint32_t> ordinalPerDst(fixture.worldSize, 0);

    for (const auto& entry : executable) {
        if (entry.srcRank != srcRank) {
            continue;
        }
        const uint64_t offset = static_cast<uint64_t>(entry.dstRank) * segmentBytes +
                                static_cast<uint64_t>(ordinalPerDst[entry.dstRank]) * fixture.hiddenBytes;
        ++ordinalPerDst[entry.dstRank];
        const auto row = MakeDispatchRowPayload(fixture, srcRank, entry.rowIdx);
        std::memcpy(publication.data() + offset, row.data(), row.size());
    }
    return publication;
}

std::vector<uint8_t> BuildHostDispatchOracle(const RoutingFixture& fixture,
                                             uint32_t localRank,
                                             const routing::RoutingPlanBundle& plan)
{
    std::vector<std::vector<uint8_t>> publications;
    publications.reserve(fixture.worldSize);
    for (uint32_t srcRank = 0; srcRank < fixture.worldSize; ++srcRank) {
        publications.push_back(BuildDispatchPublicationForRank(fixture, srcRank));
    }

    uint32_t totalRows = 0;
    for (uint32_t value : plan.view.dispatch.gatheredExpertCount) {
        totalRows += value;
    }
    std::vector<uint8_t> oracle(static_cast<size_t>(totalRows) * fixture.hiddenBytes, 0);
    for (const auto& task : plan.dispatchTasks) {
        const auto& publication = publications.at(task.srcRank);
        std::memcpy(oracle.data() + task.dstPayloadOffsetBytes,
                    publication.data() + task.srcPayloadOffsetBytes,
                    fixture.hiddenBytes);
    }
    return oracle;
}

HostComputeOracle BuildHostComputeOracle(uint32_t localRank)
{
    HostComputeOracle oracle;
    oracle.fixture = BuildComputeFixture(localRank);
    oracle.sideband = {
        .payloadOffsetBytes = 0,
        .scale1OffsetBytes = static_cast<uint64_t>(oracle.fixture.quantInput.size() * sizeof(int8_t)),
        .scale2OffsetBytes = static_cast<uint64_t>(oracle.fixture.quantInput.size() * sizeof(int8_t) +
                                                   oracle.fixture.inputScale.size() * sizeof(float)),
        .slotCount = 2,
    };
    oracle.quantPayload = oracle.fixture.quantInput;
    oracle.scale1 = oracle.fixture.inputScale;
    oracle.scale2 = {1.0f};

    std::vector<float> dequantInput(oracle.fixture.quantInput.size(), 0.0f);
    for (size_t i = 0; i < oracle.fixture.quantInput.size(); ++i) {
        dequantInput[i] = static_cast<float>(oracle.fixture.quantInput[i]) * oracle.fixture.inputScale[0];
    }

    oracle.gmm1Out.assign(4, 0.0f);
    for (uint32_t n = 0; n < 4; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 2; ++k) {
            acc += dequantInput[k] * oracle.fixture.weight1[k * 4 + n];
        }
        oracle.gmm1Out[n] = acc;
    }

    oracle.swigluOut.assign(2, 0.0f);
    for (uint32_t i = 0; i < 2; ++i) {
        const float gate = oracle.gmm1Out[i];
        const float up = oracle.gmm1Out[i + 2];
        oracle.swigluOut[i] = gate * 0.5f * up;
    }

    oracle.gmm2Out.assign(2, 0.0f);
    for (uint32_t n = 0; n < 2; ++n) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < 2; ++k) {
            acc += oracle.swigluOut[k] * oracle.fixture.weight2[k * 2 + n];
        }
        oracle.gmm2Out[n] = acc;
    }
    return oracle;
}

HostCombineOracle BuildHostCombineOracle(uint32_t localRank)
{
    HostCombineOracle oracle;
    const auto fixture = BuildTwoRankRoutingFixtureForRank1();
    const auto raw = routing::ExpandBeforeCapacity(fixture);
    const auto executable = routing::KeepExecutableEntries(raw, fixture);
    const auto plan = routing::BuildRoutePlanForRank(fixture, localRank);
    const float localScalar = ProducerScalarForRank(localRank);
    oracle.pushedPayload.assign(plan.combineTasks.size(), localScalar);
    oracle.restoredRows.assign(fixture.rowsPerRank.at(localRank), 0.0f);
    for (const auto& entry : executable) {
        if (entry.ownerRank != localRank) {
            continue;
        }
        oracle.restoredRows.at(entry.rowIdx) += entry.prob * ProducerScalarForRank(entry.dstRank);
    }
    return oracle;
}

HostFullChainOracle BuildHostFullChainOracle(uint32_t localRank)
{
    HostFullChainOracle oracle;
    const auto fixture = BuildTwoRankRoutingFixtureForRank1();
    const auto plan = routing::BuildRoutePlanForRank(fixture, localRank);
    oracle.dispatchBytes = BuildHostDispatchOracle(fixture, localRank, plan);
    oracle.compute = BuildHostComputeOracle(localRank);
    oracle.output = BuildHostCombineOracle(localRank).restoredRows;
    return oracle;
}

OverlapTimeline BuildSteadyStateTimeline(uint32_t groupCount)
{
    OverlapTimeline timeline;
    if (groupCount == 0) {
        return timeline;
    }
    timeline.resize(groupCount + 2);
    for (uint32_t tick = 0; tick < timeline.size(); ++tick) {
        if (tick < groupCount) {
            timeline[tick].push_back({OverlapStage::Dispatch, tick});
        }
        if (tick >= 1 && (tick - 1) < groupCount) {
            timeline[tick].push_back({OverlapStage::Compute, tick - 1});
        }
        if (tick >= 2 && (tick - 2) < groupCount) {
            timeline[tick].push_back({OverlapStage::Combine, tick - 2});
        }
    }
    return timeline;
}

bool CompareBytes(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs)
{
    return lhs == rhs;
}

bool CompareFloats(const std::vector<float>& lhs, const std::vector<float>& rhs, float atol)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > atol) {
            return false;
        }
    }
    return true;
}

}  // namespace mc2::v4
