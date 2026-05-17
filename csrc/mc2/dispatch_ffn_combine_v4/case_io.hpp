#pragma once

#include <cstdint>
#include <vector>

namespace mc2::v4 {

namespace routing {
struct RoutingPlanBundle;
}

struct RoutingFixture {
    uint32_t worldSize = 0;
    uint32_t expertsPerRank = 0;
    uint32_t topk = 0;
    uint32_t maxOutputSize = 0;
    uint32_t hiddenBytes = 256;
    uint32_t outputBytes = sizeof(float);
    std::vector<uint32_t> rowsPerRank;
    std::vector<std::vector<uint8_t>> xActiveMaskPerRank;
    std::vector<std::vector<uint32_t>> topkIdsPerRank;
    std::vector<std::vector<float>> topkProbsPerRank;
};

struct ComputeFixture {
    std::vector<float> input;
    std::vector<int8_t> quantInput;
    std::vector<float> inputScale;
    std::vector<float> weight1;
    std::vector<float> weight2;
};

struct QuantSidebandView {
    uint64_t payloadOffsetBytes = 0;
    uint64_t scale1OffsetBytes = 0;
    uint64_t scale2OffsetBytes = 0;
    uint32_t slotCount = 2;
};

struct HostComputeOracle {
    ComputeFixture fixture;
    QuantSidebandView sideband;
    std::vector<int8_t> quantPayload;
    std::vector<float> scale1;
    std::vector<float> scale2;
    std::vector<float> gmm1Out;
    std::vector<float> swigluOut;
    std::vector<float> gmm2Out;
};

struct HostCombineOracle {
    std::vector<float> pushedPayload;
    std::vector<float> restoredRows;
};

struct HostFullChainOracle {
    std::vector<uint8_t> dispatchBytes;
    HostComputeOracle compute;
    std::vector<float> output;
};

enum class OverlapStage : uint32_t {
    Dispatch = 0,
    Compute = 1,
    Combine = 2,
};

struct OverlapStep {
    OverlapStage stage = OverlapStage::Dispatch;
    uint32_t groupId = 0;

    bool operator==(const OverlapStep& other) const {
        return stage == other.stage && groupId == other.groupId;
    }
};

using OverlapTimeline = std::vector<std::vector<OverlapStep>>;

RoutingFixture BuildTwoRankRoutingFixtureForRank1();
RoutingFixture BuildTwoRankMaskAndCapacityFixtureForRank1();
std::vector<uint8_t> BuildDispatchPublicationForRank(const RoutingFixture& fixture, uint32_t srcRank);
std::vector<uint8_t> BuildHostDispatchOracle(const RoutingFixture& fixture,
                                             uint32_t localRank,
                                             const routing::RoutingPlanBundle& plan);
HostComputeOracle BuildHostComputeOracle(uint32_t localRank = 0);
HostCombineOracle BuildHostCombineOracle(uint32_t localRank);
HostFullChainOracle BuildHostFullChainOracle(uint32_t localRank);
OverlapTimeline BuildSteadyStateTimeline(uint32_t groupCount);
bool CompareBytes(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs);
bool CompareFloats(const std::vector<float>& lhs, const std::vector<float>& rhs, float atol = 1e-5f);

}  // namespace mc2::v4
