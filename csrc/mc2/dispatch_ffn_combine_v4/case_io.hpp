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
    uint32_t outputBytes = 512;
    std::vector<uint32_t> rowsPerRank;
    std::vector<std::vector<uint8_t>> xActiveMaskPerRank;
    std::vector<std::vector<uint32_t>> topkIdsPerRank;
    std::vector<std::vector<float>> topkProbsPerRank;
};

struct ComputeFixture {
    std::vector<float> input;
    std::vector<float> weight1;
    std::vector<float> weight2;
};

struct HostComputeOracle {
    ComputeFixture fixture;
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

RoutingFixture BuildTwoRankRoutingFixtureForRank1();
RoutingFixture BuildTwoRankMaskAndCapacityFixtureForRank1();
std::vector<uint8_t> BuildDispatchPublicationForRank(const RoutingFixture& fixture, uint32_t srcRank);
std::vector<uint8_t> BuildHostDispatchOracle(const RoutingFixture& fixture,
                                             uint32_t localRank,
                                             const routing::RoutingPlanBundle& plan);
HostComputeOracle BuildHostComputeOracle(uint32_t localRank = 0);
HostCombineOracle BuildHostCombineOracle(uint32_t localRank);
HostFullChainOracle BuildHostFullChainOracle(uint32_t localRank);
bool CompareBytes(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs);
bool CompareFloats(const std::vector<float>& lhs, const std::vector<float>& rhs, float atol = 1e-5f);

}  // namespace mc2::v4
