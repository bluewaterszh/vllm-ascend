#include "../case_io.hpp"
#include "../op_kernel/routing/route_plan.hpp"

#include <cassert>
#include <utility>
#include <vector>

int main() {
    auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    auto bundle = mc2::v4::routing::BuildRoutePlanForRank(fixture, /*localRank=*/1);
    const auto& dispatch = bundle.view.dispatch;
    const auto& combine = bundle.view.combine;

    const std::vector<uint32_t> expectedTokenPerExpert{1, 1, 1, 2};
    const std::vector<uint32_t> expectedGatheredExpertCount{2, 3};
    const std::vector<uint32_t> expectedLocalExpertPrefix{0, 2};
    const std::vector<uint32_t> expectedCumsumMM{0, 1, 0, 1};
    const std::vector<uint32_t> expectedSrcOffset{0, 1, 0, 1, 2};
    const std::vector<uint32_t> expectedDstOffset{0, 2, 1, 3, 4};
    const std::vector<std::pair<uint32_t, uint32_t>> expectedRanges{{0, 2}, {2, 4}};
    const std::vector<uint32_t> expectedCombineDstOffset{0, 1, 2, 3};

    assert(dispatch.tokenPerExpert == expectedTokenPerExpert);
    assert(dispatch.gatheredExpertCount == expectedGatheredExpertCount);
    assert(dispatch.localExpertPrefix == expectedLocalExpertPrefix);
    assert(dispatch.cumsumMM == expectedCumsumMM);
    assert(dispatch.srcOffset == expectedSrcOffset);
    assert(dispatch.dstOffset == expectedDstOffset);
    assert(combine.rowToExpandedRange == expectedRanges);
    assert(combine.combineDstOffset == expectedCombineDstOffset);
    return 0;
}
