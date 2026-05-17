#include "../case_io.hpp"
#include "../op_kernel/routing/route_plan.hpp"

#include <algorithm>
#include <cassert>

int main() {
    auto fixture = mc2::v4::BuildTwoRankMaskAndCapacityFixtureForRank1();
    auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, /*localRank=*/1);

    assert(plan.dispatchTasks.size() == 3);
    assert(plan.combineTasks.size() == 1);

    const auto hasSentinelDispatch = std::any_of(
        plan.dispatchTasks.begin(), plan.dispatchTasks.end(),
        [&](const mc2::v4::protocol::DispatchPullTask& task) {
            return task.expertId == fixture.worldSize * fixture.expertsPerRank;
        });
    assert(!hasSentinelDispatch);

    const auto hasClippedLocalDispatch = std::any_of(
        plan.dispatchTasks.begin(), plan.dispatchTasks.end(),
        [](const mc2::v4::protocol::DispatchPullTask& task) {
            return task.srcRank == 1 && task.dstRank == 1;
        });
    assert(!hasClippedLocalDispatch);

    const auto hasRemoteOwnerReturn = std::any_of(
        plan.combineTasks.begin(), plan.combineTasks.end(),
        [](const mc2::v4::protocol::CombinePushTask& task) {
            return task.srcRank == 0 && task.dstRank == 1 && task.dstPayloadOffsetBytes == 0;
        });
    assert(hasRemoteOwnerReturn);

    const auto rank0Oracle = mc2::v4::BuildHostCombineOracle(0);
    const auto rank1Oracle = mc2::v4::BuildHostCombineOracle(1);
    assert(mc2::v4::CompareFloats(rank0Oracle.restoredRows, std::vector<float>({3.5f}), 1e-6f));
    assert(mc2::v4::CompareFloats(rank1Oracle.restoredRows, std::vector<float>({0.0f}), 1e-6f));
    return 0;
}
