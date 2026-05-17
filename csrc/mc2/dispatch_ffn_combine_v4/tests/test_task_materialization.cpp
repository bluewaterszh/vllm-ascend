#include "../case_io.hpp"
#include "../op_kernel/routing/route_plan.hpp"
#include "../op_kernel/dispatch/dispatch_progress.hpp"
#include "../op_kernel/protocol/ready_queue.hpp"

#include <algorithm>
#include <cassert>

int main() {
    auto fixture = mc2::v4::BuildTwoRankMaskAndCapacityFixtureForRank1();
    auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, /*localRank=*/1);

    assert(plan.dispatchTasks.size() == 3);
    assert(plan.combineTasks.size() == 3);

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
            return task.srcRank == 1 && task.dstRank == 0 && task.dstPayloadOffsetBytes == 0;
        });
    assert(hasRemoteOwnerReturn);
    assert(plan.combineTasks[0].dstPayloadOffsetBytes == 0);
    assert(plan.combineTasks[1].dstPayloadOffsetBytes == mc2::v4::protocol::kCombineTransportBytes);
    assert(plan.combineTasks[2].dstPayloadOffsetBytes == 2 * mc2::v4::protocol::kCombineTransportBytes);

    const auto rank0Oracle = mc2::v4::BuildHostCombineOracle(0);
    const auto rank1Oracle = mc2::v4::BuildHostCombineOracle(1);
    assert(mc2::v4::CompareFloats(rank0Oracle.restoredRows, std::vector<float>({3.2f, 3.4f}), 1e-6f));
    assert(mc2::v4::CompareFloats(rank1Oracle.restoredRows, std::vector<float>({4.0f, 2.9f}), 1e-6f));

    const auto timeline = mc2::v4::BuildSteadyStateTimeline(/*groupCount=*/3);
    const auto expectedTick0 = std::vector<mc2::v4::OverlapStep>{{mc2::v4::OverlapStage::Dispatch, 0}};
    const auto expectedTick2 = std::vector<mc2::v4::OverlapStep>{{mc2::v4::OverlapStage::Dispatch, 2},
                                                                  {mc2::v4::OverlapStage::Compute, 1},
                                                                  {mc2::v4::OverlapStage::Combine, 0}};
    assert(timeline.size() == 5);
    assert(timeline[0] == expectedTick0);
    assert(timeline[2] == expectedTick2);

    mc2::v4::dispatch::DispatchGroupProgress progress{};
    progress.groupId = 0;
    progress.expectedSourceCount = 3;
    progress.completedSourceCount = 2;
    progress.publishedEpoch = 1;
    auto summary = mc2::v4::dispatch::BuildDispatchSummaryView(progress);
    assert(!mc2::v4::dispatch::CanLaunchCompute(progress, summary));
    progress.completedSourceCount = 3;
    summary = mc2::v4::dispatch::BuildDispatchSummaryView(progress);
    assert(mc2::v4::dispatch::CanLaunchCompute(progress, summary));

    const auto schedule = mc2::v4::protocol::BuildExpertGroupSchedule(/*experts=*/16);
    assert(schedule.widths == std::vector<uint32_t>({8, 4, 2, 1, 1}));
    return 0;
}
