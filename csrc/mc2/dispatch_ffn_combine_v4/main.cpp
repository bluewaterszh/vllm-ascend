#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "case_io.hpp"
#include "comm_mpi.hpp"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/combine/combine_progress.hpp"
#include "op_kernel/dispatch/dispatch_progress.hpp"
#include "op_kernel/protocol/ready_queue.hpp"
#include "op_kernel/protocol/signal_protocol.hpp"
#include "op_kernel/routing/route_plan.hpp"
#include "runtime_context.hpp"

namespace {

mc2::v4::ModeConfig ParseMode(int argc, char** argv)
{
    mc2::v4::ModeConfig cfg;
    auto parseUint = [](const char* text) -> uint32_t {
        return static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && (i + 1) < argc) {
            cfg.mode = argv[++i];
        } else if (arg == "--world-size" && (i + 1) < argc) {
            cfg.worldSize = parseUint(argv[++i]);
        } else if (arg == "--m" && (i + 1) < argc) {
            cfg.m = parseUint(argv[++i]);
        } else if (arg == "--k" && (i + 1) < argc) {
            cfg.k = parseUint(argv[++i]);
        } else if (arg == "--n" && (i + 1) < argc) {
            cfg.n = parseUint(argv[++i]);
        } else if (arg == "--topk" && (i + 1) < argc) {
            cfg.topk = parseUint(argv[++i]);
        } else if (arg == "--experts" && (i + 1) < argc) {
            cfg.expertsPerRank = parseUint(argv[++i]);
        } else if (arg == "--max-output-size" && (i + 1) < argc) {
            cfg.maxOutputSize = parseUint(argv[++i]);
        } else if (arg == "--check-golden") {
            cfg.checkGolden = true;
        } else if (!arg.empty() && arg[0] != '-') {
            cfg.mode = arg;
        }
    }
    return cfg;
}

int RunSignalRoundtrip()
{
    using namespace mc2::v4::protocol;
    const auto a = DispatchReadyIndex(0, 0);
    const auto b = DispatchReadyIndex(1, 2);
    const auto c = CombineReadyIndex(0, 0);
    return (a == 0 && b != a && c > b) ? 0 : 1;
}

bool VerifyMetadataGolden()
{
    auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    auto bundle = mc2::v4::routing::BuildRoutePlanForRank(fixture, 1);
    const auto& dispatch = bundle.view.dispatch;
    const auto& combine = bundle.view.combine;

    return dispatch.tokenPerExpert == std::vector<uint32_t>({1, 1, 1, 2}) &&
           dispatch.gatheredExpertCount == std::vector<uint32_t>({2, 3}) &&
           dispatch.localExpertPrefix == std::vector<uint32_t>({0, 2}) &&
           dispatch.cumsumMM == std::vector<uint32_t>({0, 1, 0, 1}) &&
           dispatch.srcOffset == std::vector<uint32_t>({0, 1, 0, 1, 2}) &&
           dispatch.dstOffset == std::vector<uint32_t>({0, 2, 1, 3, 4}) &&
           combine.rowToExpandedRange == std::vector<std::pair<uint32_t, uint32_t>>({{0, 2}, {2, 4}}) &&
           combine.combineDstOffset == std::vector<uint32_t>({0, 1, 2, 3});
}

void* AddBytes(void* base, uint64_t offset)
{
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(base) + offset);
}

const void* AddBytes(const void* base, uint64_t offset)
{
    return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(base) + offset);
}

std::vector<mc2::v4::protocol::CombineRowRange> BuildCombineRowRanges(
    const mc2::v4::routing::CombineTruthView& combine)
{
    std::vector<mc2::v4::protocol::CombineRowRange> rowRanges;
    rowRanges.reserve(combine.rowToExpandedRange.size());
    for (const auto& range : combine.rowToExpandedRange) {
        rowRanges.push_back({range.first, range.second});
    }
    return rowRanges;
}

std::vector<float> BuildCombineTransportPayload(const std::vector<float>& logicalPayload)
{
    std::vector<float> transport(logicalPayload.size() * mc2::v4::protocol::kCombineTransportElems, 0.0f);
    for (size_t i = 0; i < logicalPayload.size(); ++i) {
        transport[i * mc2::v4::protocol::kCombineTransportElems] = logicalPayload[i];
    }
    return transport;
}

std::vector<float> BuildCombineTransportPayloadForTasks(
    const std::vector<mc2::v4::protocol::CombinePushTask>& tasks,
    size_t slotCount,
    float scalar)
{
    std::vector<float> transport(slotCount * mc2::v4::protocol::kCombineTransportElems, 0.0f);
    for (const auto& task : tasks) {
        const size_t slot = task.srcPayloadOffsetBytes / mc2::v4::protocol::kCombineTransportBytes;
        if (slot < slotCount) {
            transport[slot * mc2::v4::protocol::kCombineTransportElems] = scalar;
        }
    }
    return transport;
}

std::vector<mc2::v4::protocol::DispatchPullTask> FilterDispatchTasksByGroup(
    const std::vector<mc2::v4::protocol::DispatchPullTask>& tasks,
    uint32_t groupId)
{
    std::vector<mc2::v4::protocol::DispatchPullTask> out;
    for (const auto& task : tasks) {
        if (task.expertGroupId == groupId) {
            out.push_back(task);
        }
    }
    return out;
}

std::vector<mc2::v4::protocol::CombinePushTask> FilterCombineTasksByGroup(
    const std::vector<mc2::v4::protocol::CombinePushTask>& tasks,
    uint32_t groupId)
{
    std::vector<mc2::v4::protocol::CombinePushTask> out;
    for (const auto& task : tasks) {
        if (task.expertGroupId == groupId) {
            out.push_back(task);
        }
    }
    return out;
}

uint32_t CountExpectedSourceCount(const std::vector<mc2::v4::protocol::DispatchPullTask>& tasks)
{
    bool seen[mc2::v4::protocol::kMaxRemoteRanks] = {false};
    uint32_t count = 0;
    for (const auto& task : tasks) {
        if (!seen[task.srcRank]) {
            seen[task.srcRank] = true;
            ++count;
        }
    }
    return count;
}

std::string BuildScheduleTag(const mc2::v4::protocol::ExpertGroupSchedule& schedule)
{
    std::string tag;
    for (size_t i = 0; i < schedule.widths.size(); ++i) {
        if (!tag.empty()) {
            tag += 'x';
        }
        tag += std::to_string(schedule.widths[i]);
    }
    return tag.empty() ? std::string("none") : tag;
}

uint64_t CountDispatchQuantBytes(const std::vector<mc2::v4::protocol::DispatchPullTask>& tasks)
{
    uint64_t bytes = 0;
    for (const auto& task : tasks) {
        bytes += static_cast<uint64_t>(task.rowCount) * task.scaleBytes;
    }
    return bytes;
}

bool CopyHostToDeviceRegion(void* base, uint64_t offset, const void* src, size_t bytes)
{
    if (bytes == 0) {
        return true;
    }
    return aclrtMemcpy(AddBytes(base, offset), bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool CopyDeviceToHostRegion(std::vector<uint8_t>& dst, const void* base, uint64_t offset, size_t bytes)
{
    dst.resize(bytes);
    if (bytes == 0) {
        return true;
    }
    return aclrtMemcpy(dst.data(), bytes, AddBytes(base, offset), bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool AllocateAndCopyDeviceBuffer(void*& dst, const void* src, size_t bytes)
{
    dst = nullptr;
    if (bytes == 0) {
        return true;
    }
    if (aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || dst == nullptr) {
        return false;
    }
    return aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool AllocateDeviceBuffer(void*& dst, size_t bytes)
{
    dst = nullptr;
    if (bytes == 0) {
        return true;
    }
    return aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS && dst != nullptr;
}

bool CopyDeviceToHostFloats(std::vector<float>& dst, const void* src, size_t count)
{
    dst.resize(count);
    if (count == 0) {
        return true;
    }
    const size_t bytes = count * sizeof(float);
    return aclrtMemcpy(dst.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

size_t FirstMismatch(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs)
{
    const size_t n = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
    for (size_t i = 0; i < n; ++i) {
        if (lhs[i] != rhs[i]) {
            return i;
        }
    }
    return n;
}

bool ExchangeStageReady(uint8_t localReady, int worldSize, int rank)
{
    std::vector<uint8_t> gathered(rank == 0 ? static_cast<size_t>(worldSize) : 0, 0);
    CommMpiGather(&localReady,
                  1,
                  COMM_MPI_CHAR,
                  rank == 0 ? gathered.data() : nullptr,
                  1,
                  COMM_MPI_CHAR,
                  0);
    uint8_t globalReady = 1;
    if (rank == 0) {
        for (uint8_t value : gathered) {
            if (value == 0) {
                globalReady = 0;
                break;
            }
        }
    }
    CommMpiBcast(&globalReady, 1, COMM_MPI_CHAR, 0);
    return globalReady != 0;
}

int RunComputeOnly()
{
    const auto oracle = mc2::v4::BuildHostComputeOracle();
    bool aclReady = false;
    bool pass = false;
    bool ok = true;
    const char* failStage = "acl-init";
    aclrtStream stream = nullptr;
    void* inputDev = nullptr;
    void* quantInputDev = nullptr;
    void* scale1Dev = nullptr;
    void* scale2Dev = nullptr;
    void* weight1Dev = nullptr;
    void* gmm1OutDev = nullptr;
    void* swigluOutDev = nullptr;
    void* weight2Dev = nullptr;
    void* gmm2OutDev = nullptr;
    void* paramsDev = nullptr;
    void* tilingDev = nullptr;

    ok = aclInit(nullptr) == ACL_SUCCESS;
    aclReady = ok;
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(0) == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "create-stream";
        ok = aclrtCreateStream(&stream) == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "input-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(inputDev,
                                         oracle.fixture.input.data(),
                                         oracle.fixture.input.size() * sizeof(float));
    }
    if (ok) {
        failStage = "quant-input-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(quantInputDev,
                                         oracle.quantPayload.data(),
                                         oracle.quantPayload.size() * sizeof(int8_t));
    }
    if (ok) {
        failStage = "scale1-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(scale1Dev,
                                         oracle.scale1.data(),
                                         oracle.scale1.size() * sizeof(float));
    }
    if (ok) {
        failStage = "scale2-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(scale2Dev,
                                         oracle.scale2.data(),
                                         oracle.scale2.size() * sizeof(float));
    }
    if (ok) {
        failStage = "weight1-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(weight1Dev,
                                         oracle.fixture.weight1.data(),
                                         oracle.fixture.weight1.size() * sizeof(float));
    }
    if (ok) {
        failStage = "gmm1-buffer-alloc";
        ok = AllocateDeviceBuffer(gmm1OutDev, oracle.gmm1Out.size() * sizeof(float));
    }
    if (ok) {
        failStage = "swiglu-buffer-alloc";
        ok = AllocateDeviceBuffer(swigluOutDev, oracle.swigluOut.size() * sizeof(float));
    }
    if (ok) {
        failStage = "weight2-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(weight2Dev,
                                         oracle.fixture.weight2.data(),
                                         oracle.fixture.weight2.size() * sizeof(float));
    }
    if (ok) {
        failStage = "gmm2-buffer-alloc";
        ok = AllocateDeviceBuffer(gmm2OutDev, oracle.gmm2Out.size() * sizeof(float));
    }

    mc2::v4::ComputeOnlyParams params{};
    params.input = reinterpret_cast<uint64_t>(inputDev);
    params.quantInput = reinterpret_cast<uint64_t>(quantInputDev);
    params.scale1 = reinterpret_cast<uint64_t>(scale1Dev);
    params.scale2 = reinterpret_cast<uint64_t>(scale2Dev);
    params.weight1 = reinterpret_cast<uint64_t>(weight1Dev);
    params.gmm1Out = reinterpret_cast<uint64_t>(gmm1OutDev);
    params.swigluOut = reinterpret_cast<uint64_t>(swigluOutDev);
    params.weight2 = reinterpret_cast<uint64_t>(weight2Dev);
    params.gmm2Out = reinterpret_cast<uint64_t>(gmm2OutDev);
    if (ok) {
        failStage = "params-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(paramsDev, &params, sizeof(params));
    }

    mc2::v4::StandaloneKernelTilingData tiling{};
    tiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::ComputeOnly);
    if (ok) {
        failStage = "tiling-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(tilingDev, &tiling, sizeof(tiling));
    }

    std::vector<float> actual;
    if (ok) {
        failStage = "kernel-launch";
        mc2::v4::ComputeOnlyLaunchArgs launchArgs;
        launchArgs.params = paramsDev;
        launchArgs.tiling = tilingDev;
        launchArgs.blockDim = 1;
        mc2::v4::launchComputeOnly(launchArgs, stream);
        const aclError syncErr = aclrtSynchronizeStream(stream);
        if (syncErr != ACL_SUCCESS) {
            std::cerr << "compute-only syncErr=" << syncErr
                      << " lastErr=" << aclrtGetLastError(ACL_RT_THREAD_LEVEL)
                      << " recentErrMsg=" << aclGetRecentErrMsg() << '\n';
        }
        ok = syncErr == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "copy-back";
        ok = CopyDeviceToHostFloats(actual,
                                    gmm2OutDev,
                                    oracle.gmm2Out.size());
    }
    if (ok) {
        pass = mc2::v4::CompareFloats(actual, oracle.gmm2Out, 1e-4f);
        if (!pass) {
            std::cerr << "compute-only mismatch actual=";
            for (size_t i = 0; i < actual.size(); ++i) {
                std::cerr << actual[i] << (i + 1 == actual.size() ? "" : ",");
            }
            std::cerr << " oracle=";
            for (size_t i = 0; i < oracle.gmm2Out.size(); ++i) {
                std::cerr << oracle.gmm2Out[i] << (i + 1 == oracle.gmm2Out.size() ? "" : ",");
            }
            std::cerr << '\n';
        }
    }

    if (!ok) {
        std::cerr << "compute-only failed at stage: " << failStage << '\n';
    }
    std::cout << "compute-only " << ((ok && pass) ? "PASS" : "FAIL") << '\n';

    if (tilingDev != nullptr) {
        aclrtFree(tilingDev);
    }
    if (paramsDev != nullptr) {
        aclrtFree(paramsDev);
    }
    if (gmm2OutDev != nullptr) {
        aclrtFree(gmm2OutDev);
    }
    if (weight2Dev != nullptr) {
        aclrtFree(weight2Dev);
    }
    if (swigluOutDev != nullptr) {
        aclrtFree(swigluOutDev);
    }
    if (gmm1OutDev != nullptr) {
        aclrtFree(gmm1OutDev);
    }
    if (weight1Dev != nullptr) {
        aclrtFree(weight1Dev);
    }
    if (scale2Dev != nullptr) {
        aclrtFree(scale2Dev);
    }
    if (scale1Dev != nullptr) {
        aclrtFree(scale1Dev);
    }
    if (quantInputDev != nullptr) {
        aclrtFree(quantInputDev);
    }
    if (inputDev != nullptr) {
        aclrtFree(inputDev);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (aclReady) {
        aclrtResetDevice(0);
        aclFinalize();
    }
    return (ok && pass) ? 0 : 1;
}

int RunCombineOnly(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank = CommMpiRank();
    const int worldSize = CommMpiSize();
    bool aclReady = false;
    bool pass = false;
    bool ok = worldSize == 2;
    const char* failStage = "world-size-check";
    if (!ok) {
        std::cerr << "combine-only requires worldSize=2, got " << worldSize << '\n';
    }

    mc2::v4::StandaloneRankRuntime runtime{};
    void* partialDev = nullptr;
    void* restoredRowsDev = nullptr;
    void* combineTaskDev = nullptr;
    void* combineProbDev = nullptr;
    void* rowRangesDev = nullptr;
    void* paramsDev = nullptr;
    void* tilingDev = nullptr;
    HcclRootInfo rootInfo{};

    if (ok) {
        failStage = "acl-init";
        ok = aclInit(nullptr) == ACL_SUCCESS;
        aclReady = ok;
    }
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(rank) == ACL_SUCCESS;
    }
    if (rank == 0 && ok) {
        failStage = "hccl-root-info";
        ok = HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);

    mc2::v4::ModeConfig mode;
    mode.mode = "combine-only";
    mode.localRank = static_cast<uint32_t>(rank);
    mode.worldSize = static_cast<uint32_t>(worldSize);
    if (ok) {
        failStage = "runtime-init";
        ok = mc2::v4::InitStandaloneRankRuntime(runtime, mode, rootInfo);
    }

    const auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    const auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, static_cast<uint32_t>(rank));
    const auto oracle = mc2::v4::BuildHostCombineOracle(static_cast<uint32_t>(rank));
    const auto rowRanges = BuildCombineRowRanges(plan.view.combine);
    const auto partialTransport = BuildCombineTransportPayload(oracle.pushedPayload);
    std::vector<float> zeroCombine(runtime.layout.combineBytes / sizeof(float), 0.0f);
    if (ok) {
        failStage = "combine-region-zero";
        void* localWindow = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        ok = localWindow != nullptr &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.combineRegionOffset,
                                    zeroCombine.data(),
                                    zeroCombine.size() * sizeof(float));
    }

    if (ok) {
        failStage = "partial-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(partialDev,
                                         partialTransport.data(),
                                         partialTransport.size() * sizeof(float));
    }
    if (ok) {
        failStage = "restored-rows-buffer-alloc";
        ok = AllocateDeviceBuffer(restoredRowsDev, oracle.restoredRows.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-task-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTaskDev,
                                         plan.combineTasks.data(),
                                         plan.combineTasks.size() * sizeof(mc2::v4::protocol::CombinePushTask));
    }
    if (ok) {
        failStage = "combine-prob-copy";
        ok = AllocateAndCopyDeviceBuffer(combineProbDev,
                                         plan.view.combine.expandedProb.data(),
                                         plan.view.combine.expandedProb.size() * sizeof(float));
    }
    if (ok) {
        failStage = "row-range-copy";
        ok = AllocateAndCopyDeviceBuffer(rowRangesDev,
                                         rowRanges.data(),
                                         rowRanges.size() * sizeof(mc2::v4::protocol::CombineRowRange));
    }

    mc2::v4::CombineOnlyParams params{};
    params.localPartial = reinterpret_cast<uint64_t>(partialDev);
    params.restoredRows = reinterpret_cast<uint64_t>(restoredRowsDev);
    params.combineTasks = reinterpret_cast<uint64_t>(combineTaskDev);
    params.expandedProb = reinterpret_cast<uint64_t>(combineProbDev);
    params.rowRanges = reinterpret_cast<uint64_t>(rowRangesDev);
    params.taskCount = static_cast<uint32_t>(plan.combineTasks.size());
    params.localRowCount = static_cast<uint32_t>(oracle.restoredRows.size());
    params.phase = 1;
    if (ok) {
        failStage = "params-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(paramsDev, &params, sizeof(params));
    }

    mc2::v4::StandaloneKernelTilingData tiling{};
    tiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly);
    if (ok) {
        failStage = "tiling-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(tilingDev, &tiling, sizeof(tiling));
    }

    const bool canLaunchCombine = ExchangeStageReady(ok ? 1 : 0, worldSize, rank);
    if (ok && canLaunchCombine) {
        failStage = "prelaunch-push-barrier";
        ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
             aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    } else if (!canLaunchCombine) {
        ok = false;
    }
    if (ok) {
        failStage = "push-phase-launch";
        mc2::v4::CombineOnlyLaunchArgs launchArgs;
        launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
        launchArgs.params = paramsDev;
        launchArgs.tiling = tilingDev;
        launchArgs.blockDim = params.taskCount == 0 ? 1 : params.taskCount;
        mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
        ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "post-push-barrier";
        ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
             aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "params-phase-update";
        params.phase = 2;
        ok = aclrtMemcpy(paramsDev, sizeof(params), &params, sizeof(params), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }

    std::vector<float> actual;
    if (ok) {
        failStage = "restore-phase-launch";
        mc2::v4::CombineOnlyLaunchArgs launchArgs;
        launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
        launchArgs.params = paramsDev;
        launchArgs.tiling = tilingDev;
        launchArgs.blockDim = 1;
        mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
        ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    }
    if (ok) {
        failStage = "copy-back";
        ok = CopyDeviceToHostFloats(actual, restoredRowsDev, oracle.restoredRows.size());
    }
    if (ok) {
        pass = mc2::v4::CompareFloats(actual, oracle.restoredRows, 1e-6f);
        if (!pass) {
            std::cerr << "[rank " << rank << "] combine-only mismatch actual=";
            for (size_t i = 0; i < actual.size(); ++i) {
                std::cerr << actual[i] << (i + 1 == actual.size() ? "" : ",");
            }
            std::cerr << " oracle=";
            for (size_t i = 0; i < oracle.restoredRows.size(); ++i) {
                std::cerr << oracle.restoredRows[i] << (i + 1 == oracle.restoredRows.size() ? "" : ",");
            }
            std::cerr << '\n';
        }
    }

    if (!ok) {
        std::cerr << "[rank " << rank << "] combine-only failed at stage: " << failStage << '\n';
    }
    std::cout << "[rank " << rank << "] combine-only " << ((ok && pass) ? "PASS" : "FAIL") << '\n';

    if (tilingDev != nullptr) {
        aclrtFree(tilingDev);
    }
    if (paramsDev != nullptr) {
        aclrtFree(paramsDev);
    }
    if (rowRangesDev != nullptr) {
        aclrtFree(rowRangesDev);
    }
    if (combineProbDev != nullptr) {
        aclrtFree(combineProbDev);
    }
    if (combineTaskDev != nullptr) {
        aclrtFree(combineTaskDev);
    }
    if (restoredRowsDev != nullptr) {
        aclrtFree(restoredRowsDev);
    }
    if (partialDev != nullptr) {
        aclrtFree(partialDev);
    }
    mc2::v4::DestroyStandaloneRankRuntime(runtime);
    if (aclReady) {
        aclrtResetDevice(rank);
        aclFinalize();
    }
    CommMpiFinalize();
    return (ok && pass) ? 0 : 1;
}

int RunFullChain(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank = CommMpiRank();
    const int worldSize = CommMpiSize();
    bool aclReady = false;
    bool pass = false;
    bool ok = worldSize == 2;
    const char* failStage = "world-size-check";
    if (!ok) {
        std::cerr << "full-chain requires worldSize=2, got " << worldSize << '\n';
    }

    mc2::v4::StandaloneRankRuntime runtime{};
    HcclRootInfo rootInfo{};
    void* dispatchTaskDev = nullptr;
    void* dispatchTilingDev = nullptr;
    void* inputDev = nullptr;
    void* weight1Dev = nullptr;
    void* gmm1OutDev = nullptr;
    void* swigluOutDev = nullptr;
    void* weight2Dev = nullptr;
    void* gmm2OutDev = nullptr;
    void* computeParamsDev = nullptr;
    void* computeTilingDev = nullptr;
    void* combinePartialDev = nullptr;
    void* restoredRowsDev = nullptr;
    void* combineTaskDev = nullptr;
    void* combineProbDev = nullptr;
    void* rowRangesDev = nullptr;
    void* combineParamsDev = nullptr;
    void* combineTilingDev = nullptr;

    if (ok) {
        failStage = "acl-init";
        ok = aclInit(nullptr) == ACL_SUCCESS;
        aclReady = ok;
    }
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(rank) == ACL_SUCCESS;
    }
    if (rank == 0 && ok) {
        failStage = "hccl-root-info";
        ok = HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);

    mc2::v4::ModeConfig mode;
    mode.mode = "full-chain";
    mode.localRank = static_cast<uint32_t>(rank);
    mode.worldSize = static_cast<uint32_t>(worldSize);
    if (ok) {
        failStage = "runtime-init";
        ok = mc2::v4::InitStandaloneRankRuntime(runtime, mode, rootInfo);
    }

    const auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    const auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, static_cast<uint32_t>(rank));
    const auto publication = mc2::v4::BuildDispatchPublicationForRank(fixture, static_cast<uint32_t>(rank));
    const auto fullOracle = mc2::v4::BuildHostFullChainOracle(static_cast<uint32_t>(rank));
    const auto computeOracle = mc2::v4::BuildHostComputeOracle(static_cast<uint32_t>(rank));
    const auto rowRanges = BuildCombineRowRanges(plan.view.combine);

    std::vector<uint8_t> zeroCompute(runtime.layout.computeBytes, 0);
    std::vector<uint8_t> zeroCombine(runtime.layout.combineBytes, 0);
    std::vector<int32_t> signalWords(runtime.layout.signalBytes / sizeof(int32_t), 0);
    for (const auto& task : plan.dispatchTasks) {
        const uint32_t readyIndex = mc2::v4::protocol::DispatchReadyIndex(task.srcRank, task.srcRowBegin);
        if (readyIndex < signalWords.size()) {
            signalWords[readyIndex] = static_cast<int32_t>(task.readyEpoch);
        }
    }

    if (ok) {
        failStage = "dispatch-seed";
        void* localWindow = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        ok = localWindow != nullptr &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.dispatchRegionOffset,
                                    publication.data(),
                                    publication.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                    zeroCompute.data(),
                                    zeroCompute.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.combineRegionOffset,
                                    zeroCombine.data(),
                                    zeroCombine.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.signalRegionOffset,
                                    signalWords.data(),
                                    signalWords.size() * sizeof(int32_t));
    }
    if (ok) {
        failStage = "dispatch-task-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchTaskDev,
                                         plan.dispatchTasks.data(),
                                         plan.dispatchTasks.size() * sizeof(mc2::v4::protocol::DispatchPullTask));
    }
    mc2::v4::StandaloneKernelTilingData dispatchTiling{};
    dispatchTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::DispatchOnly);
    dispatchTiling.taskCount = static_cast<uint32_t>(plan.dispatchTasks.size());
    dispatchTiling.hiddenBytes = fixture.hiddenBytes;
    dispatchTiling.outputBytes = fixture.outputBytes;
    if (ok) {
        failStage = "dispatch-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchTilingDev, &dispatchTiling, sizeof(dispatchTiling));
    }

    bool dispatchPass = false;
    std::vector<uint8_t> dispatchActual;
    if (ok) {
        const bool canLaunch = ExchangeStageReady(1, worldSize, rank);
        ok = canLaunch;
        if (ok) {
            failStage = "dispatch-launch";
            const double t0 = mc2::v4::NowMs();
            mc2::v4::DispatchOnlyLaunchArgs launchArgs;
            launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
            launchArgs.dispatchTasks = dispatchTaskDev;
            launchArgs.tiling = dispatchTilingDev;
            launchArgs.blockDim = dispatchTiling.taskCount == 0 ? 1 : dispatchTiling.taskCount;
            mc2::v4::launchDispatchFFNCombine(launchArgs, runtime.compute_stream);
            ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
            if (ok) {
                ok = CopyDeviceToHostRegion(dispatchActual,
                                            runtime.hccl.WindowIn(static_cast<uint32_t>(rank)),
                                            runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                            fullOracle.dispatchBytes.size());
            }
            if (ok) {
                dispatchPass = mc2::v4::CompareBytes(dispatchActual, fullOracle.dispatchBytes);
            }
            mc2::v4::PrintPerfRecord({"full-chain", "dispatch", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                      static_cast<uint64_t>(fullOracle.dispatchBytes.size()),
                                      static_cast<uint32_t>(plan.dispatchTasks.size())});
        }
    }

    std::vector<float> zeroPartial(8, 0.0f);
    if (ok) {
        failStage = "compute-input-copy";
        ok = AllocateAndCopyDeviceBuffer(inputDev,
                                         computeOracle.fixture.input.data(),
                                         computeOracle.fixture.input.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-weight1-copy";
        ok = AllocateAndCopyDeviceBuffer(weight1Dev,
                                         computeOracle.fixture.weight1.data(),
                                         computeOracle.fixture.weight1.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-gmm1-alloc";
        ok = AllocateDeviceBuffer(gmm1OutDev, computeOracle.gmm1Out.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-swiglu-alloc";
        ok = AllocateDeviceBuffer(swigluOutDev, computeOracle.swigluOut.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-weight2-copy";
        ok = AllocateAndCopyDeviceBuffer(weight2Dev,
                                         computeOracle.fixture.weight2.data(),
                                         computeOracle.fixture.weight2.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-gmm2-alloc";
        ok = AllocateAndCopyDeviceBuffer(gmm2OutDev,
                                         zeroPartial.data(),
                                         zeroPartial.size() * sizeof(float));
    }
    mc2::v4::ComputeOnlyParams computeParams{};
    computeParams.input = reinterpret_cast<uint64_t>(inputDev);
    computeParams.weight1 = reinterpret_cast<uint64_t>(weight1Dev);
    computeParams.gmm1Out = reinterpret_cast<uint64_t>(gmm1OutDev);
    computeParams.swigluOut = reinterpret_cast<uint64_t>(swigluOutDev);
    computeParams.weight2 = reinterpret_cast<uint64_t>(weight2Dev);
    computeParams.gmm2Out = reinterpret_cast<uint64_t>(gmm2OutDev);
    if (ok) {
        failStage = "compute-params-copy";
        ok = AllocateAndCopyDeviceBuffer(computeParamsDev, &computeParams, sizeof(computeParams));
    }
    mc2::v4::StandaloneKernelTilingData computeTiling{};
    computeTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::ComputeOnly);
    if (ok) {
        failStage = "compute-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(computeTilingDev, &computeTiling, sizeof(computeTiling));
    }

    bool computePass = false;
    std::vector<float> computeActual;
    if (ok) {
        failStage = "compute-launch";
        const double t0 = mc2::v4::NowMs();
        mc2::v4::ComputeOnlyLaunchArgs launchArgs;
        launchArgs.params = computeParamsDev;
        launchArgs.tiling = computeTilingDev;
        launchArgs.blockDim = 1;
        mc2::v4::launchComputeOnly(launchArgs, runtime.compute_stream);
        ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
        if (ok) {
            ok = CopyDeviceToHostFloats(computeActual, gmm2OutDev, computeOracle.gmm2Out.size());
        }
        if (ok) {
            computePass = mc2::v4::CompareFloats(computeActual, computeOracle.gmm2Out, 1e-6f);
        }
        mc2::v4::PrintPerfRecord({"full-chain", "compute", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                  static_cast<uint64_t>(computeOracle.gmm2Out.size() * sizeof(float)),
                                  static_cast<uint32_t>(computeOracle.gmm2Out.size())});
    }

    if (ok) {
        failStage = "combine-region-reset";
        ok = CopyHostToDeviceRegion(runtime.hccl.WindowIn(static_cast<uint32_t>(rank)),
                                    runtime.hccl.host_remote_window_ctx.combineRegionOffset,
                                    zeroCombine.data(),
                                    zeroCombine.size());
    }
    std::vector<float> combinePartialHost;
    std::vector<float> combinePartialTransport;
    if (ok) {
        failStage = "combine-partial-expand";
        const float producerScalar = computeActual.empty() ? 0.0f : computeActual[0];
        combinePartialHost.assign(plan.combineTasks.size(), producerScalar);
        combinePartialTransport = BuildCombineTransportPayload(combinePartialHost);
    }
    if (ok) {
        failStage = "combine-partial-copy";
        ok = AllocateAndCopyDeviceBuffer(combinePartialDev,
                                         combinePartialTransport.data(),
                                         combinePartialTransport.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-restored-alloc";
        ok = AllocateDeviceBuffer(restoredRowsDev, fullOracle.output.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-task-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTaskDev,
                                         plan.combineTasks.data(),
                                         plan.combineTasks.size() * sizeof(mc2::v4::protocol::CombinePushTask));
    }
    if (ok) {
        failStage = "combine-prob-copy";
        ok = AllocateAndCopyDeviceBuffer(combineProbDev,
                                         plan.view.combine.expandedProb.data(),
                                         plan.view.combine.expandedProb.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-row-range-copy";
        ok = AllocateAndCopyDeviceBuffer(rowRangesDev,
                                         rowRanges.data(),
                                         rowRanges.size() * sizeof(mc2::v4::protocol::CombineRowRange));
    }
    mc2::v4::CombineOnlyParams combineParams{};
    combineParams.localPartial = reinterpret_cast<uint64_t>(combinePartialDev);
    combineParams.restoredRows = reinterpret_cast<uint64_t>(restoredRowsDev);
    combineParams.combineTasks = reinterpret_cast<uint64_t>(combineTaskDev);
    combineParams.expandedProb = reinterpret_cast<uint64_t>(combineProbDev);
    combineParams.rowRanges = reinterpret_cast<uint64_t>(rowRangesDev);
    combineParams.taskCount = static_cast<uint32_t>(plan.combineTasks.size());
    combineParams.localRowCount = static_cast<uint32_t>(fullOracle.output.size());
    combineParams.phase = 1;
    if (ok) {
        failStage = "combine-params-copy";
        ok = AllocateAndCopyDeviceBuffer(combineParamsDev, &combineParams, sizeof(combineParams));
    }
    mc2::v4::StandaloneKernelTilingData combineTiling{};
    combineTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly);
    if (ok) {
        failStage = "combine-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTilingDev, &combineTiling, sizeof(combineTiling));
    }

    bool combinePass = false;
    std::vector<float> finalActual;
    const bool canLaunchCombine = ExchangeStageReady(ok ? 1 : 0, worldSize, rank);
    if (ok && canLaunchCombine) {
        failStage = "prelaunch-combine-barrier";
        ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
             aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    } else if (!canLaunchCombine) {
        ok = false;
    }
    if (ok) {
        failStage = "combine-launch";
        const double t0 = mc2::v4::NowMs();
        mc2::v4::CombineOnlyLaunchArgs launchArgs;
        launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
        launchArgs.params = combineParamsDev;
        launchArgs.tiling = combineTilingDev;
        launchArgs.blockDim = combineParams.taskCount == 0 ? 1 : combineParams.taskCount;
        mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
        ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
        if (ok) {
            ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
                 aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
        }
        if (ok) {
            combineParams.phase = 2;
            ok = aclrtMemcpy(combineParamsDev,
                             sizeof(combineParams),
                             &combineParams,
                             sizeof(combineParams),
                             ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
        }
        if (ok) {
            launchArgs.blockDim = 1;
            mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
            ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
        }
        if (ok) {
            ok = CopyDeviceToHostFloats(finalActual, restoredRowsDev, fullOracle.output.size());
        }
        if (ok) {
            combinePass = mc2::v4::CompareFloats(finalActual, fullOracle.output, 1e-6f);
        }
        mc2::v4::PrintPerfRecord({"full-chain", "combine", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                  static_cast<uint64_t>(combinePartialTransport.size() * sizeof(float) +
                                                        fullOracle.output.size() * sizeof(float)),
                                  static_cast<uint32_t>(plan.combineTasks.size() + fullOracle.output.size())});
    }

    pass = ok && dispatchPass && computePass && combinePass;
    mc2::v4::PrintPerfRecord({"full-chain", "summary", static_cast<uint32_t>(rank), 0.0,
                              static_cast<uint64_t>(fullOracle.dispatchBytes.size() +
                                                    computeOracle.gmm2Out.size() * sizeof(float) +
                                                    plan.combineTasks.size() * mc2::v4::protocol::kCombineTransportBytes +
                                                    fullOracle.output.size() * sizeof(float)),
                              static_cast<uint32_t>(plan.dispatchTasks.size() +
                                                    computeOracle.gmm2Out.size() +
                                                    plan.combineTasks.size() +
                                                    fullOracle.output.size())});
    if (!ok) {
        std::cerr << "[rank " << rank << "] full-chain failed at stage: " << failStage << '\n';
    }
    std::cout << "[rank " << rank << "] full-chain " << (pass ? "PASS" : "FAIL") << '\n';

    if (combineTilingDev != nullptr) {
        aclrtFree(combineTilingDev);
    }
    if (combineParamsDev != nullptr) {
        aclrtFree(combineParamsDev);
    }
    if (rowRangesDev != nullptr) {
        aclrtFree(rowRangesDev);
    }
    if (combineProbDev != nullptr) {
        aclrtFree(combineProbDev);
    }
    if (combineTaskDev != nullptr) {
        aclrtFree(combineTaskDev);
    }
    if (restoredRowsDev != nullptr) {
        aclrtFree(restoredRowsDev);
    }
    if (combinePartialDev != nullptr) {
        aclrtFree(combinePartialDev);
    }
    if (computeTilingDev != nullptr) {
        aclrtFree(computeTilingDev);
    }
    if (computeParamsDev != nullptr) {
        aclrtFree(computeParamsDev);
    }
    if (gmm2OutDev != nullptr) {
        aclrtFree(gmm2OutDev);
    }
    if (weight2Dev != nullptr) {
        aclrtFree(weight2Dev);
    }
    if (swigluOutDev != nullptr) {
        aclrtFree(swigluOutDev);
    }
    if (gmm1OutDev != nullptr) {
        aclrtFree(gmm1OutDev);
    }
    if (weight1Dev != nullptr) {
        aclrtFree(weight1Dev);
    }
    if (inputDev != nullptr) {
        aclrtFree(inputDev);
    }
    if (dispatchTilingDev != nullptr) {
        aclrtFree(dispatchTilingDev);
    }
    if (dispatchTaskDev != nullptr) {
        aclrtFree(dispatchTaskDev);
    }
    mc2::v4::DestroyStandaloneRankRuntime(runtime);
    if (aclReady) {
        aclrtResetDevice(rank);
        aclFinalize();
    }
    CommMpiFinalize();
    return pass ? 0 : 1;
}

int RunFullChainOverlap(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    auto mode = ParseMode(argc, argv);
    const int rank = CommMpiRank();
    const int worldSize = CommMpiSize();
    bool aclReady = false;
    bool pass = false;
    bool ok = worldSize == 2;
    const char* failStage = "world-size-check";
    if (!ok) {
        std::cerr << "full-chain requires worldSize=2, got " << worldSize << '\n';
    }

    mode.localRank = static_cast<uint32_t>(rank);
    mode.worldSize = static_cast<uint32_t>(worldSize);
    mode.mode = "full-chain";

    mc2::v4::StandaloneRankRuntime runtime{};
    HcclRootInfo rootInfo{};
    void* dispatchTaskDev = nullptr;
    void* dispatchTilingDev = nullptr;
    void* inputDev = nullptr;
    void* quantInputDev = nullptr;
    void* scale1Dev = nullptr;
    void* scale2Dev = nullptr;
    void* weight1Dev = nullptr;
    void* gmm1OutDev = nullptr;
    void* swigluOutDev = nullptr;
    void* weight2Dev = nullptr;
    void* gmm2OutDev = nullptr;
    void* computeParamsDev = nullptr;
    void* computeTilingDev = nullptr;
    void* combinePartialDev = nullptr;
    void* restoredRowsDev = nullptr;
    void* combineTaskDev = nullptr;
    void* combineProbDev = nullptr;
    void* rowRangesDev = nullptr;
    void* combineParamsDev = nullptr;
    void* combineTilingDev = nullptr;

    if (ok) {
        failStage = "acl-init";
        ok = aclInit(nullptr) == ACL_SUCCESS;
        aclReady = ok;
    }
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(rank) == ACL_SUCCESS;
    }
    if (rank == 0 && ok) {
        failStage = "hccl-root-info";
        ok = HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);

    if (ok) {
        failStage = "runtime-init";
        ok = mc2::v4::InitStandaloneRankRuntime(runtime, mode, rootInfo);
    }

    const auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    const auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, static_cast<uint32_t>(rank));
    const auto publication = mc2::v4::BuildDispatchPublicationForRank(fixture, static_cast<uint32_t>(rank));
    const auto fullOracle = mc2::v4::BuildHostFullChainOracle(static_cast<uint32_t>(rank));
    const auto computeOracle = mc2::v4::BuildHostComputeOracle(static_cast<uint32_t>(rank));
    const auto rowRanges = BuildCombineRowRanges(plan.view.combine);
    const uint32_t groupCount = fixture.expertsPerRank;
    const auto timeline = mc2::v4::BuildSteadyStateTimeline(groupCount);
    const auto activationSchedule = mc2::v4::protocol::BuildExpertGroupSchedule(mode.m >= 4097 ? 16U : groupCount);
    const std::string scheduleTag = BuildScheduleTag(activationSchedule);
    const std::string shapeClass = mode.m >= 4097 ? "large" : "small";
    const uint64_t quantSidebandBytes = CountDispatchQuantBytes(plan.dispatchTasks) +
                                        computeOracle.scale1.size() * sizeof(float) +
                                        computeOracle.scale2.size() * sizeof(float);

    std::vector<uint8_t> zeroCompute(runtime.layout.computeBytes, 0);
    std::vector<uint8_t> zeroCombine(runtime.layout.combineBytes, 0);
    std::vector<int32_t> signalWords(runtime.layout.signalBytes / sizeof(int32_t), 0);
    for (const auto& task : plan.dispatchTasks) {
        const uint32_t readyIndex = mc2::v4::protocol::DispatchReadyIndex(task.srcRank, task.srcRowBegin);
        if (readyIndex < signalWords.size()) {
            signalWords[readyIndex] = static_cast<int32_t>(task.readyEpoch);
        }
    }

    if (ok) {
        failStage = "dispatch-seed";
        void* localWindow = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        ok = localWindow != nullptr &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.dispatchRegionOffset,
                                    publication.data(),
                                    publication.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                    zeroCompute.data(),
                                    zeroCompute.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.combineRegionOffset,
                                    zeroCombine.data(),
                                    zeroCombine.size()) &&
             CopyHostToDeviceRegion(localWindow,
                                    runtime.hccl.host_remote_window_ctx.signalRegionOffset,
                                    signalWords.data(),
                                    signalWords.size() * sizeof(int32_t));
    }

    const size_t dispatchTaskCapacityBytes = plan.dispatchTasks.size() * sizeof(mc2::v4::protocol::DispatchPullTask);
    if (ok) {
        failStage = "dispatch-task-alloc";
        ok = AllocateDeviceBuffer(dispatchTaskDev, dispatchTaskCapacityBytes);
    }
    mc2::v4::StandaloneKernelTilingData dispatchTiling{};
    dispatchTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::DispatchOnly);
    dispatchTiling.hiddenBytes = fixture.hiddenBytes;
    dispatchTiling.outputBytes = fixture.outputBytes;
    if (ok) {
        failStage = "dispatch-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(dispatchTilingDev, &dispatchTiling, sizeof(dispatchTiling));
    }

    if (ok) {
        failStage = "compute-input-copy";
        ok = AllocateAndCopyDeviceBuffer(inputDev,
                                         computeOracle.fixture.input.data(),
                                         computeOracle.fixture.input.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-quant-input-copy";
        ok = AllocateAndCopyDeviceBuffer(quantInputDev,
                                         computeOracle.quantPayload.data(),
                                         computeOracle.quantPayload.size() * sizeof(int8_t));
    }
    if (ok) {
        failStage = "compute-scale1-copy";
        ok = AllocateAndCopyDeviceBuffer(scale1Dev,
                                         computeOracle.scale1.data(),
                                         computeOracle.scale1.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-scale2-copy";
        ok = AllocateAndCopyDeviceBuffer(scale2Dev,
                                         computeOracle.scale2.data(),
                                         computeOracle.scale2.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-weight1-copy";
        ok = AllocateAndCopyDeviceBuffer(weight1Dev,
                                         computeOracle.fixture.weight1.data(),
                                         computeOracle.fixture.weight1.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-gmm1-alloc";
        ok = AllocateDeviceBuffer(gmm1OutDev, computeOracle.gmm1Out.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-swiglu-alloc";
        ok = AllocateDeviceBuffer(swigluOutDev, computeOracle.swigluOut.size() * sizeof(float));
    }
    if (ok) {
        failStage = "compute-weight2-copy";
        ok = AllocateAndCopyDeviceBuffer(weight2Dev,
                                         computeOracle.fixture.weight2.data(),
                                         computeOracle.fixture.weight2.size() * sizeof(float));
    }
    std::vector<float> zeroPartial(computeOracle.gmm2Out.size(), 0.0f);
    if (ok) {
        failStage = "compute-gmm2-alloc";
        ok = AllocateAndCopyDeviceBuffer(gmm2OutDev,
                                         zeroPartial.data(),
                                         zeroPartial.size() * sizeof(float));
    }
    mc2::v4::ComputeOnlyParams computeParams{};
    computeParams.input = reinterpret_cast<uint64_t>(inputDev);
    computeParams.quantInput = reinterpret_cast<uint64_t>(quantInputDev);
    computeParams.scale1 = reinterpret_cast<uint64_t>(scale1Dev);
    computeParams.scale2 = reinterpret_cast<uint64_t>(scale2Dev);
    computeParams.weight1 = reinterpret_cast<uint64_t>(weight1Dev);
    computeParams.gmm1Out = reinterpret_cast<uint64_t>(gmm1OutDev);
    computeParams.swigluOut = reinterpret_cast<uint64_t>(swigluOutDev);
    computeParams.weight2 = reinterpret_cast<uint64_t>(weight2Dev);
    computeParams.gmm2Out = reinterpret_cast<uint64_t>(gmm2OutDev);
    if (ok) {
        failStage = "compute-params-copy";
        ok = AllocateAndCopyDeviceBuffer(computeParamsDev, &computeParams, sizeof(computeParams));
    }
    mc2::v4::StandaloneKernelTilingData computeTiling{};
    computeTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::ComputeOnly);
    if (ok) {
        failStage = "compute-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(computeTilingDev, &computeTiling, sizeof(computeTiling));
    }

    const size_t combinePartialCapacityBytes =
        plan.combineTasks.size() * mc2::v4::protocol::kCombineTransportBytes;
    if (ok) {
        failStage = "combine-partial-alloc";
        ok = AllocateDeviceBuffer(combinePartialDev, combinePartialCapacityBytes);
    }
    if (ok) {
        failStage = "combine-restored-alloc";
        ok = AllocateDeviceBuffer(restoredRowsDev, fullOracle.output.size() * sizeof(float));
    }
    const size_t combineTaskCapacityBytes = plan.combineTasks.size() * sizeof(mc2::v4::protocol::CombinePushTask);
    if (ok) {
        failStage = "combine-task-alloc";
        ok = AllocateDeviceBuffer(combineTaskDev, combineTaskCapacityBytes);
    }
    if (ok) {
        failStage = "combine-prob-copy";
        ok = AllocateAndCopyDeviceBuffer(combineProbDev,
                                         plan.view.combine.expandedProb.data(),
                                         plan.view.combine.expandedProb.size() * sizeof(float));
    }
    if (ok) {
        failStage = "combine-row-range-copy";
        ok = AllocateAndCopyDeviceBuffer(rowRangesDev,
                                         rowRanges.data(),
                                         rowRanges.size() * sizeof(mc2::v4::protocol::CombineRowRange));
    }
    mc2::v4::CombineOnlyParams combineParams{};
    combineParams.localPartial = reinterpret_cast<uint64_t>(combinePartialDev);
    combineParams.restoredRows = reinterpret_cast<uint64_t>(restoredRowsDev);
    combineParams.combineTasks = reinterpret_cast<uint64_t>(combineTaskDev);
    combineParams.expandedProb = reinterpret_cast<uint64_t>(combineProbDev);
    combineParams.rowRanges = reinterpret_cast<uint64_t>(rowRangesDev);
    combineParams.localRowCount = static_cast<uint32_t>(fullOracle.output.size());
    if (ok) {
        failStage = "combine-params-copy";
        ok = AllocateAndCopyDeviceBuffer(combineParamsDev, &combineParams, sizeof(combineParams));
    }
    mc2::v4::StandaloneKernelTilingData combineTiling{};
    combineTiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly);
    combineTiling.outputBytes = fixture.outputBytes;
    if (ok) {
        failStage = "combine-tiling-copy";
        ok = AllocateAndCopyDeviceBuffer(combineTilingDev, &combineTiling, sizeof(combineTiling));
    }

    std::vector<mc2::v4::dispatch::DispatchGroupProgress> dispatchProgress(groupCount);
    std::vector<mc2::v4::combine::CombineGroupProgress> combineProgress(groupCount);
    for (uint32_t groupId = 0; groupId < groupCount; ++groupId) {
        dispatchProgress[groupId].groupId = groupId;
        const auto groupDispatchTasks = FilterDispatchTasksByGroup(plan.dispatchTasks, groupId);
        dispatchProgress[groupId].expectedTasks = static_cast<uint32_t>(groupDispatchTasks.size());
        dispatchProgress[groupId].expectedSourceCount = CountExpectedSourceCount(groupDispatchTasks);
        combineProgress[groupId].groupId = groupId;
    }
    mc2::v4::protocol::ReadyQueue queue;
    mc2::v4::protocol::ReadyQueueSummary queueSummary{};
    std::vector<float> groupScalars(groupCount, 0.0f);
    std::vector<uint8_t> dispatchActual;
    std::vector<float> computeActual;
    std::vector<float> finalActual(fullOracle.output.size(), 0.0f);
    bool dispatchPass = true;
    bool computePass = true;
    bool combinePass = true;
    const double e2eStartMs = mc2::v4::NowMs();

    for (uint32_t tick = 0; ok && tick < timeline.size(); ++tick) {
        for (const auto& step : timeline[tick]) {
            if (step.stage == mc2::v4::OverlapStage::Dispatch) {
                const auto groupDispatchTasks = FilterDispatchTasksByGroup(plan.dispatchTasks, step.groupId);
                if (groupDispatchTasks.empty()) {
                    continue;
                }
                const bool canLaunchGroup = ExchangeStageReady(1, worldSize, rank);
                if (!canLaunchGroup) {
                    ok = false;
                    failStage = "dispatch-group-ready";
                    break;
                }
                const size_t groupDispatchBytes =
                    groupDispatchTasks.size() * sizeof(mc2::v4::protocol::DispatchPullTask);
                failStage = "dispatch-group-copy";
                ok = aclrtMemcpy(dispatchTaskDev,
                                 dispatchTaskCapacityBytes,
                                 groupDispatchTasks.data(),
                                 groupDispatchBytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                dispatchTiling.taskCount = static_cast<uint32_t>(groupDispatchTasks.size());
                dispatchTiling.groupId = step.groupId;
                dispatchTiling.groupWidth = activationSchedule.WidthForStep(tick);
                failStage = "dispatch-group-tiling-copy";
                ok = aclrtMemcpy(dispatchTilingDev,
                                 sizeof(dispatchTiling),
                                 &dispatchTiling,
                                 sizeof(dispatchTiling),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                failStage = "dispatch-group-launch";
                const double t0 = mc2::v4::NowMs();
                mc2::v4::DispatchOnlyLaunchArgs launchArgs;
                launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
                launchArgs.dispatchTasks = dispatchTaskDev;
                launchArgs.tiling = dispatchTilingDev;
                launchArgs.blockDim = dispatchTiling.taskCount == 0 ? 1 : dispatchTiling.taskCount;
                mc2::v4::launchDispatchFFNCombine(launchArgs, runtime.compute_stream);
                ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                auto& progress = dispatchProgress[step.groupId];
                progress.completedTasks = progress.expectedTasks;
                progress.completedSourceCount = progress.expectedSourceCount;
                progress.publishedEpoch = 1;
                ++progress.summaryCount;
                queue.PushDispatchReady(step.groupId, progress.publishedEpoch, progress.expectedSourceCount);
                const auto summary = mc2::v4::dispatch::BuildDispatchSummaryView(progress);
                if (mc2::v4::dispatch::CanLaunchCompute(progress, summary)) {
                    queue.PushComputeReady(step.groupId, progress.publishedEpoch);
                }
                ++queueSummary.dispatchReady;
                mc2::v4::PrintPerfRecord({"full-chain", "dispatch", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                          static_cast<uint64_t>(groupDispatchTasks.size() * fixture.hiddenBytes),
                                          static_cast<uint32_t>(groupDispatchTasks.size()),
                                          "tick=" + std::to_string(tick) +
                                              " group=" + std::to_string(step.groupId) +
                                              " summary_visible_sources=" + std::to_string(summary.visibleSourceCount) +
                                              " quant_sideband_bytes=" + std::to_string(CountDispatchQuantBytes(groupDispatchTasks))});
            } else if (step.stage == mc2::v4::OverlapStage::Compute) {
                auto& progress = dispatchProgress[step.groupId];
                const auto summary = mc2::v4::dispatch::BuildDispatchSummaryView(progress);
                if (!mc2::v4::dispatch::CanLaunchCompute(progress, summary)) {
                    ok = false;
                    failStage = "compute-summary-gate";
                    break;
                }
                computeTiling.groupId = step.groupId;
                computeTiling.groupWidth = activationSchedule.WidthForStep(tick);
                failStage = "compute-group-tiling-copy";
                ok = aclrtMemcpy(computeTilingDev,
                                 sizeof(computeTiling),
                                 &computeTiling,
                                 sizeof(computeTiling),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                failStage = "compute-group-launch";
                const double t0 = mc2::v4::NowMs();
                mc2::v4::ComputeOnlyLaunchArgs launchArgs;
                launchArgs.params = computeParamsDev;
                launchArgs.tiling = computeTilingDev;
                launchArgs.blockDim = 1;
                mc2::v4::launchComputeOnly(launchArgs, runtime.compute_stream);
                ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                ok = CopyDeviceToHostFloats(computeActual, gmm2OutDev, computeOracle.gmm2Out.size());
                if (!ok) {
                    failStage = "compute-group-copy-back";
                    break;
                }
                computePass = computePass && mc2::v4::CompareFloats(computeActual, computeOracle.gmm2Out, 1e-6f);
                groupScalars[step.groupId] = computeActual.empty() ? 0.0f : computeActual[0];
                queue.PushCombineReady(step.groupId, progress.publishedEpoch);
                ++queueSummary.computeReady;
                mc2::v4::PrintPerfRecord({"full-chain", "compute", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                          static_cast<uint64_t>(computeOracle.gmm2Out.size() * sizeof(float)),
                                          static_cast<uint32_t>(computeOracle.gmm2Out.size()),
                                          "tick=" + std::to_string(tick) +
                                              " group=" + std::to_string(step.groupId) +
                                              " schedule_width=" + std::to_string(activationSchedule.WidthForStep(tick)) +
                                              " schedule_tag=" + scheduleTag});
            } else {
                const auto groupCombineTasks = FilterCombineTasksByGroup(plan.combineTasks, step.groupId);
                if (groupCombineTasks.empty()) {
                    continue;
                }
                const auto combinePartialTransport = BuildCombineTransportPayloadForTasks(
                    groupCombineTasks, plan.combineTasks.size(), groupScalars[step.groupId]);
                failStage = "combine-group-partial-copy";
                ok = aclrtMemcpy(combinePartialDev,
                                 combinePartialCapacityBytes,
                                 combinePartialTransport.data(),
                                 combinePartialTransport.size() * sizeof(float),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                const size_t groupCombineTaskBytes =
                    groupCombineTasks.size() * sizeof(mc2::v4::protocol::CombinePushTask);
                failStage = "combine-group-task-copy";
                ok = aclrtMemcpy(combineTaskDev,
                                 combineTaskCapacityBytes,
                                 groupCombineTasks.data(),
                                 groupCombineTaskBytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                combineParams.taskCount = static_cast<uint32_t>(groupCombineTasks.size());
                combineParams.phase = 1;
                failStage = "combine-group-params-copy";
                ok = aclrtMemcpy(combineParamsDev,
                                 sizeof(combineParams),
                                 &combineParams,
                                 sizeof(combineParams),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                combineTiling.groupId = step.groupId;
                combineTiling.groupWidth = activationSchedule.WidthForStep(tick);
                failStage = "combine-group-tiling-copy";
                ok = aclrtMemcpy(combineTilingDev,
                                 sizeof(combineTiling),
                                 &combineTiling,
                                 sizeof(combineTiling),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                const bool canLaunchCombine = ExchangeStageReady(1, worldSize, rank);
                if (!canLaunchCombine) {
                    ok = false;
                    failStage = "combine-group-ready";
                    break;
                }
                failStage = "combine-group-barrier";
                ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
                     aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                failStage = "combine-group-launch";
                const double t0 = mc2::v4::NowMs();
                mc2::v4::CombineOnlyLaunchArgs launchArgs;
                launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
                launchArgs.params = combineParamsDev;
                launchArgs.tiling = combineTilingDev;
                launchArgs.blockDim = combineParams.taskCount == 0 ? 1 : combineParams.taskCount;
                mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
                ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
                     aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    failStage = "combine-group-post-barrier";
                    break;
                }
                combineParams.phase = 2;
                failStage = "combine-group-restore-params-copy";
                ok = aclrtMemcpy(combineParamsDev,
                                 sizeof(combineParams),
                                 &combineParams,
                                 sizeof(combineParams),
                                 ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
                if (!ok) {
                    break;
                }
                launchArgs.blockDim = 1;
                mc2::v4::launchCombineOnly(launchArgs, runtime.compute_stream);
                ok = aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
                if (!ok) {
                    failStage = "combine-group-restore-launch";
                    break;
                }
                ok = CopyDeviceToHostFloats(finalActual, restoredRowsDev, fullOracle.output.size());
                if (!ok) {
                    failStage = "combine-group-copy-back";
                    break;
                }
                auto& progress = combineProgress[step.groupId];
                progress.tileId = static_cast<uint32_t>(groupCombineTasks.size());
                progress.completedTasks = static_cast<uint32_t>(groupCombineTasks.size());
                progress.publishedEpoch = 1;
                ++queueSummary.combineReady;
                mc2::v4::PrintPerfRecord({"full-chain", "combine", static_cast<uint32_t>(rank), mc2::v4::NowMs() - t0,
                                          static_cast<uint64_t>(combinePartialTransport.size() * sizeof(float) +
                                                                fullOracle.output.size() * sizeof(float)),
                                          static_cast<uint32_t>(groupCombineTasks.size() + fullOracle.output.size()),
                                          "tick=" + std::to_string(tick) +
                                              " group=" + std::to_string(step.groupId) +
                                              " restored_rows=" + std::to_string(fullOracle.output.size())});
            }
        }
    }

    if (ok) {
        failStage = "dispatch-final-copy";
        ok = CopyDeviceToHostRegion(dispatchActual,
                                    runtime.hccl.WindowIn(static_cast<uint32_t>(rank)),
                                    runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                    fullOracle.dispatchBytes.size());
    }
    if (ok) {
        dispatchPass = mc2::v4::CompareBytes(dispatchActual, fullOracle.dispatchBytes);
        combinePass = mc2::v4::CompareFloats(finalActual, fullOracle.output, 1e-6f);
    }

    const double e2eElapsedMs = mc2::v4::NowMs() - e2eStartMs;
    const uint64_t e2eBytes = static_cast<uint64_t>(fullOracle.dispatchBytes.size() +
                                                   computeOracle.gmm2Out.size() * sizeof(float) +
                                                   plan.combineTasks.size() * mc2::v4::protocol::kCombineTransportBytes +
                                                   fullOracle.output.size() * sizeof(float));
    const uint32_t e2eItems = static_cast<uint32_t>(plan.dispatchTasks.size() +
                                                   computeOracle.gmm2Out.size() +
                                                   plan.combineTasks.size() +
                                                   fullOracle.output.size());
    pass = ok && dispatchPass && computePass && combinePass;
    mc2::v4::PrintPerfRecord({"full-chain", "e2e", static_cast<uint32_t>(rank), e2eElapsedMs,
                              e2eBytes,
                              e2eItems,
                              "schedule_tag=" + scheduleTag +
                                  " overlap_group_count=" + std::to_string(groupCount) +
                                  " quant_sideband_bytes=" + std::to_string(quantSidebandBytes) +
                                  " shape_class=" + shapeClass +
                                  " dispatch_ready=" + std::to_string(queueSummary.dispatchReady) +
                                  " compute_ready=" + std::to_string(queueSummary.computeReady) +
                                  " combine_ready=" + std::to_string(queueSummary.combineReady)});
    mc2::v4::PrintPerfRecord({"full-chain", "overlap-summary", static_cast<uint32_t>(rank), 0.0,
                              e2eBytes,
                              e2eItems,
                              "schedule_tag=" + scheduleTag +
                                  " overlap_group_count=" + std::to_string(groupCount) +
                                  " quant_sideband_bytes=" + std::to_string(quantSidebandBytes) +
                                  " shape_class=" + shapeClass +
                                  " dispatch_ready=" + std::to_string(queueSummary.dispatchReady) +
                                  " compute_ready=" + std::to_string(queueSummary.computeReady) +
                                  " combine_ready=" + std::to_string(queueSummary.combineReady)});
    if (!ok) {
        std::cerr << "[rank " << rank << "] full-chain overlap failed at stage: " << failStage << '\n';
    }
    std::cout << "[rank " << rank << "] full-chain " << (pass ? "PASS" : "FAIL") << '\n';

    if (combineTilingDev != nullptr) {
        aclrtFree(combineTilingDev);
    }
    if (combineParamsDev != nullptr) {
        aclrtFree(combineParamsDev);
    }
    if (rowRangesDev != nullptr) {
        aclrtFree(rowRangesDev);
    }
    if (combineProbDev != nullptr) {
        aclrtFree(combineProbDev);
    }
    if (combineTaskDev != nullptr) {
        aclrtFree(combineTaskDev);
    }
    if (restoredRowsDev != nullptr) {
        aclrtFree(restoredRowsDev);
    }
    if (combinePartialDev != nullptr) {
        aclrtFree(combinePartialDev);
    }
    if (computeTilingDev != nullptr) {
        aclrtFree(computeTilingDev);
    }
    if (computeParamsDev != nullptr) {
        aclrtFree(computeParamsDev);
    }
    if (gmm2OutDev != nullptr) {
        aclrtFree(gmm2OutDev);
    }
    if (weight2Dev != nullptr) {
        aclrtFree(weight2Dev);
    }
    if (swigluOutDev != nullptr) {
        aclrtFree(swigluOutDev);
    }
    if (gmm1OutDev != nullptr) {
        aclrtFree(gmm1OutDev);
    }
    if (weight1Dev != nullptr) {
        aclrtFree(weight1Dev);
    }
    if (scale2Dev != nullptr) {
        aclrtFree(scale2Dev);
    }
    if (scale1Dev != nullptr) {
        aclrtFree(scale1Dev);
    }
    if (quantInputDev != nullptr) {
        aclrtFree(quantInputDev);
    }
    if (inputDev != nullptr) {
        aclrtFree(inputDev);
    }
    if (dispatchTilingDev != nullptr) {
        aclrtFree(dispatchTilingDev);
    }
    if (dispatchTaskDev != nullptr) {
        aclrtFree(dispatchTaskDev);
    }
    mc2::v4::DestroyStandaloneRankRuntime(runtime);
    if (aclReady) {
        aclrtResetDevice(rank);
        aclFinalize();
    }
    CommMpiFinalize();
    return pass ? 0 : 1;
}

int RunDispatchOnly(int argc, char** argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank = CommMpiRank();
    const int worldSize = CommMpiSize();
    bool aclReady = false;
    bool pass = false;
    const char* failStage = "world-size-check";
    bool ok = worldSize == 2;
    if (!ok) {
        std::cerr << "dispatch-only requires worldSize=2, got " << worldSize << '\n';
    }

    mc2::v4::StandaloneRankRuntime runtime{};
    void* taskDev = nullptr;
    void* tilingDev = nullptr;
    HcclRootInfo rootInfo{};

    if (ok) {
        failStage = "acl-init";
        ok = aclInit(nullptr) == ACL_SUCCESS;
        aclReady = ok;
    }
    if (ok) {
        failStage = "set-device";
        ok = aclrtSetDevice(rank) == ACL_SUCCESS;
    }
    if (rank == 0 && ok) {
        failStage = "hccl-root-info";
        ok = HcclGetRootInfo(&rootInfo) == HCCL_SUCCESS;
    }
    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);

    mc2::v4::ModeConfig mode;
    mode.mode = "dispatch-only";
    mode.localRank = static_cast<uint32_t>(rank);
    mode.worldSize = static_cast<uint32_t>(worldSize);

    if (ok) {
        failStage = "runtime-init";
        ok = mc2::v4::InitStandaloneRankRuntime(runtime, mode, rootInfo);
    }

    const auto fixture = mc2::v4::BuildTwoRankRoutingFixtureForRank1();
    const auto plan = mc2::v4::routing::BuildRoutePlanForRank(fixture, static_cast<uint32_t>(rank));
    const auto publication = mc2::v4::BuildDispatchPublicationForRank(fixture, static_cast<uint32_t>(rank));
    const auto oracle = mc2::v4::BuildHostDispatchOracle(fixture, static_cast<uint32_t>(rank), plan);

    std::vector<uint8_t> zeroCompute(runtime.layout.computeBytes, 0);
    std::vector<int32_t> signalWords(runtime.layout.signalBytes / sizeof(int32_t), 0);
    for (const auto& task : plan.dispatchTasks) {
        const uint32_t readyIndex =
            mc2::v4::protocol::DispatchReadyIndex(task.srcRank, task.srcRowBegin);
        if (readyIndex < signalWords.size()) {
            signalWords[readyIndex] = static_cast<int32_t>(task.readyEpoch);
        }
    }

    if (ok) {
        failStage = "publication-seed";
        void* localWindow = runtime.hccl.WindowIn(static_cast<uint32_t>(rank));
        ok = localWindow != nullptr;
        if (ok) {
            ok = CopyHostToDeviceRegion(localWindow,
                                        runtime.hccl.host_remote_window_ctx.dispatchRegionOffset,
                                        publication.data(),
                                        publication.size()) &&
                 CopyHostToDeviceRegion(localWindow,
                                        runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                        zeroCompute.data(),
                                        zeroCompute.size()) &&
                 CopyHostToDeviceRegion(localWindow,
                                        runtime.hccl.host_remote_window_ctx.signalRegionOffset,
                                        signalWords.data(),
                                        signalWords.size() * sizeof(int32_t));
        }
    }

    if (ok) {
        failStage = "task-buffer-copy";
        const size_t taskBytes = plan.dispatchTasks.size() * sizeof(mc2::v4::protocol::DispatchPullTask);
        ok = AllocateAndCopyDeviceBuffer(taskDev, plan.dispatchTasks.data(), taskBytes);
    }

    mc2::v4::StandaloneKernelTilingData tiling{};
    tiling.mode = static_cast<uint32_t>(mc2::v4::KernelMode::DispatchOnly);
    tiling.taskCount = static_cast<uint32_t>(plan.dispatchTasks.size());
    tiling.hiddenBytes = fixture.hiddenBytes;
    tiling.outputBytes = fixture.outputBytes;
    if (ok) {
        failStage = "tiling-buffer-copy";
        ok = AllocateAndCopyDeviceBuffer(tilingDev, &tiling, sizeof(tiling));
    }

    const bool canLaunch = ExchangeStageReady(ok ? 1 : 0, worldSize, rank);
    if (ok && canLaunch) {
        failStage = "prelaunch-hccl-barrier";
        ok = HcclBarrier(runtime.hccl.comm, runtime.compute_stream) == HCCL_SUCCESS &&
             aclrtSynchronizeStream(runtime.compute_stream) == ACL_SUCCESS;
    }
    if (ok && canLaunch) {
        failStage = "kernel-launch";
        mc2::v4::DispatchOnlyLaunchArgs launchArgs;
        launchArgs.remoteWindow = runtime.hccl.RemoteWindowContextPtr();
        launchArgs.dispatchTasks = taskDev;
        launchArgs.tiling = tilingDev;
        launchArgs.blockDim = tiling.taskCount == 0 ? 1 : tiling.taskCount;
        mc2::v4::launchDispatchFFNCombine(launchArgs, runtime.compute_stream);
        const aclError syncErr = aclrtSynchronizeStream(runtime.compute_stream);
        if (syncErr != ACL_SUCCESS) {
            std::cerr << "[rank " << rank << "] kernel-launch syncErr=" << syncErr
                      << " lastErr=" << aclrtGetLastError(ACL_RT_THREAD_LEVEL)
                      << " recentErrMsg=" << aclGetRecentErrMsg()
                      << " blockDim=" << launchArgs.blockDim
                      << " taskCount=" << tiling.taskCount
                      << '\n';
        }
        ok = syncErr == ACL_SUCCESS;
    } else {
        ok = false;
    }

    std::vector<uint8_t> actual;
    if (ok) {
        failStage = "copy-back";
        ok = CopyDeviceToHostRegion(actual,
                                    runtime.hccl.WindowIn(static_cast<uint32_t>(rank)),
                                    runtime.hccl.host_remote_window_ctx.computeRegionOffset,
                                    oracle.size());
    }
    if (ok) {
        pass = mc2::v4::CompareBytes(actual, oracle);
        if (!pass) {
            const size_t mismatch = FirstMismatch(actual, oracle);
            std::cerr << "[rank " << rank << "] dispatch-only mismatch at byte " << mismatch;
            const size_t sample = actual.size() < 8 ? actual.size() : 8;
            std::cerr << " actual=";
            for (size_t i = 0; i < sample; ++i) {
                std::cerr << static_cast<int>(actual[i]) << (i + 1 == sample ? "" : ",");
            }
            std::cerr << " oracle=";
            for (size_t i = 0; i < sample; ++i) {
                std::cerr << static_cast<int>(oracle[i]) << (i + 1 == sample ? "" : ",");
            }
            std::cerr << '\n';
        }
    }

    if (!ok) {
        std::cerr << "[rank " << rank << "] dispatch-only failed at stage: " << failStage << '\n';
    }
    std::cout << "[rank " << rank << "] dispatch-only " << ((ok && pass) ? "PASS" : "FAIL") << '\n';

    if (tilingDev != nullptr) {
        aclrtFree(tilingDev);
    }
    if (taskDev != nullptr) {
        aclrtFree(taskDev);
    }
    mc2::v4::DestroyStandaloneRankRuntime(runtime);
    if (aclReady) {
        aclrtResetDevice(rank);
        aclFinalize();
    }
    CommMpiFinalize();
    return (ok && pass) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const auto mode = ParseMode(argc, argv);

    if (mode.mode == "signal-roundtrip") {
        return RunSignalRoundtrip();
    }

    if (mode.mode == "metadata-only") {
        const bool pass = VerifyMetadataGolden();
        std::cout << (pass ? "PASS\n" : "FAIL\n");
        return pass ? 0 : 1;
    }

    if (mode.mode == "dispatch-only") {
        return RunDispatchOnly(argc, argv);
    }

    if (mode.mode == "compute-only") {
        return RunComputeOnly();
    }

    if (mode.mode == "combine-only") {
        return RunCombineOnly(argc, argv);
    }

    if (mode.mode == "full-chain") {
        return RunFullChainOverlap(argc, argv);
    }

    auto ctx = mc2::v4::BuildSkeletonContext(mode);
    if (mode.mode == "default") {
        return ctx.remote.workspaceBase == 0 ? 1 : 0;
    }

    std::cerr << "mode not implemented yet: " << mode.mode << '\n';
    return 2;
}
