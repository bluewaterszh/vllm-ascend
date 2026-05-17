#include "runtime_context.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tiling_builder.hpp"

namespace mc2::v4 {

void StandaloneHcclContext::ReleaseRemoteWindowContext() {
    if (owns_remote_window_ctx && remote_window_ctx != nullptr) {
        aclrtFree(remote_window_ctx);
    }
    remote_window_ctx = nullptr;
    owns_remote_window_ctx = false;
}

void StandaloneHcclContext::ResetHostRemoteWindowContext() {
    host_remote_window_ctx = {};
}

void StandaloneHcclContext::SetHostRankInfo(uint32_t rank, uint32_t rankCount, uint64_t windowBytes) {
    host_remote_window_ctx.rank = rank;
    host_remote_window_ctx.rankSize = rankCount;
    host_remote_window_ctx.segmentBytes = windowBytes;
    host_remote_window_ctx.workspaceBytes = windowBytes;
}

void StandaloneHcclContext::SetHostWindow(uint32_t rank, uint64_t windowIn, uint64_t windowOut) {
    host_remote_window_ctx.windowIn[rank] = windowIn;
    host_remote_window_ctx.windowOut[rank] = windowOut;
    if (rank == host_remote_window_ctx.rank) {
        host_remote_window_ctx.workspaceBase = windowIn;
    }
}

void StandaloneHcclContext::SetHostWindowLayout(const WorkspaceLayoutConfig& layout) {
    host_remote_window_ctx.controlRegionOffset = 0;
    host_remote_window_ctx.controlRegionBytes = layout.controlBytes;
    host_remote_window_ctx.dispatchRegionOffset = host_remote_window_ctx.controlRegionOffset + layout.controlBytes;
    host_remote_window_ctx.dispatchRegionBytes = layout.dispatchBytes;
    host_remote_window_ctx.computeRegionOffset = host_remote_window_ctx.dispatchRegionOffset + layout.dispatchBytes;
    host_remote_window_ctx.computeRegionBytes = layout.computeBytes;
    host_remote_window_ctx.combineRegionOffset = host_remote_window_ctx.computeRegionOffset + layout.computeBytes;
    host_remote_window_ctx.combineRegionBytes = layout.combineBytes;
    host_remote_window_ctx.signalRegionOffset = host_remote_window_ctx.combineRegionOffset + layout.combineBytes;
    host_remote_window_ctx.signalRegionBytes = layout.signalBytes;
}

bool StandaloneHcclContext::CopyHostRemoteWindowContextToDevice() {
    ReleaseRemoteWindowContext();
    void* newDevMem = nullptr;
    if (aclrtMalloc(&newDevMem, sizeof(protocol::RemoteWindowContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        newDevMem == nullptr) {
        return false;
    }
    if (aclrtMemcpy(newDevMem,
                    sizeof(protocol::RemoteWindowContext),
                    &host_remote_window_ctx,
                    sizeof(protocol::RemoteWindowContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(newDevMem);
        return false;
    }
    remote_window_ctx = reinterpret_cast<protocol::RemoteWindowContext*>(newDevMem);
    owns_remote_window_ctx = true;
    return true;
}

StandaloneRuntimeContext BuildSkeletonContext(const ModeConfig& mode) {
    StandaloneRuntimeContext ctx;
    ctx.mode = mode;
    ctx.layout = BuildDefaultWorkspaceLayout();
    ctx.launch = BuildDefaultLaunchConfig();
    ctx.remote.rank = mode.localRank;
    ctx.remote.rankSize = mode.worldSize;
    ctx.remote.workspaceBase = 0x100000;
    ctx.remote.workspaceBytes = ctx.layout.TotalBytes();
    ctx.remote.segmentBytes = ctx.layout.TotalBytes();
    ctx.remote.windowIn[mode.localRank] = ctx.remote.workspaceBase;
    ctx.remote.windowOut[mode.localRank] = ctx.remote.workspaceBase;
    ctx.remote.controlRegionOffset = 0;
    ctx.remote.controlRegionBytes = ctx.layout.controlBytes;
    ctx.remote.dispatchRegionOffset = ctx.remote.controlRegionOffset + ctx.layout.controlBytes;
    ctx.remote.dispatchRegionBytes = ctx.layout.dispatchBytes;
    ctx.remote.computeRegionOffset = ctx.remote.dispatchRegionOffset + ctx.layout.dispatchBytes;
    ctx.remote.computeRegionBytes = ctx.layout.computeBytes;
    ctx.remote.combineRegionOffset = ctx.remote.computeRegionOffset + ctx.layout.computeBytes;
    ctx.remote.combineRegionBytes = ctx.layout.combineBytes;
    ctx.remote.signalRegionOffset = ctx.remote.combineRegionOffset + ctx.layout.combineBytes;
    ctx.remote.signalRegionBytes = ctx.layout.signalBytes;
    return ctx;
}

namespace {
namespace pto_hccl_compat {

constexpr uint32_t MAX_CC_TILING_NUM = 8U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64U;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2U;
constexpr uint32_t V4_MAX_RANKS = 64U;

struct CompatRemoteWindowContext {
    uint64_t workspaceBase = 0;
    uint64_t workspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t windowBytes = 0;
    uint64_t windowIn[V4_MAX_RANKS] = {};
    uint64_t windowOut[V4_MAX_RANKS] = {};
};

struct CommResourceInitV2 {
    uint32_t version = 0;
    uint32_t hcommCount = 0;
    uint32_t offset[MAX_CC_TILING_NUM] = {};
    uint8_t debugMode = 0;
    uint8_t preparePosition = 0;
    uint16_t queueNum = 0;
    uint16_t commBlockNum = 0;
    uint8_t devType = 0;
    char reserved[17] = {};
};

struct CommResourceConfigV2 {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    uint8_t version = 0;
    char reserved[9] = {};
    uint8_t commEngine = 0;
    uint8_t srcDataType = 0;
    uint8_t dstDataType = 0;
    char groupName[GROUP_NAME_SIZE] = {};
    char algConfig[ALG_CONFIG_SIZE] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};

struct CommResourceTilingV2 {
    CommResourceInitV2 init{};
    CommResourceConfigV2 inner{};
};

struct HcclSignalInfo {
    uint64_t resId = 0;
    uint64_t addr = 0;
    uint32_t devId = 0;
    uint32_t tsId = 0;
    uint32_t rankId = 0;
    uint32_t flag = 0;
};

struct HcclStreamInfo {
    int32_t streamIds = 0;
    uint32_t sqIds = 0;
    uint32_t cqIds = 0;
    uint32_t logicCqids = 0;
};

struct ListCommon {
    uint64_t nextHost = 0;
    uint64_t preHost = 0;
    uint64_t nextDevice = 0;
    uint64_t preDevice = 0;
};

struct LocalResInfoV2 {
    uint32_t streamNum = 0;
    uint32_t signalNum = 0;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM] = {};
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM] = {};
    HcclStreamInfo mainStreamInfo{};
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM] = {};
    ListCommon nextTagRes{};
};

struct AlgoTopoInfo {
    uint32_t userRank = 0;
    uint32_t userRankSize = 0;
    int32_t deviceLogicId = 0;
    bool isSingleMeshAggregation = false;
    uint32_t deviceNumPerAggregation = 0;
    uint32_t superPodNum = 0;
    uint32_t devicePhyId = 0;
    uint32_t topoType = 0;
    uint32_t deviceType = 0;
    uint32_t serverNum = 0;
    uint32_t meshAggregationRankSize = 0;
    uint32_t multiModuleDiffDeviceNumMode = 0;
    uint32_t multiSuperPodDiffServerNumMode = 0;
    uint32_t realUserRank = 0;
    bool isDiffDeviceModule = false;
    bool isDiffDeviceType = false;
    uint32_t gcdDeviceNumPerAggregation = 0;
    uint32_t moduleNum = 0;
    uint32_t isUsedRdmaRankPairNum = 0;
    uint64_t isUsedRdmaRankPair = 0;
    uint32_t pairLinkCounterNum = 0;
    uint64_t pairLinkCounter = 0;
    uint32_t nicNum = 0;
    uint64_t nicList = 0;
    uint64_t complanRankLength = 0;
    uint64_t complanRank = 0;
    uint64_t bridgeRankNum = 0;
    uint64_t bridgeRank = 0;
    uint64_t serverAndsuperPodRankLength = 0;
    uint64_t serverAndsuperPodRank = 0;
};

struct HcclOpConfig {
    uint8_t deterministic = 0;
    uint8_t retryEnable = 0;
    uint8_t highPerfEnable = 0;
    uint8_t padding[5] = {};
    uint8_t linkTimeOut[8] = {};
    uint64_t notifyWaitTime = 0;
    uint32_t retryHoldTime = 0;
    uint32_t retryIntervalTime = 0;
    bool interXLinkDisable = false;
    uint32_t floatOverflowMode = 0;
    uint32_t multiQpThreshold = 0;
};

struct RemoteResPtr {
    uint64_t nextHostPtr = 0;
    uint64_t nextDevicePtr = 0;
};

struct HcclWorkspaceInfo {
    uint64_t workspace = 0;
    uint64_t workspaceSize = 0;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId = 0;
    uint32_t remoteWorldRank = 0;
    uint64_t windowsIn = 0;
    uint64_t windowsOut = 0;
    uint64_t windowsExp = 0;
    ListCommon nextTagRes{};
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
};

struct HcclOpResParam {
    HcclWorkspaceInfo workSpaceInfo{};
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
    uint32_t rWinStart = 0;
    uint32_t rWinOffset = 0;
    uint64_t version = 0;
    LocalResInfoV2 localRes{};
    AlgoTopoInfo topoInfo{};
    HcclOpConfig config{};
    uint64_t hostStateInfo = 0;
    uint64_t aicpuStateInfo = 0;
    uint64_t lockAddr = 0;
    uint32_t rsv[16] = {};
    uint32_t notifysize = 0;
    uint32_t remoteResNum = 0;
    RemoteResPtr remoteRes[1] = {};
};

}  // namespace pto_hccl_compat

constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;

bool LoadMeshRemoteWindowContext(StandaloneHcclContext& hccl, void* ctxPtr) {
    pto_hccl_compat::CompatRemoteWindowContext compat{};
    if (aclrtMemcpy(&compat,
                    sizeof(compat),
                    ctxPtr,
                    sizeof(compat),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    hccl.ResetHostRemoteWindowContext();
    hccl.SetHostRankInfo(compat.rank, compat.rankSize, compat.windowBytes);
    hccl.host_remote_window_ctx.workspaceBase = compat.workspaceBase;
    hccl.host_remote_window_ctx.workspaceBytes = compat.workspaceBytes;
    for (uint32_t i = 0; i < compat.rankSize && i < protocol::kMaxRemoteRanks; ++i) {
        hccl.SetHostWindow(i, compat.windowIn[i], compat.windowOut[i]);
    }
    return true;
}

bool ReadRingParams(uint8_t* rawCtx,
                    pto_hccl_compat::HcclOpResParamHead& head,
                    std::vector<pto_hccl_compat::RemoteResPtr>& remoteResArr) {
    const size_t headOffset = offsetof(pto_hccl_compat::HcclOpResParam, localUsrRankId);
    if (aclrtMemcpy(&head, sizeof(head), rawCtx + headOffset, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    if (head.rankSize == 0 || head.rankSize > protocol::kMaxRemoteRanks) {
        return false;
    }
    const size_t remoteResOffset = offsetof(pto_hccl_compat::HcclOpResParam, remoteRes);
    const size_t remoteResBytes = static_cast<size_t>(head.rankSize) * sizeof(pto_hccl_compat::RemoteResPtr);
    remoteResArr.resize(head.rankSize);
    if (aclrtMemcpy(remoteResArr.data(),
                    remoteResBytes,
                    rawCtx + remoteResOffset,
                    remoteResBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

bool BuildRingHostRemoteWindowContext(StandaloneHcclContext& hccl,
                                      uint8_t* rawCtx,
                                      const pto_hccl_compat::HcclOpResParamHead& head,
                                      const std::vector<pto_hccl_compat::RemoteResPtr>& remoteResArr) {
    hccl.ResetHostRemoteWindowContext();
    uint64_t workspaceFields[2] = {0, 0};
    if (aclrtMemcpy(workspaceFields,
                    sizeof(workspaceFields),
                    rawCtx,
                    sizeof(workspaceFields),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
        hccl.host_remote_window_ctx.workspaceBase = workspaceFields[0];
        hccl.host_remote_window_ctx.workspaceBytes = workspaceFields[1];
    }
    hccl.SetHostRankInfo(head.localUsrRankId, head.rankSize, head.winSize);
    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            hccl.SetHostWindow(i, head.localWindowsIn, head.localWindowsOut);
            continue;
        }
        const uint64_t devPtr = remoteResArr[i].nextDevicePtr;
        if (devPtr == 0) {
            return false;
        }
        pto_hccl_compat::HcclRankRelationResV2 remoteInfo{};
        if (aclrtMemcpy(&remoteInfo,
                        sizeof(remoteInfo),
                        reinterpret_cast<void*>(devPtr),
                        sizeof(remoteInfo),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }
        hccl.SetHostWindow(i, remoteInfo.windowsIn, remoteInfo.windowsOut);
    }
    return true;
}

}  // namespace

bool InitStandaloneRankRuntime(StandaloneRankRuntime& runtime, const ModeConfig& mode, const HcclRootInfo& rootInfo) {
    runtime.mode = mode;
    runtime.layout = BuildDefaultWorkspaceLayout();
    runtime.launch = BuildDefaultLaunchConfig();
    runtime.launch.m = mode.m;
    runtime.launch.k = mode.k;
    runtime.launch.n = mode.n;
    runtime.launch.topk = mode.topk;
    runtime.launch.expertsPerRank = mode.expertsPerRank;
    runtime.launch.worldSize = mode.worldSize;
    runtime.launch.numExpertGroups = mode.expertsPerRank;
    runtime.launch.maxOutputSize = mode.maxOutputSize;
    runtime.hccl.rank_id = static_cast<int>(mode.localRank);
    runtime.hccl.world_size = static_cast<int>(mode.worldSize);
    runtime.hccl.device_id = static_cast<int>(mode.localRank);

    if (aclrtSetDevice(runtime.hccl.device_id) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtCreateStream(&runtime.compute_stream) != ACL_SUCCESS) {
        return false;
    }
    if (rtStreamCreate(&runtime.hccl.hccl_stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
        return false;
    }
    if (HcclCommInitRootInfo(mode.worldSize, &rootInfo, mode.localRank, &runtime.hccl.comm) != HCCL_SUCCESS) {
        return false;
    }

    char group[pto_hccl_compat::GROUP_NAME_SIZE] = {};
    if (HcclGetCommName(runtime.hccl.comm, group) != HCCL_SUCCESS) {
        return false;
    }

    uint32_t topo = 0;
    if (HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE) != HCCL_SUCCESS) {
        return false;
    }

    HcclComm commHandle = nullptr;
    if (HcomGetCommHandleByGroup(group, &commHandle) != HCCL_SUCCESS) {
        return false;
    }

    pto_hccl_compat::CommResourceTilingV2 tiling{};
    tiling.init.version = 100U;
    tiling.init.hcommCount = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) -
                                                  reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    std::strncpy(tiling.inner.groupName, group, pto_hccl_compat::GROUP_NAME_SIZE - 1U);
    std::strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", pto_hccl_compat::ALG_CONFIG_SIZE - 1U);

    void* ctxPtr = nullptr;
    if (HcclAllocComResourceByTiling(commHandle, runtime.hccl.hccl_stream, &tiling, &ctxPtr) != HCCL_SUCCESS ||
        ctxPtr == nullptr) {
        return false;
    }

    bool ok = false;
    if (topo == COMM_TOPO_MESH) {
        ok = LoadMeshRemoteWindowContext(runtime.hccl, ctxPtr);
    } else {
        auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);
        pto_hccl_compat::HcclOpResParamHead head{};
        std::vector<pto_hccl_compat::RemoteResPtr> remoteResArr;
        ok = ReadRingParams(rawCtx, head, remoteResArr) &&
             BuildRingHostRemoteWindowContext(runtime.hccl, rawCtx, head, remoteResArr);
    }
    if (!ok) {
        return false;
    }

    runtime.hccl.SetHostWindowLayout(runtime.layout);
    return runtime.hccl.CopyHostRemoteWindowContextToDevice();
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime& runtime) {
    runtime.hccl.ReleaseRemoteWindowContext();
    runtime.hccl.ResetHostRemoteWindowContext();
    if (runtime.hccl.comm != nullptr) {
        HcclCommDestroy(runtime.hccl.comm);
        runtime.hccl.comm = nullptr;
    }
    if (runtime.hccl.hccl_stream != nullptr) {
        rtStreamDestroy(runtime.hccl.hccl_stream);
        runtime.hccl.hccl_stream = nullptr;
    }
    if (runtime.compute_stream != nullptr) {
        aclrtDestroyStream(runtime.compute_stream);
        runtime.compute_stream = nullptr;
    }
}

}  // namespace mc2::v4
