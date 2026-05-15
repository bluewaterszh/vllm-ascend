#include "runtime_context.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;
constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;

struct Mc2InitTilingInner {
    uint32_t version = 0;
    uint32_t mc2HcommCnt = 0;
    uint32_t offset[MAX_CC_TILING_NUM] = {};
    uint8_t debugMode = 0;
    uint8_t preparePosition = 0;
    uint16_t queueNum = 0;
    uint16_t commBlockNum = 0;
    uint8_t devType = 0;
    char reserved[17] = {};
};

struct Mc2cCTilingInner {
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

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init{};
    Mc2cCTilingInner inner{};
};

namespace hccl_compat {
struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};
} // namespace hccl_compat

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool ReadRingParams(int rank_id,
                    uint8_t *raw_ctx,
                    hccl_compat::HcclOpResParamHead &head,
                    std::vector<hccl_compat::RemoteResPtr> &remote_res_arr)
{
    const size_t head_offset = offsetof(hccl_compat::HcclOpResParam, localUsrRankId);
    if (aclrtMemcpy(&head, sizeof(head), raw_ctx + head_offset, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }

    if (head.rankSize == 0 || head.rankSize > HcclDeviceContext::HCCL_MAX_RANK_NUM) {
        return false;
    }

    const size_t remote_res_offset = offsetof(hccl_compat::HcclOpResParam, remoteRes);
    const size_t remote_res_bytes = static_cast<size_t>(head.rankSize) * sizeof(hccl_compat::RemoteResPtr);
    remote_res_arr.resize(head.rankSize);
    if (aclrtMemcpy(remote_res_arr.data(), remote_res_bytes, raw_ctx + remote_res_offset, remote_res_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

bool BuildRingHostCtx(StandaloneRankRuntime &runtime,
                      uint8_t *raw_ctx,
                      const hccl_compat::HcclOpResParamHead &head,
                      const std::vector<hccl_compat::RemoteResPtr> &remote_res_arr)
{
    std::memset(&runtime.hccl.host_ctx, 0, sizeof(runtime.hccl.host_ctx));

    uint64_t workspace_fields[2] = {0, 0};
    if (aclrtMemcpy(workspace_fields, sizeof(workspace_fields), raw_ctx, sizeof(workspace_fields),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
        runtime.hccl.host_ctx.workSpace = workspace_fields[0];
        runtime.hccl.host_ctx.workSpaceSize = workspace_fields[1];
    }

    runtime.hccl.host_ctx.rankId = head.localUsrRankId;
    runtime.hccl.host_ctx.rankNum = head.rankSize;
    runtime.hccl.host_ctx.winSize = head.winSize;

    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            runtime.hccl.host_ctx.windowsIn[i] = head.localWindowsIn;
            runtime.hccl.host_ctx.windowsOut[i] = head.localWindowsOut;
            continue;
        }

        const uint64_t dev_ptr = remote_res_arr[i].nextDevicePtr;
        if (dev_ptr == 0) {
            return false;
        }

        hccl_compat::HcclRankRelationResV2 remote_info{};
        if (aclrtMemcpy(&remote_info, sizeof(remote_info), reinterpret_cast<void *>(dev_ptr), sizeof(remote_info),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }

        runtime.hccl.host_ctx.windowsIn[i] = remote_info.windowsIn;
        runtime.hccl.host_ctx.windowsOut[i] = remote_info.windowsOut;
    }
    return true;
}

bool CopyHostCtxToDevice(StandaloneRankRuntime &runtime)
{
    void *new_dev_mem = nullptr;
    if (aclrtMalloc(&new_dev_mem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        new_dev_mem == nullptr) {
        return false;
    }

    if (aclrtMemcpy(new_dev_mem, sizeof(HcclDeviceContext), &runtime.hccl.host_ctx, sizeof(HcclDeviceContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(new_dev_mem);
        return false;
    }

    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(new_dev_mem);
    runtime.hccl.owns_device_ctx = true;
    return true;
}
} // namespace

bool InitStandaloneRankRuntime(StandaloneRankRuntime &runtime, int rank_id, int world_size, const HcclRootInfo &root_info)
{
    runtime.hccl.rank_id = rank_id;
    runtime.hccl.world_size = world_size;
    runtime.hccl.device_id = rank_id;

    if (aclrtSetDevice(runtime.hccl.device_id) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtCreateStream(&runtime.compute_stream) != ACL_SUCCESS) {
        return false;
    }
    if (rtStreamCreate(&runtime.hccl.hccl_stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
        return false;
    }
    if (HcclCommInitRootInfo(static_cast<uint32_t>(world_size), &root_info, static_cast<uint32_t>(rank_id),
                             &runtime.hccl.comm) != HCCL_SUCCESS) {
        return false;
    }

    char group[GROUP_NAME_SIZE] = {};
    if (HcclGetCommName(runtime.hccl.comm, group) != HCCL_SUCCESS) {
        return false;
    }

    uint32_t topo = 0;
    if (HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE) != HCCL_SUCCESS) {
        return false;
    }

    HcclComm comm_handle = nullptr;
    if (HcomGetCommHandleByGroup(group, &comm_handle) != HCCL_SUCCESS) {
        return false;
    }

    Mc2CommConfigV2 tiling{};
    tiling.init.version = 100U;
    tiling.init.mc2HcommCnt = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) -
                                                  reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    std::strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1U);
    std::strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1U);

    void *ctx_ptr = nullptr;
    if (HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr) != HCCL_SUCCESS ||
        ctx_ptr == nullptr) {
        return false;
    }

    if (topo == COMM_TOPO_MESH) {
        return LoadMeshContext(runtime, ctx_ptr);
    }

    auto *raw_ctx = reinterpret_cast<uint8_t *>(ctx_ptr);
    hccl_compat::HcclOpResParamHead head{};
    std::vector<hccl_compat::RemoteResPtr> remote_res_arr;
    if (!ReadRingParams(rank_id, raw_ctx, head, remote_res_arr)) {
        return false;
    }
    if (!BuildRingHostCtx(runtime, raw_ctx, head, remote_res_arr)) {
        return false;
    }
    return CopyHostCtxToDevice(runtime);
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    if (runtime.hccl.owns_device_ctx && runtime.hccl.device_ctx != nullptr) {
        aclrtFree(runtime.hccl.device_ctx);
    }
    runtime.hccl.device_ctx = nullptr;
    runtime.hccl.owns_device_ctx = false;
    runtime.hccl.host_ctx = {};

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
