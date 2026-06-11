#include "runtime_context.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64U;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2U;
constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;

namespace pto_hccl_compat {

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

} // namespace pto_hccl_compat

bool VerboseLogEnabled()
{
    const char *value = std::getenv("DISPATCH_FFN_COMBINE_V2_VERBOSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool TraceEnabled()
{
    const char *value = std::getenv("DISPATCH_FFN_COMBINE_V2_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void TraceLog(int rank_id, const std::string &message)
{
    if (!TraceEnabled()) {
        return;
    }

    std::ostringstream os;
    os << "[V2TRACE] rank=" << rank_id << " pid=" << static_cast<long long>(getpid()) << ' ' << message;
    const std::string line = os.str();
    std::cerr << line << std::endl;

    const char *trace_dir = std::getenv("DISPATCH_FFN_COMBINE_V2_TRACE_FILE_DIR");
    if (trace_dir != nullptr && trace_dir[0] != '\0') {
        std::ofstream trace_file(std::string(trace_dir) + "/rank" + std::to_string(rank_id) + ".log",
                                 std::ios::app);
        if (trace_file) {
            trace_file << line << '\n';
        }
    }
}

std::string RecentAclErrorText()
{
    const char *recent = aclGetRecentErrMsg();
    return recent != nullptr && recent[0] != '\0' ? std::string(recent) : std::string();
}

void TraceFailure(int rank_id, const std::string &message)
{
    std::string full_message = message;
    const std::string recent = RecentAclErrorText();
    if (!recent.empty()) {
        full_message += ", recent=" + recent;
    }
    std::cerr << "rank=" << rank_id << " runtime init failed: " << full_message << std::endl;
    TraceLog(rank_id, "runtime init failed: " + full_message);
}

void BuildCommResourceTiling(const char *group, pto_hccl_compat::CommResourceTilingV2 &tiling)
{
    tiling = pto_hccl_compat::CommResourceTilingV2{};
    tiling.init.version = 100U;
    tiling.init.hcommCount = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) -
                                                  reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    std::strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1U);
    std::strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1U);
}

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool ReadRingParams(int rank_id,
                    uint8_t *raw_ctx,
                    pto_hccl_compat::HcclOpResParamHead &head,
                    std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr)
{
    const size_t head_offset = offsetof(pto_hccl_compat::HcclOpResParam, localUsrRankId);
    if (aclrtMemcpy(&head, sizeof(head), raw_ctx + head_offset, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        TraceLog(rank_id, "ReadRingParams failed to read HcclOpResParam head");
        return false;
    }

    if (head.rankSize == 0 || head.rankSize > HCCL_STANDALONE_MAX_RANK_NUM) {
        TraceLog(rank_id, "ReadRingParams invalid rankSize=" + std::to_string(head.rankSize));
        return false;
    }

    const size_t remote_res_offset = offsetof(pto_hccl_compat::HcclOpResParam, remoteRes);
    const size_t remote_res_bytes = static_cast<size_t>(head.rankSize) * sizeof(pto_hccl_compat::RemoteResPtr);
    remote_res_arr.resize(head.rankSize);
    if (aclrtMemcpy(remote_res_arr.data(), remote_res_bytes, raw_ctx + remote_res_offset, remote_res_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        TraceLog(rank_id, "ReadRingParams failed to read remoteRes");
        return false;
    }
    return true;
}

bool BuildRingHostContext(int rank_id,
                          uint8_t *raw_ctx,
                          const pto_hccl_compat::HcclOpResParamHead &head,
                          const std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr,
                          HcclDeviceContext &host_ctx)
{
    host_ctx = HcclDeviceContext{};

    uint64_t workspace_fields[2] = {0, 0};
    if (aclrtMemcpy(workspace_fields, sizeof(workspace_fields), raw_ctx, sizeof(workspace_fields),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
        host_ctx.workSpace = workspace_fields[0];
        host_ctx.workSpaceSize = workspace_fields[1];
    }

    host_ctx.rankId = head.localUsrRankId;
    host_ctx.rankNum = head.rankSize;
    host_ctx.winSize = head.winSize;

    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            host_ctx.windowsIn[i] = head.localWindowsIn;
            host_ctx.windowsOut[i] = head.localWindowsOut;
            continue;
        }

        const uint64_t dev_ptr = remote_res_arr[i].nextDevicePtr;
        if (dev_ptr == 0) {
            TraceLog(rank_id, "BuildRingHostContext remoteRes[" + std::to_string(i) + "].nextDevicePtr is null");
            return false;
        }

        pto_hccl_compat::HcclRankRelationResV2 remote_info{};
        if (aclrtMemcpy(&remote_info, sizeof(remote_info), reinterpret_cast<void *>(dev_ptr), sizeof(remote_info),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            TraceLog(rank_id, "BuildRingHostContext failed to read remote rank " + std::to_string(i));
            return false;
        }
        host_ctx.windowsIn[i] = remote_info.windowsIn;
        host_ctx.windowsOut[i] = remote_info.windowsOut;
    }
    return true;
}

bool CopyHostContextToDevice(StandaloneRankRuntime &runtime)
{
    void *device_ctx = nullptr;
    if (aclrtMalloc(&device_ctx, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        device_ctx == nullptr) {
        return false;
    }
    if (aclrtMemcpy(device_ctx, sizeof(HcclDeviceContext), &runtime.hccl.host_ctx, sizeof(HcclDeviceContext),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(device_ctx);
        return false;
    }
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(device_ctx);
    runtime.hccl.owns_device_ctx = true;
    return true;
}

bool LoadRingContext(StandaloneRankRuntime &runtime, int rank_id, void *ctx_ptr)
{
    auto *raw_ctx = reinterpret_cast<uint8_t *>(ctx_ptr);
    pto_hccl_compat::HcclOpResParamHead head{};
    std::vector<pto_hccl_compat::RemoteResPtr> remote_res_arr;
    if (!ReadRingParams(rank_id, raw_ctx, head, remote_res_arr)) {
        return false;
    }
    TraceLog(rank_id, "ReadRingParams done rankId=" + std::to_string(head.localUsrRankId) +
                          " rankSize=" + std::to_string(head.rankSize) +
                          " winSize=" + std::to_string(head.winSize));
    if (!BuildRingHostContext(rank_id, raw_ctx, head, remote_res_arr, runtime.hccl.host_ctx)) {
        return false;
    }
    return CopyHostContextToDevice(runtime);
}

void DumpMeshContextDebug(int rank_id, const HcclDeviceContext *device_ctx)
{
    if (!VerboseLogEnabled() && !TraceEnabled()) {
        return;
    }

    HcclDeviceContext debug_ctx{};
    if (aclrtMemcpy(&debug_ctx, sizeof(debug_ctx), device_ctx, sizeof(debug_ctx), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        std::cerr << "rank=" << rank_id << " failed to copy HCCL debug context" << std::endl;
        return;
    }

    std::cerr << "rank=" << rank_id
              << " hccl_ctx rankId=" << debug_ctx.rankId
              << " rankNum=" << debug_ctx.rankNum
              << " winSize=" << debug_ctx.winSize
              << " workSpace=0x" << std::hex << debug_ctx.workSpace
              << " workSpaceSize=" << std::dec << debug_ctx.workSpaceSize
              << std::endl;
    const uint32_t slots = std::min(debug_ctx.rankNum, HCCL_STANDALONE_MAX_RANK_NUM);
    for (uint32_t i = 0; i < slots; ++i) {
        std::cerr << "rank=" << rank_id
                  << " hccl_ctx windowsIn[" << i << "]=0x" << std::hex
                  << debug_ctx.windowsIn[i]
                  << " windowsOut[" << i << "]=0x"
                  << debug_ctx.windowsOut[i]
                  << std::dec << std::endl;
    }
}

} // namespace

bool InitStandaloneRankRuntime(StandaloneRankRuntime &runtime, int rank_id, int world_size, const HcclRootInfo &root_info)
{
    TraceLog(rank_id, "InitStandaloneRankRuntime enter world_size=" + std::to_string(world_size));
    runtime.hccl.rank_id = rank_id;
    runtime.hccl.world_size = world_size;
    runtime.hccl.device_id = rank_id;

    TraceLog(rank_id, "runtime aclrtSetDevice begin device=" + std::to_string(runtime.hccl.device_id));
    aclError acl_set_device_ret = aclrtSetDevice(runtime.hccl.device_id);
    if (acl_set_device_ret != ACL_SUCCESS) {
        TraceFailure(rank_id, "aclrtSetDevice ret=" + std::to_string(acl_set_device_ret));
        return false;
    }
    TraceLog(rank_id, "runtime aclrtSetDevice done");
    TraceLog(rank_id, "aclrtCreateStream begin");
    aclError stream_ret = aclrtCreateStream(&runtime.compute_stream);
    if (stream_ret != ACL_SUCCESS) {
        TraceFailure(rank_id, "aclrtCreateStream ret=" + std::to_string(stream_ret));
        return false;
    }
    TraceLog(rank_id, "aclrtCreateStream done stream=" +
                          [&]() {
                              std::ostringstream os;
                              os << runtime.compute_stream;
                              return os.str();
                          }());
    TraceLog(rank_id, "rtStreamCreate begin");
    rtError_t rt_stream_ret = rtStreamCreate(&runtime.hccl.hccl_stream, RT_STREAM_PRIORITY_DEFAULT);
    if (rt_stream_ret != 0) {
        TraceFailure(rank_id, "rtStreamCreate ret=" + std::to_string(rt_stream_ret));
        return false;
    }
    TraceLog(rank_id, "rtStreamCreate done stream=" +
                          [&]() {
                              std::ostringstream os;
                              os << runtime.hccl.hccl_stream;
                              return os.str();
                          }());
    TraceLog(rank_id, "HcclCommInitRootInfo begin");
    HcclResult hccl_ret = HcclCommInitRootInfo(static_cast<uint32_t>(world_size), &root_info,
                                               static_cast<uint32_t>(rank_id), &runtime.hccl.comm);
    if (hccl_ret != HCCL_SUCCESS) {
        TraceFailure(rank_id, "HcclCommInitRootInfo ret=" + std::to_string(hccl_ret));
        return false;
    }
    TraceLog(rank_id, "HcclCommInitRootInfo done");

    char group[GROUP_NAME_SIZE] = {};
    TraceLog(rank_id, "HcclGetCommName begin");
    hccl_ret = HcclGetCommName(runtime.hccl.comm, group);
    if (hccl_ret != HCCL_SUCCESS) {
        TraceFailure(rank_id, "HcclGetCommName ret=" + std::to_string(hccl_ret));
        return false;
    }
    TraceLog(rank_id, std::string("HcclGetCommName done group=") + group);

    uint32_t topo = 0;
    TraceLog(rank_id, "HcomGetL0TopoTypeEx begin");
    hccl_ret = HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE);
    if (hccl_ret != HCCL_SUCCESS) {
        TraceFailure(rank_id, "HcomGetL0TopoTypeEx ret=" + std::to_string(hccl_ret) +
                                  " topo=" + std::to_string(topo));
        return false;
    }
    TraceLog(rank_id, "HcomGetL0TopoTypeEx done topo=" + std::to_string(topo));

    HcclComm comm_handle = nullptr;
    TraceLog(rank_id, "HcomGetCommHandleByGroup begin");
    hccl_ret = HcomGetCommHandleByGroup(group, &comm_handle);
    if (hccl_ret != HCCL_SUCCESS) {
        TraceFailure(rank_id, "HcomGetCommHandleByGroup ret=" + std::to_string(hccl_ret));
        return false;
    }
    TraceLog(rank_id, "HcomGetCommHandleByGroup done");

    pto_hccl_compat::CommResourceTilingV2 tiling{};
    BuildCommResourceTiling(group, tiling);

    void *ctx_ptr = nullptr;
    TraceLog(rank_id, "HcclAllocComResourceByTiling begin opType=" + std::to_string(tiling.inner.opType) +
                          " algConfig=" + std::string(tiling.inner.algConfig));
    hccl_ret = HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr);
    if (hccl_ret != HCCL_SUCCESS || ctx_ptr == nullptr) {
        TraceFailure(rank_id, "HcclAllocComResourceByTiling ret=" + std::to_string(hccl_ret) +
                                  " ctxPtr=" +
                                  [&]() {
                                      std::ostringstream os;
                                      os << ctx_ptr;
                                      return os.str();
                                  }());
        return false;
    }
    TraceLog(rank_id, "HcclAllocComResourceByTiling done ctxPtr=" +
                          [&]() {
                              std::ostringstream os;
                              os << ctx_ptr;
                              return os.str();
                          }());
    if (topo == COMM_TOPO_MESH) {
        TraceLog(rank_id, "LoadMeshContext begin");
        if (!LoadMeshContext(runtime, ctx_ptr)) {
            TraceFailure(rank_id, "LoadMeshContext failed");
            return false;
        }
        TraceLog(rank_id, "LoadMeshContext done rankId=" + std::to_string(runtime.hccl.host_ctx.rankId) +
                              " rankNum=" + std::to_string(runtime.hccl.host_ctx.rankNum) +
                              " winSize=" + std::to_string(runtime.hccl.host_ctx.winSize));
    } else {
        TraceLog(rank_id, "LoadRingContext begin topo=" + std::to_string(topo));
        if (!LoadRingContext(runtime, rank_id, ctx_ptr)) {
            TraceFailure(rank_id, "LoadRingContext failed topo=" + std::to_string(topo));
            return false;
        }
        TraceLog(rank_id, "LoadRingContext done rankId=" + std::to_string(runtime.hccl.host_ctx.rankId) +
                              " rankNum=" + std::to_string(runtime.hccl.host_ctx.rankNum) +
                              " winSize=" + std::to_string(runtime.hccl.host_ctx.winSize));
    }
    DumpMeshContextDebug(rank_id, runtime.hccl.device_ctx);
    const uint32_t local_rank = static_cast<uint32_t>(rank_id);
    const bool valid_context = runtime.hccl.host_ctx.rankNum == static_cast<uint32_t>(world_size) &&
                               runtime.hccl.host_ctx.rankId == local_rank &&
                               runtime.hccl.host_ctx.winSize > 0 &&
                               local_rank < HCCL_STANDALONE_MAX_RANK_NUM &&
                               runtime.hccl.host_ctx.windowsIn[local_rank] != 0;
    if (!valid_context) {
        TraceFailure(rank_id, "invalid HCCL context rankId=" + std::to_string(runtime.hccl.host_ctx.rankId) +
                                  " rankNum=" + std::to_string(runtime.hccl.host_ctx.rankNum) +
                                  " winSize=" + std::to_string(runtime.hccl.host_ctx.winSize) +
                                  " localWindow=0x" +
                                  [&]() {
                                      std::ostringstream os;
                                      if (local_rank < HCCL_STANDALONE_MAX_RANK_NUM) {
                                          os << std::hex << runtime.hccl.host_ctx.windowsIn[local_rank];
                                      } else {
                                          os << "out_of_range";
                                      }
                                      return os.str();
                                  }());
        return false;
    }
    TraceLog(rank_id, "InitStandaloneRankRuntime done");
    return true;
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    TraceLog(runtime.hccl.rank_id, "DestroyStandaloneRankRuntime enter");
    if (runtime.hccl.owns_device_ctx && runtime.hccl.device_ctx != nullptr) {
        TraceLog(runtime.hccl.rank_id, "aclrtFree device_ctx begin");
        aclrtFree(runtime.hccl.device_ctx);
        runtime.hccl.device_ctx = nullptr;
        runtime.hccl.owns_device_ctx = false;
        TraceLog(runtime.hccl.rank_id, "aclrtFree device_ctx done");
    }
    if (runtime.hccl.comm != nullptr) {
        TraceLog(runtime.hccl.rank_id, "HcclCommDestroy begin");
        HcclCommDestroy(runtime.hccl.comm);
        runtime.hccl.comm = nullptr;
        TraceLog(runtime.hccl.rank_id, "HcclCommDestroy done");
    }
    if (runtime.hccl.hccl_stream != nullptr &&
        runtime.hccl.hccl_stream != reinterpret_cast<rtStream_t>(runtime.compute_stream)) {
        TraceLog(runtime.hccl.rank_id, "rtStreamDestroy begin");
        rtStreamDestroy(runtime.hccl.hccl_stream);
        runtime.hccl.hccl_stream = nullptr;
        TraceLog(runtime.hccl.rank_id, "rtStreamDestroy done");
    }
    if (runtime.compute_stream != nullptr) {
        TraceLog(runtime.hccl.rank_id, "aclrtDestroyStream begin");
        aclrtDestroyStream(runtime.compute_stream);
        runtime.compute_stream = nullptr;
        TraceLog(runtime.hccl.rank_id, "aclrtDestroyStream done");
    }
    TraceLog(runtime.hccl.rank_id, "DestroyStandaloneRankRuntime exit");
}
