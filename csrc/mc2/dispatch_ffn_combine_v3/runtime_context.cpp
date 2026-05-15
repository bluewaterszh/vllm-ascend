#include "runtime_context.hpp"

#include <cstdint>
#include <cstring>

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

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool BuildWindowTable(StandaloneRankRuntime &runtime)
{
    runtime.window_table_host.assign(runtime.hccl.host_ctx.rankNum, 0);
    for (uint32_t i = 0; i < runtime.hccl.host_ctx.rankNum; ++i) {
        runtime.window_table_host[i] = runtime.hccl.host_ctx.windowsIn[i];
    }

    const size_t bytes = runtime.window_table_host.size() * sizeof(uint64_t);
    if (aclrtMalloc(&runtime.window_table_dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        runtime.window_table_dev = nullptr;
        return false;
    }
    return aclrtMemcpy(runtime.window_table_dev, bytes, runtime.window_table_host.data(), bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
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
    if (HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE) != HCCL_SUCCESS || topo != COMM_TOPO_MESH) {
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
    std::strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1);
    std::strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1);

    void *ctx_ptr = nullptr;
    if (HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr) != HCCL_SUCCESS ||
        ctx_ptr == nullptr) {
        return false;
    }
    if (!LoadMeshContext(runtime, ctx_ptr)) {
        return false;
    }
    return BuildWindowTable(runtime);
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    if (runtime.window_table_dev != nullptr) {
        aclrtFree(runtime.window_table_dev);
        runtime.window_table_dev = nullptr;
    }
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
