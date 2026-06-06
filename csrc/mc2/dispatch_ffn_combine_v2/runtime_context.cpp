#include "runtime_context.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "hccl/hccl_tiling.h"
#include "kernel_tiling.h"

namespace {
constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;

struct StandaloneMc2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
};

bool VerboseLogEnabled()
{
    const char *value = std::getenv("DISPATCH_FFN_COMBINE_V2_VERBOSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

void DumpMeshContextDebug(int rank_id, const HcclDeviceContext *device_ctx)
{
    if (!VerboseLogEnabled()) {
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
    runtime.hccl.rank_id = rank_id;
    runtime.hccl.world_size = world_size;
    runtime.hccl.device_id = rank_id;

    if (aclrtSetDevice(runtime.hccl.device_id) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtCreateStream(&runtime.compute_stream) != ACL_SUCCESS) {
        return false;
    }
    runtime.hccl.hccl_stream = reinterpret_cast<rtStream_t>(runtime.compute_stream);
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

    StandaloneMc2TilingData tiling{};
    const uint32_t op_type = 8U;
    const std::string alg_config = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2_tiling_config(group, op_type, alg_config);
    if (mc2_tiling_config.GetTiling(tiling.mc2InitTiling) != 0 ||
        mc2_tiling_config.GetTiling(tiling.mc2CcTiling) != 0) {
        return false;
    }

    void *ctx_ptr = nullptr;
    if (HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr) != HCCL_SUCCESS ||
        ctx_ptr == nullptr) {
        return false;
    }
    if (!LoadMeshContext(runtime, ctx_ptr)) {
        return false;
    }
    DumpMeshContextDebug(rank_id, runtime.hccl.device_ctx);
    const uint32_t local_rank = static_cast<uint32_t>(rank_id);
    return runtime.hccl.host_ctx.rankNum == static_cast<uint32_t>(world_size) &&
           runtime.hccl.host_ctx.rankId == local_rank &&
           runtime.hccl.host_ctx.winSize > 0 &&
           local_rank < HCCL_STANDALONE_MAX_RANK_NUM &&
           runtime.hccl.host_ctx.windowsIn[local_rank] != 0;
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    if (runtime.hccl.comm != nullptr) {
        HcclCommDestroy(runtime.hccl.comm);
        runtime.hccl.comm = nullptr;
    }
    if (runtime.hccl.hccl_stream != nullptr &&
        runtime.hccl.hccl_stream != reinterpret_cast<rtStream_t>(runtime.compute_stream)) {
        rtStreamDestroy(runtime.hccl.hccl_stream);
        runtime.hccl.hccl_stream = nullptr;
    }
    if (runtime.compute_stream != nullptr) {
        aclrtDestroyStream(runtime.compute_stream);
        runtime.compute_stream = nullptr;
    }
}
