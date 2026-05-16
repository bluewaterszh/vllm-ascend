#include "runtime_context.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool ReadRingParams(uint8_t *raw_ctx,
                    pto_hccl_compat::HcclOpResParamHead &head,
                    std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr)
{
    const size_t head_offset = offsetof(pto_hccl_compat::HcclOpResParam, localUsrRankId);
    if (aclrtMemcpy(&head, sizeof(head), raw_ctx + head_offset, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }

    if (head.rankSize == 0 || head.rankSize > PTO_HCCL_MAX_RANKS) {
        return false;
    }

    const size_t remote_res_offset = offsetof(pto_hccl_compat::HcclOpResParam, remoteRes);
    const size_t remote_res_bytes = static_cast<size_t>(head.rankSize) * sizeof(pto_hccl_compat::RemoteResPtr);
    remote_res_arr.resize(head.rankSize);
    if (aclrtMemcpy(remote_res_arr.data(), remote_res_bytes, raw_ctx + remote_res_offset, remote_res_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

bool BuildRingHostCtx(StandaloneRankRuntime &runtime,
                      uint8_t *raw_ctx,
                      const pto_hccl_compat::HcclOpResParamHead &head,
                      const std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr)
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

        pto_hccl_compat::HcclRankRelationResV2 remote_info{};
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

    char group[pto_hccl_compat::GROUP_NAME_SIZE] = {};
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

    void *ctx_ptr = nullptr;
    if (HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr) != HCCL_SUCCESS ||
        ctx_ptr == nullptr) {
        return false;
    }

    if (topo == COMM_TOPO_MESH) {
        return LoadMeshContext(runtime, ctx_ptr);
    }

    auto *raw_ctx = reinterpret_cast<uint8_t *>(ctx_ptr);
    pto_hccl_compat::HcclOpResParamHead head{};
    std::vector<pto_hccl_compat::RemoteResPtr> remote_res_arr;
    if (!ReadRingParams(raw_ctx, head, remote_res_arr)) {
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
