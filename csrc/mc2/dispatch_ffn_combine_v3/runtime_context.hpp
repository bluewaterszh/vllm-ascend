#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "op_kernel/utils/hccl_context.hpp"

using rtError_t = int32_t;
using rtStream_t = void *;

extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *resourceTiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, uint32_t *topoType, uint32_t isSetDevice);

struct StandaloneHcclContext {
    int rank_id = 0;
    int world_size = 0;
    int device_id = 0;
    rtStream_t hccl_stream = nullptr;
    HcclComm comm = nullptr;
    HcclDeviceContext *device_ctx = nullptr;
    HcclDeviceContext host_ctx{};
    bool owns_device_ctx = false;

    HcclDeviceContext *DeviceContextPtr() const
    {
        return device_ctx;
    }

    uint64_t WindowBytes() const
    {
        return host_ctx.winSize;
    }

    uint32_t RankCount() const
    {
        return host_ctx.rankNum;
    }

    void *WindowIn(uint32_t rank) const
    {
        return reinterpret_cast<void *>(host_ctx.windowsIn[rank]);
    }

    void AttachExternalDeviceContext(HcclDeviceContext *deviceCtx);
    void ReleaseDeviceContext();
    void ResetHostDeviceContext();
    void SetHostDeviceWorkspace(uint64_t workspace, uint64_t workspaceSize);
    void SetHostDeviceInfo(uint32_t rankId, uint32_t rankCount, uint64_t windowBytes);
    void SetHostDeviceWindow(uint32_t rank, uint64_t windowIn, uint64_t windowOut);
    bool LoadHostDeviceContextFromDevice();
    bool CopyHostDeviceContextToDevice();
};

struct StandaloneRankRuntime {
    StandaloneHcclContext hccl;
    aclrtStream compute_stream = nullptr;
};

bool InitStandaloneRankRuntime(StandaloneRankRuntime &runtime, int rank_id, int world_size, const HcclRootInfo &root_info);
void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime);
