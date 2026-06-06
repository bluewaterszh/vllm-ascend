#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"

constexpr uint32_t HCCL_STANDALONE_MAX_RANK_NUM = 32U;

struct HcclDeviceContext {
    uint64_t workSpace = 0;
    uint64_t workSpaceSize = 0;
    uint32_t rankId = 0;
    uint32_t rankNum = 0;
    uint64_t winSize = 0;
    uint64_t windowsIn[HCCL_STANDALONE_MAX_RANK_NUM] = {};
    uint64_t windowsOut[HCCL_STANDALONE_MAX_RANK_NUM] = {};
};

using rtError_t = int32_t;
using rtStream_t = void *;

extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
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
};

struct StandaloneRankRuntime {
    StandaloneHcclContext hccl;
    aclrtStream compute_stream = nullptr;
};

bool InitStandaloneRankRuntime(StandaloneRankRuntime &runtime, int rank_id, int world_size, const HcclRootInfo &root_info);
void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime);
