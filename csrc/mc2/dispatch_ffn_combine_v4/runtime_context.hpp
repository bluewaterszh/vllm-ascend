#pragma once

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "op_kernel/dispatch_ffn_combine_tiling.h"
#include "op_kernel/protocol/remote_window.hpp"

using rtError_t = int32_t;
using rtStream_t = void*;

extern "C" rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* resourceTiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, uint32_t* topoType, uint32_t isSetDevice);

namespace mc2::v4 {

struct ModeConfig {
    std::string mode = "default";
    uint32_t localRank = 0;
    uint32_t worldSize = 2;
};

struct StandaloneRuntimeContext {
    protocol::RemoteWindowContext remote{};
    KernelLaunchConfig launch{};
    WorkspaceLayoutConfig layout{};
    ModeConfig mode{};
};

struct StandaloneHcclContext {
    int rank_id = 0;
    int world_size = 0;
    int device_id = 0;
    rtStream_t hccl_stream = nullptr;
    HcclComm comm = nullptr;
    protocol::RemoteWindowContext* remote_window_ctx = nullptr;
    protocol::RemoteWindowContext host_remote_window_ctx{};
    bool owns_remote_window_ctx = false;

    protocol::RemoteWindowContext* RemoteWindowContextPtr() const {
        return remote_window_ctx;
    }

    uint64_t WindowBytes() const {
        return host_remote_window_ctx.segmentBytes;
    }

    uint32_t RankCount() const {
        return host_remote_window_ctx.rankSize;
    }

    void* WindowIn(uint32_t rank) const {
        return reinterpret_cast<void*>(host_remote_window_ctx.windowIn[rank]);
    }

    void ReleaseRemoteWindowContext();
    void ResetHostRemoteWindowContext();
    void SetHostRankInfo(uint32_t rank, uint32_t rankCount, uint64_t windowBytes);
    void SetHostWindow(uint32_t rank, uint64_t windowIn, uint64_t windowOut);
    void SetHostWindowLayout(const WorkspaceLayoutConfig& layout);
    bool CopyHostRemoteWindowContextToDevice();
};

struct StandaloneRankRuntime {
    StandaloneHcclContext hccl;
    aclrtStream compute_stream = nullptr;
    WorkspaceLayoutConfig layout{};
    KernelLaunchConfig launch{};
    ModeConfig mode{};
};

WorkspaceLayoutConfig BuildDefaultWorkspaceLayout();
KernelLaunchConfig BuildDefaultLaunchConfig();
StandaloneRuntimeContext BuildSkeletonContext(const ModeConfig& mode = {});
bool InitStandaloneRankRuntime(StandaloneRankRuntime& runtime, const ModeConfig& mode, const HcclRootInfo& rootInfo);
void DestroyStandaloneRankRuntime(StandaloneRankRuntime& runtime);

}  // namespace mc2::v4
