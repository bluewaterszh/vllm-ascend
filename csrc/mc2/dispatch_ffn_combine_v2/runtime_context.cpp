#include "runtime_context.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>

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

bool LoadMeshContext(StandaloneRankRuntime &runtime, void *ctx_ptr)
{
    runtime.hccl.device_ctx = reinterpret_cast<HcclDeviceContext *>(ctx_ptr);
    runtime.hccl.owns_device_ctx = false;
    return aclrtMemcpy(&runtime.hccl.host_ctx, sizeof(runtime.hccl.host_ctx), runtime.hccl.device_ctx,
                       sizeof(runtime.hccl.host_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
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
    runtime.hccl.hccl_stream = reinterpret_cast<rtStream_t>(runtime.compute_stream);
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
    if (hccl_ret != HCCL_SUCCESS || topo != COMM_TOPO_MESH) {
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

    StandaloneMc2TilingData tiling{};
    const uint32_t op_type = 8U;
    const std::string alg_config = "AlltoAll=level0:fullmesh;level1:pairwise";
    TraceLog(rank_id, "Mc2CcTilingConfig begin opType=" + std::to_string(op_type) +
                          " algConfig=" + alg_config);
    AscendC::Mc2CcTilingConfig mc2_tiling_config(group, op_type, alg_config);
    const int init_tiling_ret = mc2_tiling_config.GetTiling(tiling.mc2InitTiling);
    const int cc_tiling_ret = mc2_tiling_config.GetTiling(tiling.mc2CcTiling);
    if (init_tiling_ret != 0 || cc_tiling_ret != 0) {
        TraceFailure(rank_id, "Mc2CcTilingConfig::GetTiling initRet=" + std::to_string(init_tiling_ret) +
                                  " ccRet=" + std::to_string(cc_tiling_ret));
        return false;
    }
    TraceLog(rank_id, "Mc2CcTilingConfig done");

    void *ctx_ptr = nullptr;
    TraceLog(rank_id, "HcclAllocComResourceByTiling begin");
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
    TraceLog(rank_id, "LoadMeshContext begin");
    if (!LoadMeshContext(runtime, ctx_ptr)) {
        TraceFailure(rank_id, "LoadMeshContext failed");
        return false;
    }
    TraceLog(rank_id, "LoadMeshContext done rankId=" + std::to_string(runtime.hccl.host_ctx.rankId) +
                          " rankNum=" + std::to_string(runtime.hccl.host_ctx.rankNum) +
                          " winSize=" + std::to_string(runtime.hccl.host_ctx.winSize));
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
