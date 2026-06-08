#include "tiling_builder.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_launch.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint32_t DEFAULT_AIV_NUM = 20U;
constexpr uint64_t DEFAULT_UB_SIZE = 196352U;
constexpr uint64_t MIB = 1024U * 1024U;
constexpr uint64_t HCCL_WINDOW_RESERVED_BYTES = 3U * MIB;

#ifndef DISPATCH_FFN_COMBINE_V2_REQUESTED_SOC_VERSION
#define DISPATCH_FFN_COMBINE_V2_REQUESTED_SOC_VERSION "ascend910_93"
#endif

#ifndef DISPATCH_FFN_COMBINE_V2_COMPILE_SOC_VERSION
#define DISPATCH_FFN_COMBINE_V2_COMPILE_SOC_VERSION "ascend910_9391"
#endif

struct PlatformCoreInfo {
    uint32_t aiv_num = DEFAULT_AIV_NUM;
    uint64_t ub_size = DEFAULT_UB_SIZE;
    std::string soc_version = "constant-default";
    bool from_fallback = false;
};

bool VerboseLogEnabled()
{
    const char *value = std::getenv("DISPATCH_FFN_COMBINE_V2_VERBOSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool ShouldLogFallback(int rank_id)
{
    return rank_id == 0 || VerboseLogEnabled();
}

bool StartsWith(const std::string &value, const std::string &prefix)
{
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

std::string CanonicalSocName(const std::string &soc)
{
    if (soc == "ascend910_93") {
        return "Ascend910_93";
    }
    if (StartsWith(soc, "ascend910_")) {
        return "Ascend910_" + soc.substr(std::string("ascend910_").size());
    }
    if (soc == "ascend910b") {
        return "Ascend910B";
    }
    if (StartsWith(soc, "ascend910b")) {
        return "Ascend910B" + soc.substr(std::string("ascend910b").size());
    }
    return soc;
}

void AddSocCandidate(std::vector<std::string> &candidates, const std::string &soc)
{
    const std::string canonical = CanonicalSocName(soc);
    if (canonical.empty()) {
        return;
    }
    if (std::find(candidates.begin(), candidates.end(), canonical) == candidates.end()) {
        candidates.push_back(canonical);
    }
}

std::vector<std::string> BuildSocCandidates()
{
    std::vector<std::string> candidates;
    AddSocCandidate(candidates, DISPATCH_FFN_COMBINE_V2_REQUESTED_SOC_VERSION);
    AddSocCandidate(candidates, DISPATCH_FFN_COMBINE_V2_COMPILE_SOC_VERSION);
    AddSocCandidate(candidates, "Ascend910B1");
    AddSocCandidate(candidates, "Ascend910_93");
    return candidates;
}

bool TryGetPlatformCoreInfo(const std::string &soc_version, PlatformCoreInfo &info)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc_version.c_str());
    if (platform == nullptr) {
        return false;
    }
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    if (aiv_num == 0) {
        return false;
    }

    uint64_t ub_size = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    info.aiv_num = aiv_num;
    info.ub_size = ub_size == 0 ? DEFAULT_UB_SIZE : ub_size;
    info.soc_version = soc_version;
    return true;
}

PlatformCoreInfo ResolvePlatformCoreInfo(int rank_id)
{
    const std::vector<std::string> candidates = BuildSocCandidates();
    const std::string requested_soc = CanonicalSocName(DISPATCH_FFN_COMBINE_V2_REQUESTED_SOC_VERSION);
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        PlatformCoreInfo info;
        if (TryGetPlatformCoreInfo(candidates[idx], info)) {
            info.from_fallback = idx != 0;
            if (info.from_fallback && ShouldLogFallback(rank_id)) {
                std::cerr << "rank=" << rank_id
                          << " platform fallback: requestedSoc=" << requested_soc
                          << " resolvedSoc=" << info.soc_version
                          << " aivNum=" << info.aiv_num
                          << " ubSize=" << info.ub_size
                          << std::endl;
            }
            return info;
        }
        if (ShouldLogFallback(rank_id)) {
            std::cerr << "rank=" << rank_id
                      << " platform query failed: requestedSoc=" << requested_soc
                      << " candidateSoc=" << candidates[idx]
                      << std::endl;
        }
    }

    PlatformCoreInfo info;
    info.from_fallback = true;
    if (ShouldLogFallback(rank_id)) {
        std::cerr << "rank=" << rank_id
                  << " platform fallback: requestedSoc=" << requested_soc
                  << " resolvedSoc=constant-default"
                  << " aivNum=" << info.aiv_num
                  << " ubSize=" << info.ub_size
                  << std::endl;
    }
    return info;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1U) / align * align;
}

uint64_t RequiredHcclWindowBytes(const CaseConfig &cfg)
{
    const uint64_t packedOffsetABytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.k + 32U);
    const uint64_t offsetAWindowBytes = packedOffsetABytes * 3U;
    const uint64_t offsetDBytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const uint64_t offsetDWindowBytes = ((offsetDBytes + HCCL_WINDOW_RESERVED_BYTES + 511U) * 3U + 1U) / 2U;
    return std::max(offsetAWindowBytes, offsetDWindowBytes);
}

void RequireHcclWindowCapacity(const CaseConfig &cfg, const StandaloneRankRuntime &runtime)
{
    const uint64_t requiredBytes = RequiredHcclWindowBytes(cfg);
    const uint64_t actualBytes = runtime.hccl.host_ctx.winSize;
    if (actualBytes >= requiredBytes) {
        return;
    }
    const uint64_t requiredMb = AlignUp(requiredBytes, MIB) / MIB;
    const uint64_t actualMb = actualBytes / MIB;
    throw std::runtime_error("HCCL window is too small for dispatch_ffn_combine_v2: actual=" +
                             std::to_string(actualBytes) + " bytes (" + std::to_string(actualMb) +
                             " MB), required>=" + std::to_string(requiredBytes) + " bytes (" +
                             std::to_string(requiredMb) + " MB). Increase HCCL_BUFFSIZE for maxOutputSize=" +
                             std::to_string(cfg.max_output_size) + " K=" + std::to_string(cfg.k));
}

void FillCoCTiling(CoCTiling &coc, const CaseConfig &cfg)
{
    coc.m0 = 128;
    coc.k0 = 256;
    coc.n0 = 256;
    coc.swizzleDirect = 1;
    coc.swizzleOffset = 7;
    coc.ubMoveNum = 16 * 1024;
    coc.pValue = 1;
    coc.commNpuSplit = cfg.world_size;
    coc.commDataSplit = 1;
    coc.lenPerLoop = coc.m0 * coc.n0 / 2;
}

uint64_t FillInitRoutingTiling(CoCTiling &coc, const CaseConfig &cfg)
{
    optiling::MoeInitRoutingQuantV2TilingBase tilingBase;
    const int64_t inputXDtypeSize = sizeof(int16_t);
    const int64_t scaleDim0 = 0;
    const int64_t ubSize = 196352;
    const int64_t expertCapacity = 0;
    const int64_t expertNum = static_cast<int64_t>(cfg.expert_per_rank) * cfg.world_size + 1;
    const int64_t activeNum = 0;
    const int64_t dropPadMode = 0;
    const int64_t expertTokensCountOrCumsumFlag = 2;
    const bool expertTokensBeforeCapacityFlag = false;
    const int64_t quantMode = 1;
    const int64_t aivNumInitRouting = 40;
    if (!tilingBase.DoTiling(cfg.m, cfg.k, cfg.topk, expertCapacity, expertNum, activeNum, dropPadMode,
                             expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inputXDtypeSize, quantMode,
                             scaleDim0, aivNumInitRouting, ubSize)) {
        throw std::runtime_error("MoeInitRoutingQuantV2TilingBase::DoTiling failed");
    }
    coc.initRoutingQuantTilingKey = tilingBase.tilingKey_;
    coc.moeInitRoutingQuantV2TilingData = tilingBase.quantTilingData;
    return static_cast<uint64_t>(tilingBase.workspaceSize_);
}

uint32_t GetBlockDim(uint32_t aivNum)
{
    // The generated direct-launch wrapper already packs the AIC/AIV mix task.
    // Passing the framework TSCH blockDim (60 on 910B) makes AIV get_block_num()
    // become 60, while the kernel's logical AIV tiling expects 20 * 2 = 40.
    return aivNum;
}
} // namespace

DispatchFFNCombineBuildResult BuildDispatchFFNCombineTiling(const CaseConfig &cfg,
                                                            const StandaloneRankRuntime &runtime)
{
    DispatchFFNCombineBuildResult result;
    auto &info = result.tiling.dispatchFFNCombineInfo;
    info.M = cfg.m;
    info.K = cfg.k;
    info.N = cfg.n;
    info.topK = cfg.topk;
    info.expertPerRank = cfg.expert_per_rank;
    info.worldSize = cfg.world_size;
    info.maxOutputSize = cfg.max_output_size;
    info.isTransposeB = 0;
    info.isWeightNz = 1;
    info.listLen = cfg.list_len;
    const PlatformCoreInfo platform_info = ResolvePlatformCoreInfo(runtime.hccl.rank_id);
    info.aivNum = platform_info.aiv_num;
    info.totalUbSize = static_cast<uint32_t>(platform_info.ub_size);

    RequireHcclWindowCapacity(cfg, runtime);
    FillCoCTiling(result.tiling.cocTiling, cfg);
    const uint64_t initRoutingWorkspace = FillInitRoutingTiling(result.tiling.cocTiling, cfg);
    const auto &moe_tiling = result.tiling.cocTiling.moeInitRoutingQuantV2TilingData;
    if (VerboseLogEnabled()) {
        std::cerr << "rank=" << runtime.hccl.rank_id
                  << " initRoutingTilingKey=" << result.tiling.cocTiling.initRoutingQuantTilingKey
                  << " initRoutingWorkspace=" << initRoutingWorkspace
                  << " aivNum=" << info.aivNum
                  << " moeCoreNum=" << moe_tiling.coreNum
                  << " vbsNeedCoreNum=" << moe_tiling.vbsComputeParamsOp.needCoreNum
                  << " vbsPerCoreElements=" << moe_tiling.vbsComputeParamsOp.perCoreElements
                  << " vbsPerCoreLoops=" << moe_tiling.vbsComputeParamsOp.perCoreLoops
                  << " vbsPerCorePerLoopElements=" << moe_tiling.vbsComputeParamsOp.perCorePerLoopElements
                  << " vmsNeedCoreNum=" << moe_tiling.vmsMiddleComputeParamsOp.needCoreNum
                  << " sortOutOneLoopMaxElements=" << moe_tiling.sortOutComputeParamsOp.oneLoopMaxElements
                  << std::endl;
    }

    result.tiling.runtimeInfo.symmetricPtr = reinterpret_cast<uint64_t>(runtime.hccl.device_ctx);
    result.tiling.runtimeInfo.segmentSize = runtime.hccl.host_ctx.winSize;
    result.tiling.runtimeInfo.rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    result.tiling.runtimeInfo.rankSize = static_cast<uint32_t>(runtime.hccl.world_size);

    const uint32_t n2 = cfg.k;
    const uint32_t k2 = cfg.n / 2;
    const uint64_t cocWorkspace = ((cfg.m + 255) / 256) * 256 * cfg.topk * sizeof(int32_t)
        + static_cast<uint64_t>(cfg.world_size) * cfg.world_size * cfg.expert_per_rank * sizeof(int32_t) * 3
        + static_cast<uint64_t>(cfg.max_output_size) * sizeof(float) * 2
        + static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(int16_t)
        + static_cast<uint64_t>(cfg.max_output_size) * n2 * sizeof(int16_t)
        + static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t)
        + static_cast<uint64_t>(cfg.max_output_size) * k2 * sizeof(int8_t)
        + static_cast<uint64_t>(cfg.world_size) * sizeof(int32_t) * 16
        + static_cast<uint64_t>(cfg.expert_per_rank + cfg.world_size) * sizeof(int32_t) * 16;

    result.block_dim = GetBlockDim(info.aivNum);
    result.workspace_bytes = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);
    if (VerboseLogEnabled()) {
        std::cerr << "rank=" << runtime.hccl.rank_id
                  << " blockDim=" << result.block_dim
                  << " cocWorkspace=" << cocWorkspace
                  << " workspaceBytes=" << result.workspace_bytes
                  << std::endl;
    }
    result.tiling.launchConfig.blockDim = result.block_dim;
    result.tiling.launchConfig.tilingKey = DISPATCH_FFN_COMBINE_DEVICE_TILING_KEY;
    result.tiling.launchConfig.workspaceBytes = result.workspace_bytes;
    return result;
}
