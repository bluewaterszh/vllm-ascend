#include "tiling_builder.hpp"

#include <algorithm>
#include <stdexcept>

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

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

void FillInitRoutingTiling(CoCTiling &coc, const CaseConfig &cfg)
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
}

uint32_t GetAivNum()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910_93");
    if (platform == nullptr) {
        return 20;
    }
    return platform->GetCoreNumAiv();
}

uint32_t GetBlockDim(uint32_t aivNum)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910_93");
    if (platform == nullptr) {
        return 60;
    }
    return platform->CalcTschBlockDim(aivNum, platform->GetCoreNumAic(), aivNum);
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
    info.aivNum = GetAivNum();
    info.totalUbSize = 196352;

    FillCoCTiling(result.tiling.cocTiling, cfg);
    FillInitRoutingTiling(result.tiling.cocTiling, cfg);

    result.tiling.runtimeInfo.remoteWindowContext =
        reinterpret_cast<uint64_t>(runtime.hccl.RemoteWindowContextPtr());
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
    result.workspace_bytes = SYSTEM_NEED_WORKSPACE + cocWorkspace;
    result.tiling.launchConfig.blockDim = result.block_dim;
    result.tiling.launchConfig.tilingKey = 1000010;
    result.tiling.launchConfig.workspaceBytes = result.workspace_bytes;
    return result;
}
