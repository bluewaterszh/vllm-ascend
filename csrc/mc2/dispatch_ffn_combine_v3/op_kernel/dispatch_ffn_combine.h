/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine.h
 * \brief
 */

#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

using namespace AscendC;

#include "kernel_operator.h"


#include "dispatch_ffn_combine_tiling.h"

#include "utils/dispatch_policy_custom.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "dispatch_ffn_combine_kernel.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"


namespace DispatchFFNCombineImpl {
#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

using namespace AscendC;
template <TemplateMMA2AClass>
class DispatchFFNCombine {
public:
    __aicore__ inline DispatchFFNCombine() {};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
                                GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                const __gm__ DispatchFFNCombineTilingData *tilingData);
    __aicore__ inline void Process();


private:
    GM_ADDR xGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR expertIdGM_;
    GM_ADDR scale1GM_;
    GM_ADDR scale2GM_;
    GM_ADDR probs_;
    GM_ADDR xActiveMaskGM_;
    GM_ADDR outGM_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;

    GM_ADDR moeInitRoutingQuantV2Scale = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset = nullptr;
    GM_ADDR expertTokensBeforeCapacity = nullptr;


    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t rank;
    int32_t rankSize;
    int32_t aivNum;
    GM_ADDR remoteWindowContext_;

    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzlOffset;
    int32_t swizzlDirect;
    int32_t ubMoveNum;
    int32_t pValue;

    int32_t commNpuSplit;
    int32_t commDataSplit;
    int32_t lenPerLoop;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t topK;
    int32_t expertPerRank;
    int32_t maxOutputSize;
    int32_t EP;
    int32_t listLen;

    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    uint64_t initRoutingQuantTilingKey;

    // Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

};


template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
                                                                    GM_ADDR probs, GM_ADDR xActiveMaskGM, GM_ADDR outGM, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,
                                                                    const __gm__ DispatchFFNCombineTilingData *tilingData)
{

    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    expertIdGM_ = expertIdGM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    probs_ = probs;
    xActiveMaskGM_ = xActiveMaskGM;

    outGM_ = outGM;
    gmExpertTokenNums_ = expertTokenNums;

    workspaceGM_ = workspaceGM;

    aivNum = tilingData->dispatchFFNCombineInfo.aivNum;

    m = tilingData->dispatchFFNCombineInfo.M;
    k = tilingData->dispatchFFNCombineInfo.K;
    n = tilingData->dispatchFFNCombineInfo.N;
    EP =  tilingData->dispatchFFNCombineInfo.worldSize;
    topK = tilingData->dispatchFFNCombineInfo.topK;
    expertPerRank = tilingData->dispatchFFNCombineInfo.expertPerRank;
    maxOutputSize = tilingData->dispatchFFNCombineInfo.maxOutputSize;
    listLen = tilingData->dispatchFFNCombineInfo.listLen;

    m0 = tilingData->cocTiling.m0;
    k0 = tilingData->cocTiling.k0;
    n0 = tilingData->cocTiling.n0;
    swizzlDirect = tilingData->cocTiling.swizzleDirect;
    swizzlOffset = tilingData->cocTiling.swizzleOffset;
    ubMoveNum = tilingData->cocTiling.ubMoveNum;
    pValue = tilingData->cocTiling.pValue;
    commNpuSplit = tilingData->cocTiling.commNpuSplit;
    commDataSplit = tilingData->cocTiling.commDataSplit;
    lenPerLoop = tilingData->cocTiling.lenPerLoop;
    moeInitRoutingQuantV2TilingData.coreNum = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.coreNum;
    moeInitRoutingQuantV2TilingData.n = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.n;
    moeInitRoutingQuantV2TilingData.cols = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.cols;
    moeInitRoutingQuantV2TilingData.k = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.k;
    moeInitRoutingQuantV2TilingData.expertCapacity = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.expertCapacity;
    moeInitRoutingQuantV2TilingData.expertNum = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.expertNum;
    moeInitRoutingQuantV2TilingData.dropPadMode = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.dropPadMode;
    moeInitRoutingQuantV2TilingData.expertTokensCountOrCumsumFlag =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.expertTokensCountOrCumsumFlag;
    moeInitRoutingQuantV2TilingData.expertTokensBeforeCapacityFlag =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.expertTokensBeforeCapacityFlag;
    moeInitRoutingQuantV2TilingData.smoothType = tilingData->cocTiling.moeInitRoutingQuantV2TilingData.smoothType;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.needCoreNum;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCorePerLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCorePerLoopElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreLastLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.perCoreLastLoopElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCorePerLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCorePerLoopElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreLastLoopElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.lastCoreLastLoopElements;
    moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.oneLoopMaxElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp.oneLoopMaxElements;
    moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp.needCoreNum;
    moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp.oneLoopMaxElements =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp.oneLoopMaxElements;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.needCoreNum;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.activateRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.perLoopCols;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp.colLoops;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.needCoreNum;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.activateRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.perLoopCols;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp.colLoops;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.needCoreNum =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.needCoreNum;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.activateRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.activateRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCorePerLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCorePerLoopRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreLastLoopRows =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreLastLoopRows;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perCoreLoops;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastCoreLoops;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.perLoopCols;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastLoopCols =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.lastLoopCols;
    moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.colLoops =
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp.colLoops;
    initRoutingQuantTilingKey = tilingData->cocTiling.initRoutingQuantTilingKey;

    rank = static_cast<int32_t>(tilingData->runtimeInfo.rank);
    rankSize = static_cast<int32_t>(tilingData->runtimeInfo.rankSize);
    remoteWindowContext_ = reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext);
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Process()
{
    // Define ArchTag
    using ArchTag = pto_ext::Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    uint32_t k2 = n/2;
    uint32_t n2 = k;

    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;

    using LayoutA = pto_ext::layout::ND;
    using LayoutB = typename std::conditional<
        Nz_,
        pto_ext::layout::Zn,
        typename std::conditional<TB_, pto_ext::layout::DN, pto_ext::layout::ND>::type
    >::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BType_>::create(k, n);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BType_>::create(k2, n2);
    using LayoutC = pto_ext::layout::ND;
    using L1TileShape = pto_ext::GemmShape<128, 256, 512>;   // M, N, K

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = pto_ext::Gemm::MmadAtlasA2PreloadAsyncFixpipe<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;

    using L0TileShape = pto_ext::GemmShape<128, 256, 128>;
    using AType = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;
    using BType = pto_ext::Gemm::GemmType<int8_t, LayoutB>;
    using CType = pto_ext::Gemm::GemmType<float16_t, pto_ext::layout::ND>;
    using D1Type = pto_ext::Gemm::GemmType<int8_t, pto_ext::layout::ND>;

    using D2Type = typename std::conditional<
        std::is_same_v<CType_, bfloat16_t>, 
        pto_ext::Gemm::GemmType<bfloat16_t, pto_ext::layout::ND>,
        pto_ext::Gemm::GemmType<CType_, pto_ext::layout::ND>
        >::type;

    using BlockMmad = pto_ext::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 = pto_ext::Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>;
    
    using ScaleType = pto_ext::Gemm::GemmType<uint64_t, pto_ext::layout::VectorLayout>;
    using PerTokenScaleType = pto_ext::Gemm::GemmType<float, pto_ext::layout::VectorLayout>;
    using ElementMulType = pto_ext::Gemm::GemmType<float, pto_ext::layout::ND>;
    using TileElemWiseMuls = pto_ext::Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = pto_ext::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
        D1Type, TileElemWiseMuls, TileCopy1>;

    using EpilogueDispatchPolicy2 = pto_ext::Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using EpilogueDispatchPolicy3 =  pto_ext::Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>;
    
    using TileCopy2 = pto_ext::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using BlockEpilogue2 = pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType,PerTokenScaleType,
        D2Type, TileCopy2>;
    using BlockEpilogue3 = pto_ext::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType,PerTokenScaleType,
        D2Type, TileCopy2>;


    using BlockScheduler = typename pto_ext::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    using MatmulKernel = pto_ext::Gemm::Kernel::DispatchFFNCombineKernel<BlockMmad,
        BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2, BlockEpilogue3>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    pto_ext::layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    pto_ext::layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    pto_ext::layout::ND layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    pto_ext::layout::ND layoutD2{static_cast<uint32_t>(m*topK), static_cast<uint32_t>(n2)};
    // Prepare params

    pto_ext::PtoShape3D problemShape = pto_ext::MakePtoShape3D(
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k));

    uint32_t epilogueCoreNum = aivNum;
    uint32_t epilogueGranularity = expertPerRank - 3;
    if (expertPerRank <= 4) {
        epilogueGranularity = expertPerRank - 1;
    }
    typename MatmulKernel::Params params{
        problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen), static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
        static_cast<uint32_t>(rank), static_cast<uint32_t>(rankSize), ubMoveNum, remoteWindowContext_,
        static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
        epilogueCoreNum, epilogueGranularity,
        xGM_, layoutA1, layoutA2,
        weight1GM_, layoutB1,
        weight2GM_, layoutB2,
        scale1GM_, layoutScale1,
        scale2GM_, layoutScale2,
        outGM_, layoutD1, layoutD2,
        expertIdGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
        expertTokensBeforeCapacity, probs_,
        workspaceGM_, gmExpertTokenNums_, xActiveMaskGM_, moeInitRoutingQuantV2TilingData};
    //Call kernel
    MatmulKernel kernel(params);
    kernel(params);
}

} // DispatchFFNCombineImpl
#endif // DISPATCH_FFN_COMBINE_H