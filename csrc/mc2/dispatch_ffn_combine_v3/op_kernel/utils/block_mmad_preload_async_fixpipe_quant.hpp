/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP
#define PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP

#include "dispatch_policy_custom.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

namespace pto_ext::Gemm::Block {
namespace detail {

using PtoShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PtoStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_DEVICE void LaunchPtoMatmul(TileAcc &cTile, TileLeft &aTile, TileRight &bTile, bool initC,
                                           uint8_t unitFlag)
{
    const bool isFinal = (unitFlag == 0b11);
    const bool isPartial = (unitFlag == 0b10);

    if (initC) {
        if (isFinal) {
            pto::TMATMUL<pto::AccPhase::Final>(cTile, aTile, bTile);
        } else if (isPartial) {
            pto::TMATMUL<pto::AccPhase::Partial>(cTile, aTile, bTile);
        } else {
            pto::TMATMUL(cTile, aTile, bTile);
        }
    } else {
        if (isFinal) {
            pto::TMATMUL_ACC<pto::AccPhase::Final>(cTile, aTile, bTile);
        } else if (isPartial) {
            pto::TMATMUL_ACC<pto::AccPhase::Partial>(cTile, aTile, bTile);
        } else {
            pto::TMATMUL_ACC(cTile, aTile, bTile);
        }
    }
}

template <typename ElementAccumulator, typename ElementA, typename ElementB, class L0TileShape>
PTO_DEVICE void PtoTileMmad(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
                                       AscendC::LocalTensor<ElementA> const &l0ATensor,
                                       AscendC::LocalTensor<ElementB> const &l0BTensor,
                                       uint32_t m, uint32_t n, uint32_t k,
                                       bool initC = true, uint8_t unitFlag = 0)
{
    using LeftTile = pto::TileLeft<ElementA, L0TileShape::M, L0TileShape::K, pto::DYNAMIC, pto::DYNAMIC>;
    using RightTile = pto::TileRight<ElementB, L0TileShape::K, L0TileShape::N, pto::DYNAMIC, pto::DYNAMIC>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, L0TileShape::M, L0TileShape::N, pto::DYNAMIC,
                                        pto::DYNAMIC>;

    LeftTile aTile(m, k);
    RightTile bTile(k, n);
    AccTile cTile(m, n);

    pto::TASSIGN(aTile, reinterpret_cast<uint64_t>(l0ATensor.GetPhyAddr()));
    pto::TASSIGN(bTile, reinterpret_cast<uint64_t>(l0BTensor.GetPhyAddr()));
    pto::TASSIGN(cTile, reinterpret_cast<uint64_t>(l0CTensor.GetPhyAddr()));

    LaunchPtoMatmul(cTile, aTile, bTile, initC, unitFlag);

    constexpr uint32_t kPipeBarrierThreshold = 10;
    constexpr uint32_t kFractalEdge = 16;
    if ((m / kFractalEdge) * (n / kFractalEdge) < kPipeBarrierThreshold) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
PTO_DEVICE void PtoStoreAccToGm(AscendC::GlobalTensor<ElementDst> const &dst,
                                    AscendC::LocalTensor<ElementAccumulator> const &src,
                                    AscendC::LocalTensor<uint64_t> const &scale,
                                    layout::ND const &dstLayout)
{
    using GlobalDataOut = pto::GlobalTensor<ElementDst, PtoShapeDyn, PtoStrideDyn, pto::Layout::ND>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1,
                                  pto::DYNAMIC, pto::SLayout::NoneBox>;

    const int validRow = static_cast<int>(dstLayout.shape(0));
    const int validCol = static_cast<int>(dstLayout.shape(1));
    const int64_t leadingDim = static_cast<int64_t>(dstLayout.stride(0));

    PtoShapeDyn shape(1, 1, 1, validRow, validCol);
    PtoStrideDyn stride(validRow * leadingDim, validRow * leadingDim, validRow * leadingDim, leadingDim, 1);

    auto *dstPtr = const_cast<__gm__ ElementDst *>(dst.GetPhyAddr());
    GlobalDataOut dstGlobal(dstPtr, shape, stride);
    AccTile accTile(validRow, validCol);
    ScalingTile scalingTile(validCol);

    pto::TASSIGN(accTile, reinterpret_cast<uint64_t>(src.GetPhyAddr()));
    pto::TASSIGN(scalingTile, reinterpret_cast<uint64_t>(scale.GetPhyAddr()));

    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, pto::AtomicType::AtomicNone, reluMode>(
            dstGlobal, accTile, scalingTile);
    } else {
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, accTile, scalingTile);
    }
}

template <typename CopyGmToL1S, typename CopyL1ToFP>
PTO_DEVICE void StagePerChannelScale(CopyGmToL1S &copyGmToL1S,
                                         CopyL1ToFP &copyL1ToFP,
                                         AscendC::LocalTensor<uint64_t> const &l1STensor,
                                         AscendC::LocalTensor<uint64_t> const &fixpipeBuf,
                                         AscendC::GlobalTensor<uint64_t> const &gmBlockS,
                                         layout::VectorLayout const &layoutScale,
                                         uint32_t cols)
{
    auto layoutTileS = layoutScale.GetTileLayout(MakePtoCoord1D(cols));
    layout::VectorLayout layoutFpBuf{cols};
    copyGmToL1S(l1STensor, gmBlockS, layoutTileS, layoutTileS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(0);
    copyL1ToFP(fixpipeBuf, l1STensor, layoutFpBuf, layoutTileS);
}

template <typename ElementA, typename ElementC, typename ElementAccumulator, int Rows, int Cols, typename CopyL0CToGm>
PTO_DEVICE void StoreAccumulator(AscendC::GlobalTensor<ElementC> const &dst,
                                     AscendC::LocalTensor<ElementAccumulator> const &src,
                                     AscendC::LocalTensor<uint64_t> const &scale,
                                     CopyL0CToGm &copyL0CToGm,
                                     layout::ND const &dstLayout,
                                     layout::Zn const &srcLayout,
                                     uint8_t unitFlag = 0)
{
    if constexpr (std::is_same_v<ElementA, int8_t>) {
        PtoStoreAccToGm<ElementC, ElementAccumulator, Rows, Cols>(dst, src, scale, dstLayout);
    } else if constexpr (std::is_same_v<ElementA, half>) {
        if (unitFlag == 0) {
            copyL0CToGm(dst, src, dstLayout, srcLayout);
        } else {
            copyL0CToGm(dst, src, dstLayout, srcLayout, unitFlag);
        }
    }
}

template <class ArchTag, class TileCopy_, class AType_, class BType_, class CType_>
struct MatmulShell {
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;

    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyGmToL1S = pto_ext::Gemm::PtoCopyGmToL1<ArchTag, Gemm::GemmType<uint64_t, layout::VectorLayout>>;
    using CopyL1ToFP = typename pto_ext::Gemm::PtoQuantTileCopy<
        ArchTag,
        AType_,
        BType_,
        CType_,
        void,
        pto_ext::Gemm::Tile::ScaleGranularity::PER_CHANNEL>::CopyL1ToFP;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator = typename pto_ext::Gemm::PtoElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyL0CToGm = typename std::conditional<
        std::is_same_v<ElementA, int8_t>,
        pto_ext::Gemm::PtoCopyL0CToGm<ArchTag, ElementAccumulator, CType_, Gemm::Tile::ScaleGranularity::PER_CHANNEL>,
        typename TileCopy_::CopyL0CToGm>::type;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using L1AAlignHelper = pto_ext::Gemm::PtoL1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = pto_ext::Gemm::PtoL1AlignHelper<ElementB, LayoutB>;
};

}  // namespace detail


template<AscendC::HardEvent event>
__aicore__ inline void SyncFlagFunc(int32_t eventID)
{
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <
    uint32_t PRELOAD_STAGES_,
    uint32_t L1_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2PreloadAsyncFixpipe<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2PreloadAsyncFixpipe<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using MatmulShell = detail::MatmulShell<ArchTag, TileCopy_, AType_, BType_, CType_>;
    using CopyGmToL1A = typename MatmulShell::CopyGmToL1A;
    using CopyGmToL1B = typename MatmulShell::CopyGmToL1B;
    using CopyGmToL1S = typename MatmulShell::CopyGmToL1S;
    using CopyL1ToFP = typename MatmulShell::CopyL1ToFP;
    using CopyL1ToL0A = typename MatmulShell::CopyL1ToL0A;
    using CopyL1ToL0B = typename MatmulShell::CopyL1ToL0B;
    using ElementAccumulator = typename MatmulShell::ElementAccumulator;
    using CopyL0CToGm = typename MatmulShell::CopyL0CToGm;
    using LayoutAInL1 = typename MatmulShell::LayoutAInL1;
    using LayoutBInL1 = typename MatmulShell::LayoutBInL1;
    using LayoutAInL0 = typename MatmulShell::LayoutAInL0;
    using LayoutBInL0 = typename MatmulShell::LayoutBInL0;
    using LayoutCInL0 = layout::Zn;

    using L1AAlignHelper = typename MatmulShell::L1AAlignHelper;
    using L1BAlignHelper = typename MatmulShell::L1BAlignHelper;

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L1S_TILE_SIZE = L1TileShape::N * sizeof(int64_t);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::ND>, "LayoutC only supports ND.");

    // Check L1TileShape
    static_assert(
        (std::is_same_v<ElementA, int8_t> 
            ? (L1A_TILE_SIZE + L1B_TILE_SIZE + L1S_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE
            : (L1A_TILE_SIZE + L1B_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE),
        "L1TileShape exceeding the L1 space for the given data type"
    );

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    PTO_DEVICE static LayoutAInL1 MakeL1ALayout()
    {
        return LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
    }

    PTO_DEVICE static LayoutBInL1 MakeL1BLayout()
    {
        return LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
    }

    PTO_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, __gm__ int32_t* flagPtr = nullptr, int32_t expertPerRank = 0, 
                uint32_t l1BufAddrStart = 0, uint32_t FpAddrStart = 0)
    {
        syncGroupIdx = 0;
        ptrSoftFlagBase_ = flagPtr;
        expertPerRank_ = expertPerRank;
        InitL1(resource, l1BufAddrStart);
        InitFpBuf(resource, FpAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
    }

    PTO_DEVICE
    ~BlockMmad()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        }
    }

    PTO_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        AscendC::GlobalTensor<uint64_t> const &gmBlockS, layout::VectorLayout const &layoutScale,
        PtoShape3D const &actualShape, int32_t syncLoopIdx = -1, int32_t flag = 0
    )
    {
        uint32_t actualM = GetPtoShapeM(actualShape);
        uint32_t actualN = GetPtoShapeN(actualShape);
        uint32_t actualK = GetPtoShapeK(actualShape);
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualK);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ?
                (startTileIdx + kLoopIdx) : (startTileIdx + kLoopIdx - kTileCount);

            uint32_t kActual = (kTileIdx < kTileCount - 1) ?
                L1TileShape::K : (actualK - kTileIdx * L1TileShape::K);

            // Emission load instruction from GM to L1
            auto gmTileAOffset = MakePtoCoord2D(0, kTileIdx * L1TileShape::K);
            auto gmTileBOffset = MakePtoCoord2D(kTileIdx * L1TileShape::K, 0);
            auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
            auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
            // Load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakePtoCoord2D(actualM, kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmTileA, MakeL1ALayout(), layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            // Load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakePtoCoord2D(kActual, actualN));
            copyGmToL1B(l1BTensorList[l1ListId], gmTileB, MakeL1BLayout(), layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // If the number of preload instructions reaches the upper limit, perform an mmad calculation on L1 tile
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Store the current load status
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES) ?
                (l1TileMmadParamsId + preloadCount) : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1ListId = l1ListId;
            l1TileMmadParams.mRound = mRound;
            l1TileMmadParams.nRound = nRound;
            l1TileMmadParams.kActual = kActual;
            l1TileMmadParams.isKLoopFirst = (kLoopIdx == 0);
            l1TileMmadParams.isKLoopLast = (kLoopIdx == kTileCount - 1);
            l1TileMmadParams.flag = flag;
            if (kLoopIdx == kTileCount - 1) {
                l1TileMmadParams.gmBlockC = gmBlockC;
                l1TileMmadParams.gmBlockS = gmBlockS;
                l1TileMmadParams.layoutCInGm = layoutC.GetTileLayout(GetPtoShapeMN(actualShape));
                l1TileMmadParams.layoutScale = layoutScale;
                l1TileMmadParams.syncLoopIdx = syncLoopIdx;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    PTO_DEVICE
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

    PTO_DEVICE
    void Finalize(int32_t target, int32_t flag = 0)
    {
        if (ptrSoftFlagBase_ != nullptr) {
            if (target < 0) {
                return;
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(EVENT_ID0);
            AscendC::GlobalTensor<int32_t> flagGlobal;
            flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ptrSoftFlagBase_) + (expertPerRank_ + AscendC::GetBlockIdx()) * FLAGSTRIDE);
            AscendC::DataCopy(flagGlobal, l1FTensor[target * 16], FLAGSTRIDE);
        }
        else {
            for(;syncGroupIdx <= target; syncGroupIdx++) {
                int32_t flagId = syncGroupIdx / 15 + flag;
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
            }
        }
    }
private:
    struct L1TileMmadParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        bool isKLoopFirst;
        bool isKLoopLast;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        AscendC::GlobalTensor<uint64_t> gmBlockS;
        LayoutC layoutCInGm;
        layout::VectorLayout layoutScale;
        int32_t syncLoopIdx;
        int32_t flag;
        PTO_DEVICE
        L1TileMmadParams() = default;
    };

    PTO_DEVICE
    void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;

        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        uint32_t l1SOffset = l1BOffset + L1B_TILE_SIZE * L1_STAGES;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            l1STensor = resource.l1Buf.template GetBufferByByte<uint64_t>(l1SOffset);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
        }
        if (ptrSoftFlagBase_ != nullptr) {
            // Initialize the flag matrix (structure as below):
            // 1 0 0 0 0 0 0 0
            // 2 0 0 0 0 0 0 0
            // ...
            // 16 0 0 0 0 0 0 0
            // Then move it to L1
            uint32_t l1FOffset = l1SOffset + L1S_TILE_SIZE;
            l1FTensor = resource.l1Buf.template GetBufferByByte<int32_t>(l1FOffset);
            AscendC::GlobalTensor<int32_t> flagBase;
            flagBase.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ptrSoftFlagBase_));
            AscendC::DataCopy(l1FTensor, flagBase, expertPerRank_ * FLAGSTRIDE);
        }
    }

    PTO_DEVICE
    void InitFpBuf(Arch::Resource<ArchTag> &resource, uint32_t FpAddrStart)
    {
        uint32_t FpOffset = FpAddrStart;
        fixpipeBuf = resource.fpBuf.template GetBufferByByte<uint64_t>(FpOffset);
    }

    PTO_DEVICE
    void InitL0A(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    PTO_DEVICE
    void InitL0B(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    PTO_DEVICE
    void InitL0C(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    PTO_DEVICE
    void L1TileMmad(L1TileMmadParams const &params)
    {
        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(params.mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(params.nRound);
        uint32_t kPartLoop = CeilDiv<L0TileShape::K>(params.kActual);
        auto &l1ATensor = l1ATensorList[params.l1ListId];
        auto &l1BTensor = l1BTensorList[params.l1ListId];

        auto &l0CTensor = l0CTensorList[l0CListId];
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakePtoCoord2D(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                L0TileShape::M : (params.mRound - mPartIdx * L0TileShape::M);

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                    L0TileShape::K : (params.kActual - kPartIdx * L0TileShape::K);

                auto &l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                auto l1AOffset = MulPtoCoord2D(MakePtoCoord2D(mPartIdx, kPartIdx), L0TileShape::ToPtoShapeMK());
                auto l1ATile = l1ATensor[MakeL1ALayout().GetOffset(l1AOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, MakeL1ALayout());
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                        L0TileShape::N : (params.nRound - nPartIdx * L0TileShape::N);

                    auto &l0BTile = l0BTensorList[l0BListId];
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                    auto l1BOffset = MulPtoCoord2D(MakePtoCoord2D(kPartIdx, nPartIdx), L0TileShape::ToPtoShapeKN());
                    auto l1BTile = l1BTensor[MakeL1BLayout().GetOffset(l1BOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
                    }
                    copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, MakeL1BLayout());
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    auto l0COffset = MulPtoCoord2D(MakePtoCoord2D(mPartIdx, nPartIdx), L0TileShape::ToPtoShapeMN());
                    auto l0CTile = l0CTensor[layoutCInL0.GetOffset(l0COffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (params.isKLoopLast &&
                            (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    detail::PtoTileMmad<ElementAccumulator, ElementA, ElementB, L0TileShape>(
                        l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
                detail::StagePerChannelScale(
                    copyGmToL1S,
                    copyL1ToFP,
                    l1STensor,
                    fixpipeBuf,
                    params.gmBlockS,
                    params.layoutScale,
                    layoutCInGm.shape(1));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(0);
                AscendC::PipeBarrier<PIPE_FIX>();
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                    params.gmBlockC, l0CTensor, fixpipeBuf, copyL0CToGm, layoutCInGm, layoutCInL0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                    params.gmBlockC, l0CTensor, fixpipeBuf, copyL0CToGm, layoutCInGm, layoutCInL0, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
            }
            #ifdef __TILE_SYNC__
            if (params.flag > 0) {
                int32_t flagId = params.flag + params.syncLoopIdx / 8;
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
            }
            #else
            Finalize(params.syncLoopIdx, params.flag);
            #endif
        }
    }

    AscendC::LocalTensor<uint64_t> fixpipeBuf;

    AscendC::LocalTensor<ElementA> l1ATensorList[L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];
    AscendC::LocalTensor<uint64_t> l1STensor;
    AscendC::LocalTensor<int32_t> l1FTensor;
    int32_t syncGroupIdx;
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyGmToL1S copyGmToL1S;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    CopyL1ToFP copyL1ToFP;

    __gm__ int32_t* ptrSoftFlagBase_ = nullptr;
    int32_t expertPerRank_;
};

}  // namespace pto_ext::Gemm::Block

#endif  // PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP