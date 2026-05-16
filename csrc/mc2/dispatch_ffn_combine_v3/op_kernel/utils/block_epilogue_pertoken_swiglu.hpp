/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP
#define PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP

#include "dispatch_policy_custom.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

namespace pto_ext::Epilogue::Block {
namespace swiglu_detail {

using PtoShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PtoStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoGlobalNd = pto::GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, pto::Layout::ND>;

template <typename Element>
PTO_DEVICE PtoGlobalNd<Element> MakeContiguousGlobal(AscendC::GlobalTensor<Element> const &tensor, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadVector(AscendC::LocalTensor<Element> const &dst,
                                  AscendC::GlobalTensor<Element> const &src,
                                  uint32_t elemNum)
{
    using PtoTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto srcGlobal = MakeContiguousGlobal(srcChunk, cur);
        PtoTile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TLOAD(tile, srcGlobal);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreVector(AscendC::GlobalTensor<Element> const &dst,
                                   AscendC::LocalTensor<Element> const &src,
                                   uint32_t elemNum)
{
    using PtoTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto dstGlobal = MakeContiguousGlobal(dstChunk, cur);
        PtoTile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TSTORE(dstGlobal, tile);
    }
}

}  // namespace swiglu_detail


// float scale, dequant per expert
template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileElemWiseMuls_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileElemWiseMuls_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(
        std::is_same_v<ElementC, half> && (std::is_same_v<ElementD, float> || std::is_same_v<ElementD, int8_t>),
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::ND> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::ND>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        PTO_DEVICE
        Params() {};

        PTO_DEVICE
        Params(__gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            __gm__ ElementD *ptrD_, LayoutD const &layoutD_
        ) : ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(ptrD_), layoutD(layoutD_) {}
    };

    PTO_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, int32_t n, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        uint32_t blockN = n;
        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += blockN * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += blockN * sizeof(ElementD);
            ubCFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += blockN * sizeof(float);
            ubCFp32ChunkNList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNAbsList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += ChunkTileLen * sizeof(float);
            ubCFp32ChunkNMaxList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += HalfChunkTileLen * sizeof(float);
            ubQuantS32List[i] = ubCFp32ChunkNAbsList[i].template ReinterpretCast<int32_t>();
            ubQuantF16List[i] = ubCFp32ChunkNAbsList[i].template ReinterpretCast<half>();

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

        ubPerTokenScaleOutput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    }
    PTO_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
    PTO_DEVICE
    ~BlockEpilogue()
    {
    }

    PTO_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }
    // Each tile is 1x7168, and each block covers all tokens for one expert = [group[i], 7168]
    template <typename CallbackT = pto_ext::support::NoopCallback>
    PTO_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        PtoShape2D const &shapeC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale1,
        AscendC::GlobalTensor<ElementD> const &gmD,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale2,

        uint32_t epilogueCoreNum = 40,
        CallbackT &&callback = CallbackT{}
    )
    {
        callback();
        uint32_t blockM = static_cast<uint32_t>(shapeC.shape[0]);
        uint32_t blockN = static_cast<uint32_t>(shapeC.shape[1]);

        uint32_t tileLoops = blockM;
        uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();

        uint32_t subblockNum = get_block_num() * 2;
        uint32_t moveDataCoreNum = subblockNum - epilogueCoreNum;

        if (subblockIdx < moveDataCoreNum) {
            return;
        }
        uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;

        uint32_t perCoreData =  blockM / epilogueCoreNum;
        uint32_t remainderData = blockM % epilogueCoreNum;

        uint32_t tasksForIdx  = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
        uint32_t loopStartIdx = epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData? epilogueCoreIdx : remainderData);

        uint32_t alignedPerCoreData = RoundUp<BYTE_PER_BLK / sizeof(ElementPerTokenScale)>(perCoreData + 1);

        uint32_t ChunkTileLen = blockN / 2;
        uint32_t HalfChunkTileLen = ChunkTileLen / 2;


        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {

            auto gmTileC = gmC[loopIdx * blockN];

            auto &ubC = ubCList[ubListId];
            auto &ubD = ubDList[ubListId];

            auto &ubCFp32 = ubCFp32List[ubListId];
            auto &ubCFp32ChunkN = ubCFp32ChunkNList[ubListId];
            auto &ubAbs = ubCFp32ChunkNAbsList[ubListId];
            // auto &ubMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubReduceMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubOutputTmp = ubAbs;
            auto &sharedUbTmpBuffer = ubReduceMax;
            auto &ubQuantS32 = ubQuantS32List[ubListId];
            auto &ubQuantF16 = ubQuantF16List[ubListId];

            auto gmTileD = gmD[loopIdx * ChunkTileLen];
            // Move C from GM workspace to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            swiglu_detail::PtoLoadVector(ubC, gmTileC, blockN);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // Cast C to FP32 in UB
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, blockN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // Get per-token scale from row loopIdx of gmPerTokenScale
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            // Multiply FP32 C by the per-token scale
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, blockN);
            AscendC::PipeBarrier<PIPE_V>();

            // Swiglu computation process
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            // TODO: confirm whether the division impacts subsequent data
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);

            // Quantization process; difference between the two approaches
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubAbs, ubCFp32ChunkN, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceMax<float>(ubReduceMax, ubAbs, sharedUbTmpBuffer, ChunkTileLen, false);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            // TODO: compare the efficiency of the two calculation methods
            ElementPerTokenScale GMubDequantScale = ubReduceMax.GetValue(0);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);

            auto ubPerTokenScaleOutputOffset = loopIdx - loopStartIdx;
            ubPerTokenScaleOutput.SetValue(ubPerTokenScaleOutputOffset, GMubDequantScale / 127.f);

            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::Muls(ubOutputTmp, ubCFp32ChunkN, 127.f / GMubDequantScale, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubQuantS32, ubOutputTmp, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDVMTE3List[ubListId]);
            AscendC::Cast(ubD, ubQuantF16, AscendC::RoundMode::CAST_RINT, ChunkTileLen);
            // AscendC::Muls(ubD, ubCFp32ChunkN, 127.f / GMubDequantScale, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDMTE3VList[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            swiglu_detail::PtoStoreVector(gmTileD, ubD, ChunkTileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if(tasksForIdx > 0){
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

            swiglu_detail::PtoStoreVector(gmPerTokenScale2[loopStartIdx], ubPerTokenScaleOutput[0], tasksForIdx);
        }


    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNList[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNAbsList[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNMaxList[UB_STAGES];
    AscendC::LocalTensor<int32_t> ubQuantS32List[UB_STAGES];
    AscendC::LocalTensor<half> ubQuantF16List[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleOutput;

};

}  // namespace pto_ext::Epilogue::Block

#endif  // PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_SWIGLU_HPP
