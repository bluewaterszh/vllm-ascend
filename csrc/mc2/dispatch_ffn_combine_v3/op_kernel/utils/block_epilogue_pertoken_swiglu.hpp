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

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

template <typename Element>
PTO_DEVICE PtoGlobalNd<Element> MakeContiguousGlobal(AscendC::GlobalTensor<Element> const &tensor, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <auto Pipe>
PTO_DEVICE void PtoPipeBarrier()
{
    AscendC::PipeBarrier<Pipe>();
}

template <AscendC::HardEvent Event>
PTO_DEVICE void PtoSetFlag(int32_t eventId)
{
    AscendC::SetFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
PTO_DEVICE void PtoWaitFlag(int32_t eventId)
{
    AscendC::WaitFlag<Event>(eventId);
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoLoadVector(AscendC::LocalTensor<Element> const &dst,
                              AscendC::GlobalTensor<Element> const &src,
                              uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto srcGlobal = MakeContiguousGlobal(srcChunk, cur);
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TLOAD(tile, srcGlobal);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoStoreVector(AscendC::GlobalTensor<Element> const &dst,
                               AscendC::LocalTensor<Element> const &src,
                               uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        auto dstGlobal = MakeContiguousGlobal(dstChunk, cur);
        Tile tile(1, cur);
        pto::TASSIGN(tile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TSTORE(dstGlobal, tile);
    }
}

template <typename DstElement, typename SrcElement, int TileElems = 1024>
PTO_DEVICE void PtoCastVector(AscendC::LocalTensor<DstElement> const &dst,
                              AscendC::LocalTensor<SrcElement> const &src,
                              uint32_t elemNum,
                              pto::RoundMode mode)
{
    using DstTile = PtoVecTile<DstElement, TileElems>;
    using SrcTile = PtoVecTile<SrcElement, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        DstTile dstTile(1, cur);
        SrcTile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TCVT(dstTile, srcTile, mode);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum,
                             Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TMULS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAddScalarVector(AscendC::LocalTensor<Element> const &dst,
                                   AscendC::LocalTensor<Element> const &src,
                                   uint32_t elemNum,
                                   Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TADDS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoMulElementwiseVector(AscendC::LocalTensor<Element> const &dst,
                                        AscendC::LocalTensor<Element> const &src0,
                                        AscendC::LocalTensor<Element> const &src1,
                                        uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto src0Chunk = src0[offset];
        auto src1Chunk = src1[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TMUL(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoDivVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src0,
                             AscendC::LocalTensor<Element> const &src1,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto src0Chunk = src0[offset];
        auto src1Chunk = src1[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TDIV(dstTile, src0Tile, src1Tile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoAbsVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TABS(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_DEVICE void PtoExpVector(AscendC::LocalTensor<Element> const &dst,
                             AscendC::LocalTensor<Element> const &src,
                             uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dstChunk = dst[offset];
        auto srcChunk = src[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TEXP(dstTile, srcTile);
    }
}

template <int TileElems = 1024>
PTO_DEVICE void PtoReduceMaxVector(AscendC::LocalTensor<float> const &dst,
                                   AscendC::LocalTensor<float> const &src,
                                   AscendC::LocalTensor<float> const &tmp,
                                   uint32_t elemNum)
{
    using SrcTile = PtoVecTile<float, TileElems>;
    using TmpTile = PtoVecTile<float, TileElems>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;

    bool firstChunk = true;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto srcChunk = src[offset];
        auto tmpChunk = tmp[offset];
        auto chunkMaxChunk = firstChunk ? dst[0] : tmp[0];

        SrcTile srcTile(1, cur);
        TmpTile tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TASSIGN(tmpTile, reinterpret_cast<uint64_t>(tmpChunk.GetPhyAddr()));
        pto::TASSIGN(rowMaxTile, reinterpret_cast<uint64_t>(chunkMaxChunk.GetPhyAddr()));
        pto::TROWMAX(rowMaxTile, srcTile, tmpTile);
        pto::TSYNC<pto::Op::TROWMAX>();

        if (!firstChunk) {
            auto accChunk = dst[0];
            auto newChunk = tmp[0];
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, reinterpret_cast<uint64_t>(accChunk.GetPhyAddr()));
            pto::TASSIGN(newTile, reinterpret_cast<uint64_t>(newChunk.GetPhyAddr()));
            pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(accChunk.GetPhyAddr()));
            pto::TMAX(dstTile, accTile, newTile);
            pto::TSYNC<pto::Op::TMAX>();
        }
        firstChunk = false;
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

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }

        ubPerTokenScaleOutput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    }
    PTO_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
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
            auto &ubReduceMax = ubCFp32ChunkNMaxList[ubListId];
            auto &ubOutputTmp = ubAbs;
            auto &ubQuantS32 = ubQuantS32List[ubListId];
            auto &ubQuantF16 = ubQuantF16List[ubListId];

            auto gmTileD = gmD[loopIdx * ChunkTileLen];
            // Move C from GM workspace to UB
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            swiglu_detail::PtoLoadVector(ubC, gmTileC, blockN);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // Cast C to FP32 in UB
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            swiglu_detail::PtoCastVector(ubCFp32, ubC, blockN, pto::RoundMode::CAST_NONE);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // Get per-token scale from row loopIdx of gmPerTokenScale
            ElementPerTokenScale perTokenScale = gmPerTokenScale1(loopIdx);

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
            // Multiply FP32 C by the per-token scale
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoMulVector(ubCFp32, ubCFp32, blockN, perTokenScale);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            // Swiglu computation process
            swiglu_detail::PtoMulVector(ubCFp32ChunkN, ubCFp32, ChunkTileLen, -1.0f);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoExpVector(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoAddScalarVector(ubCFp32ChunkN, ubCFp32ChunkN, ChunkTileLen, 1.0f);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoDivVector(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoMulElementwiseVector(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[ChunkTileLen], ChunkTileLen);

            // Quantization process; difference between the two approaches
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            swiglu_detail::PtoAbsVector(ubAbs, ubCFp32ChunkN, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoReduceMaxVector(ubReduceMax, ubAbs, ubAbs, ChunkTileLen);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_S>(0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_S>(0);

            ElementPerTokenScale GMubDequantScale = ubReduceMax.GetValue(0);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_V>(0);

            auto ubPerTokenScaleOutputOffset = loopIdx - loopStartIdx;
            ubPerTokenScaleOutput.SetValue(ubPerTokenScaleOutputOffset, GMubDequantScale / 127.f);

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_V>(0);
            swiglu_detail::PtoMulVector(ubOutputTmp, ubCFp32ChunkN, ChunkTileLen, 127.f / GMubDequantScale);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoCastVector(ubQuantS32, ubOutputTmp, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            swiglu_detail::PtoCastVector(ubQuantF16, ubQuantS32, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoPipeBarrier<PIPE_V>();

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDVMTE3List[ubListId]);
            swiglu_detail::PtoCastVector(ubD, ubQuantF16, ChunkTileLen, pto::RoundMode::CAST_RINT);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::V_MTE3>(eventUbDMTE3VList[ubListId]);

            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            swiglu_detail::PtoStoreVector(gmTileD, ubD, ChunkTileLen);
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }

        if(tasksForIdx > 0){
            swiglu_detail::PtoSetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            swiglu_detail::PtoWaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

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
