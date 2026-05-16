#ifndef INNER_MOE_V2_PTO_SORT_H
#define INNER_MOE_V2_PTO_SORT_H

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "moe_v2_common.h"

namespace MoeInitRoutingQuantV2 {
namespace pto_detail {

constexpr uint32_t PTO_SORT_BLOCK_ELEMS = 32;
constexpr uint32_t PTO_PACKED_SORT_BLOCK_ELEMS = 64;
constexpr uint32_t MAX_V3_SORT_ELEMS = 8192;
constexpr float PTO_SORT_NEG_INF = -3.4028235e38F;

using PtoV2SortKeyTile = pto::Tile<pto::TileType::Vec, float, 1, MAX_V3_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoV2SortPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_V3_SORT_ELEMS, pto::BLayout::RowMajor, -1, -1>;
using PtoV2PackedSortTile =
    pto::Tile<pto::TileType::Vec, float, 1, MAX_V3_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;
using PtoV2PackedPayloadTile =
    pto::Tile<pto::TileType::Vec, uint32_t, 1, MAX_V3_SORT_ELEMS * 2, pto::BLayout::RowMajor, -1, -1>;

template <typename Element, int TileElems = 1024>
using PtoV2VecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

template <typename Element>
using PtoV2ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoV2StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoV2GlobalNd = pto::GlobalTensor<Element, PtoV2ShapeDyn<Element>, PtoV2StrideDyn<Element>, pto::Layout::ND>;

template <typename Element>
PTO_INTERNAL PtoV2GlobalNd<Element> MakeContiguousGlobal(const GlobalTensor<Element> &tensor, uint32_t elemNum)
{
    PtoV2ShapeDyn<Element> shape(1, 1, 1, 1, elemNum);
    PtoV2StrideDyn<Element> stride(elemNum, elemNum, elemNum, elemNum, 1);
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return PtoV2GlobalNd<Element>(ptr, shape, stride);
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoMoveVector(const LocalTensor<Element> &dstLocal, const LocalTensor<Element> &srcLocal, uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcLocal[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TMOV(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoMulVector(const LocalTensor<Element> &dstLocal,
                               const LocalTensor<Element> &srcLocal,
                               uint32_t elemNum,
                               Element scalar)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcLocal[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TMULS(dstTile, srcTile, scalar);
    }
}

template <typename DstElement, typename SrcElement, int TileElems = 1024>
PTO_INTERNAL void PtoCastVector(const LocalTensor<DstElement> &dstLocal,
                                const LocalTensor<SrcElement> &srcLocal,
                                uint32_t elemNum,
                                pto::RoundMode mode)
{
    using DstTile = PtoV2VecTile<DstElement, TileElems>;
    using SrcTile = PtoV2VecTile<SrcElement, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcLocal[offset];
        DstTile dstTile(1, cur);
        SrcTile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TCVT(dstTile, srcTile, mode);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoLoadVector(const LocalTensor<Element> &dstLocal,
                                const GlobalTensor<Element> &srcGlobalTensor,
                                uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcGlobalTensor[offset];
        auto srcGlobal = MakeContiguousGlobal(srcChunk, cur);
        Tile dstTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TLOAD(dstTile, srcGlobal);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoStoreVector(const GlobalTensor<Element> &dstGlobalTensor,
                                 const LocalTensor<Element> &srcLocal,
                                 uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstGlobalTensor[offset];
        auto srcChunk = srcLocal[offset];
        auto dstGlobal = MakeContiguousGlobal(dstChunk, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TSTORE(dstGlobal, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoAddScalarVector(const LocalTensor<Element> &dstLocal,
                                     const LocalTensor<Element> &srcLocal,
                                     uint32_t elemNum,
                                     Element scalar)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcLocal[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TADDS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoMulElementwiseVector(const LocalTensor<Element> &dstLocal,
                                          const LocalTensor<Element> &src0Local,
                                          const LocalTensor<Element> &src1Local,
                                          uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto src0Chunk = src0Local[offset];
        auto src1Chunk = src1Local[offset];
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
PTO_INTERNAL void PtoAbsVector(const LocalTensor<Element> &dstLocal,
                               const LocalTensor<Element> &srcLocal,
                               uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto srcChunk = srcLocal[offset];
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcChunk.GetPhyAddr()));
        pto::TABS(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
PTO_INTERNAL void PtoDivVector(const LocalTensor<Element> &dstLocal,
                               const LocalTensor<Element> &src0Local,
                               const LocalTensor<Element> &src1Local,
                               uint32_t elemNum)
{
    using Tile = PtoV2VecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset < static_cast<uint32_t>(TileElems))
                                 ? (elemNum - offset)
                                 : static_cast<uint32_t>(TileElems);
        auto dstChunk = dstLocal[offset];
        auto src0Chunk = src0Local[offset];
        auto src1Chunk = src1Local[offset];
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstChunk.GetPhyAddr()));
        pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Chunk.GetPhyAddr()));
        pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Chunk.GetPhyAddr()));
        pto::TDIV(dstTile, src0Tile, src1Tile);
    }
}

PTO_INTERNAL uint32_t AlignUpSortBlock(uint32_t elemNum)
{
    return ((elemNum + PTO_SORT_BLOCK_ELEMS - 1) / PTO_SORT_BLOCK_ELEMS) * PTO_SORT_BLOCK_ELEMS;
}

PTO_INTERNAL int32_t FillTailMergeArray(int32_t *mrgArray, int32_t validCols, int32_t blockLen)
{
    int32_t arrayCount = 0;
    int32_t remainCols = validCols;
    for (int32_t curBlockLen = blockLen; curBlockLen >= static_cast<int32_t>(PTO_PACKED_SORT_BLOCK_ELEMS);
         curBlockLen /= 4) {
        int32_t count = 0;
        for (; count < remainCols / curBlockLen; ++count) {
            mrgArray[arrayCount++] = curBlockLen;
        }
        remainCols -= count * curBlockLen;
    }
    return arrayCount;
}

PTO_INTERNAL void MergeTailPackedSortRecords(PtoV2PackedSortTile &packedSortTile,
                                             PtoV2PackedSortTile &mergeTmpTile,
                                             uint32_t validCols,
                                             uint32_t blockLen)
{
    int32_t mergePlan[15] = {0};
    const int32_t mergePlanCount =
        FillTailMergeArray(mergePlan, static_cast<int32_t>(validCols), static_cast<int32_t>(blockLen));
    if (mergePlanCount <= 1) {
        return;
    }

    pto::MrgSortExecutedNumList executedNumList{};
    uint16_t mergedCols = 0;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (int32_t i = 0; i < mergePlanCount - 1; ++i) {
        mergedCols += static_cast<uint16_t>(mergePlan[i]);
        PtoV2PackedSortTile src0Tile(1, mergedCols);
        PtoV2PackedSortTile src1Tile(1, static_cast<uint16_t>(mergePlan[i + 1]));
        PtoV2PackedSortTile dstTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        PtoV2PackedSortTile tmpTile(1, mergedCols + static_cast<uint16_t>(mergePlan[i + 1]));
        pto::TASSIGN(src0Tile, packedAddr);
        pto::TASSIGN(src1Tile, packedAddr + static_cast<uint64_t>(mergedCols) * sizeof(float));
        pto::TASSIGN(dstTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, false>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
    }
}

PTO_INTERNAL void MergePackedSortRecords(PtoV2PackedSortTile &packedSortTile,
                                         PtoV2PackedSortTile &mergeTmpTile,
                                         uint32_t validCols)
{
    uint32_t blockLen = PTO_PACKED_SORT_BLOCK_ELEMS;
    const uint64_t packedAddr = reinterpret_cast<uint64_t>(packedSortTile.data());
    const uint64_t tmpAddr = reinterpret_cast<uint64_t>(mergeTmpTile.data());
    for (; blockLen * 4 <= validCols; blockLen *= 4) {
        const uint16_t cols = validCols / (blockLen * 4) * (blockLen * 4);
        PtoV2PackedSortTile srcTile(1, cols);
        PtoV2PackedSortTile tmpTile(1, cols);
        pto::TASSIGN(srcTile, packedAddr);
        pto::TASSIGN(tmpTile, tmpAddr);
        pto::TMRGSORT(tmpTile, srcTile, blockLen);
        pipe_barrier(PIPE_V);
        pto::TMOV(srcTile, tmpTile);
        pipe_barrier(PIPE_V);
    }

    if (blockLen < validCols) {
        PtoV2PackedSortTile tailTile(1, validCols);
        PtoV2PackedSortTile tailTmpTile(1, validCols);
        pto::TASSIGN(tailTile, packedAddr);
        pto::TASSIGN(tailTmpTile, tmpAddr);
        MergeTailPackedSortRecords(tailTile, tailTmpTile, validCols, blockLen);
    }
}

PTO_INTERNAL void PtoMergePackedSortRecords(LocalTensor<float> &dstLocal,
                                            LocalTensor<float> &tmpLocal,
                                            LocalTensor<float> &src0Local,
                                            LocalTensor<float> &src1Local,
                                            LocalTensor<float> &src2Local,
                                            LocalTensor<float> &src3Local,
                                            const uint16_t *elementCountList,
                                            uint32_t remainListNum,
                                            uint32_t *listSortedNums)
{
    const uint32_t src0Cols = GetSortLen<float>(elementCountList[0]);
    const uint32_t src1Cols = (remainListNum >= 2) ? GetSortLen<float>(elementCountList[1]) : 0;
    const uint32_t src2Cols = (remainListNum >= 3) ? GetSortLen<float>(elementCountList[2]) : 0;
    const uint32_t src3Cols = (remainListNum >= 4) ? GetSortLen<float>(elementCountList[3]) : 0;
    const uint32_t dstCols = src0Cols + src1Cols + src2Cols + src3Cols;
    ASCENDC_ASSERT((dstCols <= MAX_V3_SORT_ELEMS * 2), {
        KERNEL_LOG(KERNEL_ERROR, "dstCols exceeds PTO merge capacity");
    });

    PtoV2PackedSortTile dstTile(1, dstCols);
    PtoV2PackedSortTile tmpTile(1, dstCols);
    PtoV2PackedSortTile src0Tile(1, src0Cols);
    PtoV2PackedSortTile src1Tile(1, src1Cols);
    PtoV2PackedSortTile src2Tile(1, src2Cols);
    PtoV2PackedSortTile src3Tile(1, src3Cols);
    pto::TASSIGN(dstTile, reinterpret_cast<uint64_t>(dstLocal.GetPhyAddr()));
    pto::TASSIGN(tmpTile, reinterpret_cast<uint64_t>(tmpLocal.GetPhyAddr()));
    pto::TASSIGN(src0Tile, reinterpret_cast<uint64_t>(src0Local.GetPhyAddr()));
    pto::TASSIGN(src1Tile, reinterpret_cast<uint64_t>(src1Local.GetPhyAddr()));
    pto::TASSIGN(src2Tile, reinterpret_cast<uint64_t>(src2Local.GetPhyAddr()));
    pto::TASSIGN(src3Tile, reinterpret_cast<uint64_t>(src3Local.GetPhyAddr()));

    pto::MrgSortExecutedNumList executedNumList{};
    if (remainListNum == MERGE_LIST_TWO) {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, true>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    } else if (remainListNum == MERGE_LIST_THREE) {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile,
                      PtoV2PackedSortTile, true>(dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        pto::TMRGSORT<PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile, PtoV2PackedSortTile,
                      PtoV2PackedSortTile, PtoV2PackedSortTile, true>(dstTile,
                                                                      executedNumList,
                                                                      tmpTile,
                                                                      src0Tile,
                                                                      src1Tile,
                                                                      src2Tile,
                                                                      src3Tile);
    }

    listSortedNums[0] = executedNumList.mrgSortList0;
    listSortedNums[1] = executedNumList.mrgSortList1;
    listSortedNums[2] = executedNumList.mrgSortList2;
    listSortedNums[3] = executedNumList.mrgSortList3;
}

PTO_INTERNAL void PtoSortInt32ToPackedUB(LocalTensor<int32_t> &inputValueLocal,
                                        LocalTensor<uint32_t> &inputPayloadLocal,
                                        LocalTensor<float> &packedSortLocal,
                                        LocalTensor<float> &mergeTmpLocal,
                                        uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }

    const uint32_t alignedElemNum = AlignUpSortBlock(elemNum);
    ASCENDC_ASSERT((alignedElemNum <= MAX_V3_SORT_ELEMS), {
        KERNEL_LOG(KERNEL_ERROR, "alignedElemNum exceeds PTO sort capacity");
    });

    LocalTensor<float> sortKeyLocal = mergeTmpLocal;
    PtoCastVector(sortKeyLocal, inputValueLocal, elemNum, pto::RoundMode::CAST_CEIL);
    PtoMulVector(sortKeyLocal, sortKeyLocal, elemNum, -1.0F);

    __ubuf__ float *sortKeyPtr = reinterpret_cast<__ubuf__ float *>(sortKeyLocal.GetPhyAddr());
    __ubuf__ uint32_t *payloadPtr = reinterpret_cast<__ubuf__ uint32_t *>(inputPayloadLocal.GetPhyAddr());
    for (uint32_t i = elemNum; i < alignedElemNum; ++i) {
        sortKeyPtr[i] = PTO_SORT_NEG_INF;
        payloadPtr[i] = 0;
    }

    PtoV2SortKeyTile srcTile(1, alignedElemNum);
    PtoV2SortPayloadTile payloadTile(1, alignedElemNum);
    PtoV2PackedSortTile packedTile(1, alignedElemNum * 2);
    PtoV2PackedSortTile mergeTmpTile(1, alignedElemNum * 2);
    pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(sortKeyLocal.GetPhyAddr()));
    pto::TASSIGN(payloadTile, reinterpret_cast<uint64_t>(inputPayloadLocal.GetPhyAddr()));
    pto::TASSIGN(packedTile, reinterpret_cast<uint64_t>(packedSortLocal.GetPhyAddr()));
    pto::TASSIGN(mergeTmpTile, reinterpret_cast<uint64_t>(mergeTmpLocal.GetPhyAddr()));

    pto::TSORT32(packedTile, srcTile, payloadTile);
    pipe_barrier(PIPE_V);
    MergePackedSortRecords(packedTile, mergeTmpTile, alignedElemNum * 2);
}

PTO_INTERNAL void PtoExtractPackedSortResult(LocalTensor<int32_t> &sortedValueLocal,
                                             LocalTensor<uint32_t> &sortedPayloadLocal,
                                             LocalTensor<float> &packedSortLocal,
                                             uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (elemNum == 1) {
        __ubuf__ const float *packedPtr = reinterpret_cast<__ubuf__ const float *>(packedSortLocal.GetPhyAddr());
        __ubuf__ int32_t *valueOut = reinterpret_cast<__ubuf__ int32_t *>(sortedValueLocal.GetPhyAddr());
        __ubuf__ uint32_t *payloadOut = reinterpret_cast<__ubuf__ uint32_t *>(sortedPayloadLocal.GetPhyAddr());
        __ubuf__ const uint32_t *packedPayloadPtr = reinterpret_cast<__ubuf__ const uint32_t *>(packedSortLocal.GetPhyAddr());
        valueOut[0] = -static_cast<int32_t>(packedPtr[0]);
        payloadOut[0] = packedPayloadPtr[1];
        return;
    }

    LocalTensor<float> sortedValueScratchLocal = sortedValueLocal.ReinterpretCast<float>();
    PtoV2PackedPayloadTile packedPayloadTile(1, elemNum * 2);
    PtoV2SortPayloadTile sortedPayloadTile(1, elemNum);
    pto::TASSIGN(packedPayloadTile, reinterpret_cast<uint64_t>(packedSortLocal.GetPhyAddr()));
    pto::TASSIGN(sortedPayloadTile, reinterpret_cast<uint64_t>(sortedPayloadLocal.GetPhyAddr()));
    pto::TGATHER<PtoV2SortPayloadTile, PtoV2PackedPayloadTile, pto::MaskPattern::P1010>(sortedPayloadTile,
                                                                                          packedPayloadTile);
    pipe_barrier(PIPE_V);

    PtoV2SortKeyTile sortedKeyTile(1, elemNum);
    PtoV2PackedSortTile packedTile(1, elemNum * 2);
    pto::TASSIGN(sortedKeyTile, reinterpret_cast<uint64_t>(sortedValueScratchLocal.GetPhyAddr()));
    pto::TASSIGN(packedTile, reinterpret_cast<uint64_t>(packedSortLocal.GetPhyAddr()));
    pto::TGATHER<PtoV2SortKeyTile, PtoV2PackedSortTile, pto::MaskPattern::P0101>(sortedKeyTile, packedTile);
    pipe_barrier(PIPE_V);
    PtoMulVector(sortedValueScratchLocal, sortedValueScratchLocal, elemNum, -1.0F);

    PtoCastVector(sortedValueLocal, sortedValueScratchLocal, elemNum, pto::RoundMode::CAST_CEIL);
}

PTO_INTERNAL void PtoSortInt32AscendingUB(LocalTensor<int32_t> &inputValueLocal,
                                          LocalTensor<uint32_t> &inputPayloadLocal,
                                          LocalTensor<int32_t> &sortedValueLocal,
                                          LocalTensor<uint32_t> &sortedPayloadLocal,
                                          LocalTensor<float> &packedSortLocal,
                                          LocalTensor<float> &mergeTmpLocal,
                                          uint32_t elemNum)
{
    if (elemNum == 0) {
        return;
    }
    if (elemNum == 1) {
        __ubuf__ const int32_t *valueIn = reinterpret_cast<__ubuf__ const int32_t *>(inputValueLocal.GetPhyAddr());
        __ubuf__ const uint32_t *payloadIn = reinterpret_cast<__ubuf__ const uint32_t *>(inputPayloadLocal.GetPhyAddr());
        __ubuf__ int32_t *valueOut = reinterpret_cast<__ubuf__ int32_t *>(sortedValueLocal.GetPhyAddr());
        __ubuf__ uint32_t *payloadOut = reinterpret_cast<__ubuf__ uint32_t *>(sortedPayloadLocal.GetPhyAddr());
        valueOut[0] = valueIn[0];
        payloadOut[0] = payloadIn[0];
        return;
    }

    PtoSortInt32ToPackedUB(inputValueLocal, inputPayloadLocal, packedSortLocal, mergeTmpLocal, elemNum);
    PtoExtractPackedSortResult(sortedValueLocal, sortedPayloadLocal, packedSortLocal, elemNum);
}

}  // namespace pto_detail
}  // namespace MoeInitRoutingQuantV2

#endif  // INNER_MOE_V2_PTO_SORT_H
