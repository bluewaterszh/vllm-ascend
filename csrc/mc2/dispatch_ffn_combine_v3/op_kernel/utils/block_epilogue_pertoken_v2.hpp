#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "hccl_shmem.hpp"
#include "layout3d.hpp"

namespace Catlass::Epilogue::Block {
namespace detail {

using PtoShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PtoStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoGlobalNd = pto::GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, pto::Layout::ND>;

template <typename Element>
CATLASS_DEVICE PtoGlobalNd<Element> MakeContiguousGlobal(AscendC::GlobalTensor<Element> const &tensor, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element, int TileElems = 128>
CATLASS_DEVICE void PtoLoadVector(AscendC::LocalTensor<Element> const &dst,
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

template <typename Element, int TileElems = 128>
CATLASS_DEVICE void PtoStoreVector(AscendC::GlobalTensor<Element> const &dst,
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

template <typename Element, int TileElems = 128>
CATLASS_DEVICE void PtoLoadMatrixRows(AscendC::LocalTensor<Element> const &dst,
                                      AscendC::GlobalTensor<Element> const &src,
                                      uint32_t rowNum,
                                      uint32_t colNum,
                                      uint32_t dstStride,
                                      uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoLoadVector<Element, TileElems>(dst[rowIdx * dstStride], src[rowIdx * srcStride], colNum);
    }
}

template <typename Element, int TileElems = 128>
CATLASS_DEVICE void PtoStoreMatrixRows(AscendC::GlobalTensor<Element> const &dst,
                                       AscendC::LocalTensor<Element> const &src,
                                       uint32_t rowNum,
                                       uint32_t colNum,
                                       uint32_t dstStride,
                                       uint32_t srcStride)
{
    for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
        PtoStoreVector<Element, TileElems>(dst[rowIdx * dstStride], src[rowIdx * srcStride], colNum);
    }
}

}  // namespace detail

template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileCopy_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileCopy_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem shmem;
        int32_t offsetD;
        int32_t scratchOffset;
        Layout3D tokenPerExpertLayout;
        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_,
        LayoutC layoutC_, int32_t n2_, int32_t n0_, HcclShmem& shmem_, int32_t offsetD_, int32_t scratchOffset_, Layout3D tokenPerExpertLayout_) :
        ptrTokenPerExpert(ptrTokenPerExpert_), EP(EP_),
        expertPerRank(expertPerRank_),rank(rank_), layoutC(layoutC_), n2(n2_), n0(n0_),
        shmem(shmem_), offsetD(offsetD_), scratchOffset(scratchOffset_), tokenPerExpertLayout(tokenPerExpertLayout_)
         {}
    };


    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        //ub:192KB
        n0 = params.n0;
        size_t ubOffset = 0;
        for(int32_t i = 0; i < 2; i++) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += max_len * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);
            ubFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += max_len * sizeof(float);
            scaleUbList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        tokenPerExpertLayout = params.tokenPerExpertLayout;
        is_ping = true;
    }
    CATLASS_DEVICE
    void SetFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void Finalize()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        
    }
    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
        GemmCoord& blockCoord,
        GemmCoord& actualBlockShape,
        int32_t groupIdx,
        int32_t preSrcExpertSum,
        AscendC::GlobalTensor<int32_t> preSumBeforeRank
    ){
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

        auto &ubC = ubCList[is_ping];
        auto &ubD = ubDList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];
        auto &ubCFp32 = ubFp32List[is_ping];
        auto &scaleUb = scaleUbList[is_ping];

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        detail::PtoLoadMatrixRows(ubC, gmTileC, actualBlockShape.m(), actualBlockShape.n(), n0, params.n2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);


        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockCoord.m();
        if (source_scale_offset[event_id] != gmScaleOffset) {
                source_scale_offset[event_id] = gmScaleOffset;
                detail::PtoLoadVector(scaleUb, gmPerTokenScale[gmScaleOffset], actualBlockShape.m());
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

        

        
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2); // Note that the value must be MTE2_S instead of MTE2_V.
                                                                   // Otherwise, 0 will be read, causing garbled characters.
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < actualBlockShape.m(); ++row) {
                float scale = scaleUb(row);
                Muls<float, false>(ubCFp32[n0* row], ubCFp32[n0 * row] , scale, -1, (actualBlockShape.n() + 127) / 128 * 2, {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx ++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            }
            else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0){
                continue;
            }
            
            uint32_t dstOffsetInExpert = 0;
            if (stTile > stRankInExpert) {
                dstOffsetInExpert = stTile - stRankInExpert;
            }
            AscendC::GlobalTensor<ElementD> gmRemotePeer;
            __gm__ void* dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
            gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            if (dstEpIdx == params.rank) {
                detail::PtoStoreMatrixRows(gmTileD, ubD[tileOffset * n0], lenData, actualBlockShape.n(), params.n2, n0);
            } else {
                using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
                using TputGlobal = pto::GlobalTensor<ElementD, ShapeDyn, StrideDyn, pto::Layout::ND>;
                using TputTile = pto::Tile<pto::TileType::Vec, ElementD, 1, 128, pto::BLayout::RowMajor, -1, -1>;

                int32_t logicalSubCoreIdx = get_block_idx() + get_subblockid() * get_block_num();
                int64_t scratchOffsetBytes = params.scratchOffset + static_cast<int64_t>(logicalSubCoreIdx) * n0 * sizeof(ElementD);
                __gm__ ElementD* localScratch = reinterpret_cast<__gm__ ElementD*>(params.shmem(scratchOffsetBytes, params.rank));
                AscendC::GlobalTensor<ElementD> gmLocalScratch;
                gmLocalScratch.SetGlobalBuffer(localScratch);
                LayoutC layoutScratchRow{1, actualBlockShape.n(), actualBlockShape.n()};
                LayoutC layoutUbRow{1, actualBlockShape.n(), n0};
                ShapeDyn rowShape(1, 1, 1, 1, actualBlockShape.n());
                StrideDyn localStride(actualBlockShape.n(), actualBlockShape.n(), actualBlockShape.n(), actualBlockShape.n(), 1);
                StrideDyn remoteStride(params.n2, params.n2, params.n2, params.n2, 1);
                TputTile tputTile(1, actualBlockShape.n());
                __gm__ ElementD* remotePeerBase = reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr) + gmDstOffset;

                for (uint32_t rowIdx = 0; rowIdx < lenData; ++rowIdx) {
                    detail::PtoStoreVector(gmLocalScratch[0], ubD[(tileOffset + rowIdx) * n0], actualBlockShape.n());
                    TputGlobal localRowG(localScratch, rowShape, localStride);
                    TputGlobal remoteRowG(remotePeerBase + rowIdx * params.n2, rowShape, remoteStride);
                    pto::comm::TPUT(remoteRowG, localRowG, tputTile);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            tileOffset += lenData;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);

    }
private:

    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> scaleUbList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    
    int32_t repeat = 128;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;
};
}
#endif