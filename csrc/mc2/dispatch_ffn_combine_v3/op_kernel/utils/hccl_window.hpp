#ifndef HCCL_WINDOW_HPP
#define HCCL_WINDOW_HPP

#include "kernel_operator.h"
#include "const_args.hpp"
#include "hccl_context.hpp"

#include "pto/comm/pto_comm_inst.hpp"

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t HCCL_WINDOW_MEM = 700 * MB_SIZE;

constexpr uint32_t BARRIER_COUNTER_STRIDE = 16;
constexpr uint32_t BARRIER_EPOCH_INDEX = 2048;
constexpr uint32_t TOKEN_READY_BASE_INDEX = 4096;
constexpr uint32_t TOKEN_READY_STRIDE = 16;

template <typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template <typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

template <typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T *addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

class HcclWindow {
public:
    FORCE_INLINE_AICORE HcclWindow() { segmentBytes_ = HCCL_WINDOW_MEM; }

    FORCE_INLINE_AICORE void InitWindow(GM_ADDR hcclContext)
    {
        hcclCtx_ = reinterpret_cast<__gm__ HcclDeviceContext *>(hcclContext);
        rank_ = static_cast<int32_t>(hcclCtx_->rankId);
        rankSize_ = static_cast<int32_t>(hcclCtx_->rankNum);
        segmentBytes_ = static_cast<size_t>(hcclCtx_->winSize);
    }

    FORCE_INLINE_AICORE GM_ADDR LocalWindowBase() const
    {
        return reinterpret_cast<GM_ADDR>(hcclCtx_->windowsIn[rank_]);
    }

    FORCE_INLINE_AICORE GM_ADDR RankWindowBase(int32_t rankId) const
    {
        return reinterpret_cast<GM_ADDR>(hcclCtx_->windowsIn[rankId]);
    }

    FORCE_INLINE_AICORE GM_ADDR operator()() const
    {
        return LocalWindowBase();
    }

    FORCE_INLINE_AICORE GM_ADDR operator()(int32_t index) const
    {
        return RankWindowBase(index);
    }

    FORCE_INLINE_AICORE GM_ADDR operator()(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= static_cast<int64_t>(segmentBytes_) || rankId < 0 || rankId >= rankSize_) {
            return nullptr;
        }
        return RankWindowBase(rankId) + offset;
    }

    FORCE_INLINE_AICORE size_t SegmentSize() const
    {
        return segmentBytes_;
    }

    FORCE_INLINE_AICORE int32_t RankSize() const
    {
        return rankSize_;
    }

    FORCE_INLINE_AICORE void ResetLocalTokenReady()
    {
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (int i = vec_id; i < rankSize_; i += vec_size) {
            gm_store(LocalTokenReadyCounter(i), 0);
        }
    }

    FORCE_INLINE_AICORE void NotifyRemoteTokenReady(int32_t rankId)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        dsb(DSB_DDR);
        auto remoteTokenReady = RemoteTokenReadySignal(rankId, rank_);
        pto::comm::TNOTIFY(remoteTokenReady, 1, pto::comm::NotifyOp::AtomicAdd);
    }

    FORCE_INLINE_AICORE void WaitTokenReady(int32_t srcRank)
    {
        auto localTokenReady = LocalTokenReadySignal(srcRank);
        pto::comm::TWAIT(localTokenReady, 1, pto::comm::WaitCmp::GE);
    }

    FORCE_INLINE_AICORE void CrossRankSync()
    {
        __gm__ int32_t *sync_base = LocalBarrierEpoch();
        int count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        AscendC::PipeBarrier<PIPE_ALL>();
        dsb(DSB_DDR);
        for (int i = vec_id; i < rankSize_; i += vec_size) {
            auto remoteBarrier = RemoteBarrierSignal(i, rank_);
            auto localBarrier = LocalBarrierSignal(i);
            pto::comm::TNOTIFY(remoteBarrier, 1, pto::comm::NotifyOp::AtomicAdd);
            pto::comm::TWAIT(localBarrier, count, pto::comm::WaitCmp::GE);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }

    FORCE_INLINE_AICORE __gm__ int32_t *SyncBaseAddr()
    {
        return LocalBarrierEpoch();
    }

private:
    FORCE_INLINE_AICORE uint64_t SignalRegionOffsetBytes() const
    {
        return segmentBytes_ - MB_SIZE;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalSignalBase() const
    {
        return reinterpret_cast<__gm__ int32_t *>((*this)() + SignalRegionOffsetBytes());
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteSignalBase(int32_t rankId) const
    {
        return reinterpret_cast<__gm__ int32_t *>((*this)(SignalRegionOffsetBytes(), rankId));
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalBarrierCounter(int32_t srcRank) const
    {
        return LocalSignalBase() + srcRank * BARRIER_COUNTER_STRIDE;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteBarrierCounter(int32_t rankId, int32_t srcRank) const
    {
        return RemoteSignalBase(rankId) + srcRank * BARRIER_COUNTER_STRIDE;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalBarrierEpoch() const
    {
        return LocalSignalBase() + BARRIER_EPOCH_INDEX;
    }

    FORCE_INLINE_AICORE pto::comm::Signal LocalBarrierSignal(int32_t srcRank) const
    {
        return pto::comm::Signal(LocalBarrierCounter(srcRank));
    }

    FORCE_INLINE_AICORE pto::comm::Signal RemoteBarrierSignal(int32_t rankId, int32_t srcRank) const
    {
        return pto::comm::Signal(RemoteBarrierCounter(rankId, srcRank));
    }

    FORCE_INLINE_AICORE __gm__ int32_t *LocalTokenReadyCounter(int32_t srcRank) const
    {
        return LocalSignalBase() + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    FORCE_INLINE_AICORE pto::comm::Signal LocalTokenReadySignal(int32_t srcRank) const
    {
        return pto::comm::Signal(LocalTokenReadyCounter(srcRank));
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteTokenReadyCounter(int32_t rankId, int32_t srcRank) const
    {
        return RemoteSignalBase(rankId) + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    FORCE_INLINE_AICORE pto::comm::Signal RemoteTokenReadySignal(int32_t rankId, int32_t srcRank) const
    {
        return pto::comm::Signal(RemoteTokenReadyCounter(rankId, srcRank));
    }

    __gm__ HcclDeviceContext *hcclCtx_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    size_t segmentBytes_ = 0;
};

#endif
