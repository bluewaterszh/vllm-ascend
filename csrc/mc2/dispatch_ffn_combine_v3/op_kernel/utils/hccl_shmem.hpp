#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP

#include "kernel_operator.h"
#include "const_args.hpp"
#include "hccl_context.hpp"

#include "pto/comm/pto_comm_inst.hpp"

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

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

FORCE_INLINE_AICORE void pto_signal_notify_add(__gm__ int32_t *sig_addr, int32_t value = 1)
{
    pto::comm::Signal sig(sig_addr);
    pto::comm::TNOTIFY(sig, value, pto::comm::NotifyOp::AtomicAdd);
}

FORCE_INLINE_AICORE void pto_signal_wait_ge(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    pto::comm::Signal sig(sig_addr);
    pto::comm::TWAIT(sig, cmp_val, pto::comm::WaitCmp::GE);
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    pto_signal_wait_ge(sig_addr, cmp_val);
    return cmp_val;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    pto_signal_wait_ge(sig_addr, cmp_val + 1);
}

class HcclShmem {
public:
    FORCE_INLINE_AICORE HcclShmem() { segmentBytes_ = SHMEM_MEM; }

    FORCE_INLINE_AICORE void initShmem(GM_ADDR hcclContext)
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
        pto_signal_notify_add(RemoteTokenReadyCounter(rankId, rank_));
    }

    FORCE_INLINE_AICORE void WaitTokenReady(int32_t srcRank)
    {
        gm_signal_wait_until_ne(LocalTokenReadyCounter(srcRank), 0);
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
            pto_signal_notify_add(RemoteBarrierCounter(i, rank_));
            auto sync_check = LocalBarrierCounter(i);
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
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

    FORCE_INLINE_AICORE __gm__ int32_t *LocalTokenReadyCounter(int32_t srcRank) const
    {
        return LocalSignalBase() + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    FORCE_INLINE_AICORE __gm__ int32_t *RemoteTokenReadyCounter(int32_t rankId, int32_t srcRank) const
    {
        return RemoteSignalBase(rankId) + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    __gm__ HcclDeviceContext *hcclCtx_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    size_t segmentBytes_ = 0;
};

#endif
