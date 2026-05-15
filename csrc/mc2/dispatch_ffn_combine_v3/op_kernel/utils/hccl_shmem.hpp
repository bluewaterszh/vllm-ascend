#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP


#include "kernel_operator.h"
#include "const_args.hpp"

#include "pto/comm/pto_comm_inst.hpp"

#ifdef HCCL_COMM
#include "moe_distribute_base.h"
using namespace AscendC::HcclContextDef;
#else
struct StandaloneHcclDeviceContext {
    static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;

    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];
};
#endif

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

constexpr uint32_t BARRIER_COUNTER_STRIDE = 16;
constexpr uint32_t BARRIER_EPOCH_INDEX = 2048;
constexpr uint32_t TOKEN_READY_BASE_INDEX = 4096;
constexpr uint32_t TOKEN_READY_STRIDE = 16;

template<typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache) {
    return *((__gm__ T *)cache);
}

template<typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE void pto_signal_notify_add(__gm__ int32_t *sig_addr, int32_t value = 1) {
    pto::comm::Signal sig(sig_addr);
    pto::comm::TNOTIFY(sig, value, pto::comm::NotifyOp::AtomicAdd);
}

FORCE_INLINE_AICORE void pto_signal_wait_ge(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    pto::comm::Signal sig(sig_addr);
    pto::comm::TWAIT(sig, cmp_val, pto::comm::WaitCmp::GE);
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    pto_signal_wait_ge(sig_addr, cmp_val);
    return cmp_val;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    pto_signal_wait_ge(sig_addr, cmp_val + 1);
}


class HcclShmem {
public:
    #ifdef HCCL_COMM    // HCCL needs to initialize the HCCL context
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
        AscendC::LocalTensor<int32_t> ub;
        FORCE_INLINE_AICORE
        HcclShmem(){
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
            m_segmentSize = WinContext_->winSize;
        }
    #else
        FORCE_INLINE_AICORE
        HcclShmem(){
            m_segmentSize = SHMEM_MEM;
        }
        FORCE_INLINE_AICORE
        void initShmem(GM_ADDR hcclContext) {
            standaloneCtx_ = reinterpret_cast<__gm__ StandaloneHcclDeviceContext *>(hcclContext);
            m_rank = static_cast<int32_t>(standaloneCtx_->rankId);
            m_rankSize = static_cast<int32_t>(standaloneCtx_->rankNum);
            m_segmentSize = static_cast<size_t>(standaloneCtx_->winSize);
        }
    #endif

    FORCE_INLINE_AICORE
    GM_ADDR LocalWindowBase() const {
        #ifdef HCCL_COMM
            return (GM_ADDR)(WinContext_->localWindowsIn);
        #else
            return reinterpret_cast<GM_ADDR>(standaloneCtx_->windowsIn[m_rank]);
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR RankWindowBase(int32_t rankId) const {
        #ifdef HCCL_COMM
            return (GM_ADDR)((rankId == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);
        #else
            return reinterpret_cast<GM_ADDR>(standaloneCtx_->windowsIn[rankId]);
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator() () const {   // No parameters: return pointer to local peermem
        return LocalWindowBase();
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator() (int32_t index) const {  // With index parameter: return pointer to the base address of remote peermem
        return RankWindowBase(index);
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator () (int64_t offset, int32_t rankId) const  {
        if (offset < 0 || offset >= static_cast<int64_t>(m_segmentSize) || rankId < 0 || rankId >= m_rankSize) {
            return nullptr;
        }
        return RankWindowBase(rankId) + offset;
    }



    FORCE_INLINE_AICORE
    size_t SegmentSize() const {
        return m_segmentSize;
    }

    FORCE_INLINE_AICORE
    int32_t RankSize() const {
        return m_rankSize;
    }

    FORCE_INLINE_AICORE
    void ResetLocalTokenReady() {
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (int i = vec_id; i < m_rankSize; i += vec_size) {
            gm_store(LocalTokenReadyCounter(i), 0);
        }
    }

    FORCE_INLINE_AICORE
    void NotifyRemoteTokenReady(int32_t rankId) {
        AscendC::PipeBarrier<PIPE_ALL>();
        dsb(DSB_DDR);
        pto_signal_notify_add(RemoteTokenReadyCounter(rankId, m_rank));
    }

    FORCE_INLINE_AICORE
    void WaitTokenReady(int32_t srcRank) {
        gm_signal_wait_until_ne(LocalTokenReadyCounter(srcRank), 0);
    }


    FORCE_INLINE_AICORE
    ~HcclShmem() {
    }


    FORCE_INLINE_AICORE
    void CrossRankSync() {
        __gm__ int32_t* sync_base = LocalBarrierEpoch();
        int count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        AscendC::PipeBarrier<PIPE_ALL>();
        dsb(DSB_DDR);
        for(int i = vec_id; i < m_rankSize; i += vec_size) {
            pto_signal_notify_add(RemoteBarrierCounter(i, m_rank));
            auto sync_check = LocalBarrierCounter(i);
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }


    FORCE_INLINE_AICORE
    __gm__ int32_t* SyncBaseAddr() {
        return LocalBarrierEpoch();
    }

private:
    FORCE_INLINE_AICORE
    uint64_t SignalRegionOffsetBytes() const {
        return m_segmentSize - MB_SIZE;
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* LocalSignalBase() const {
        return reinterpret_cast<__gm__ int32_t*>((*this)() + SignalRegionOffsetBytes());
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* RemoteSignalBase(int32_t rankId) const {
        return reinterpret_cast<__gm__ int32_t*>((*this)(SignalRegionOffsetBytes(), rankId));
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* LocalBarrierCounter(int32_t srcRank) const {
        return LocalSignalBase() + srcRank * BARRIER_COUNTER_STRIDE;
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* RemoteBarrierCounter(int32_t rankId, int32_t srcRank) const {
        return RemoteSignalBase(rankId) + srcRank * BARRIER_COUNTER_STRIDE;
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* LocalBarrierEpoch() const {
        return LocalSignalBase() + BARRIER_EPOCH_INDEX;
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* LocalTokenReadyCounter(int32_t srcRank) const {
        return LocalSignalBase() + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t* RemoteTokenReadyCounter(int32_t rankId, int32_t srcRank) const {
        return RemoteSignalBase(rankId) + TOKEN_READY_BASE_INDEX + srcRank * TOKEN_READY_STRIDE;
    }

    #ifndef HCCL_COMM
        __gm__ StandaloneHcclDeviceContext *standaloneCtx_ = nullptr;
    #endif
    int32_t m_rank;
    int32_t m_rankSize;
    size_t m_segmentSize;
};




#endif
