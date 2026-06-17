/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dispatch_ffn_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine.h"
#endif
#include "../kernel_launch.hpp"

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
using namespace AscendC;
using namespace DispatchFFNCombineImpl;
extern "C" __global__ __aicore__ void dispatch_ffn_combine(GM_ADDR x, GM_ADDR w1, GM_ADDR w2,  GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs,
    GM_ADDR xActiveMask, GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,  GM_ADDR tilingGM, GM_ADDR profileGM,
    uint32_t stageProfile, uint32_t startSyncDebug)
{
    __gm__ uint64_t *profileEntry = nullptr;
    if (profileGM != nullptr) {
        uint64_t profileOffset =
            static_cast<uint64_t>(get_block_idx()) * DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK;
#if defined(__DAV_VEC__)
        profileOffset +=
            (static_cast<uint64_t>(get_subblockid()) + 1U) * DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES;
#endif
        profileEntry =
            reinterpret_cast<__gm__ uint64_t *>(reinterpret_cast<__gm__ uint8_t *>(profileGM) + profileOffset);
    }

    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    __gm__ DispatchFFNCombineTilingData *tilingData =
        reinterpret_cast<__gm__ DispatchFFNCombineTilingData *>(tilingGM);
    if (TILING_KEY_IS(DISPATCH_FFN_COMBINE_DEVICE_TILING_KEY)) {
        KERNEL_TASK_TYPE(DISPATCH_FFN_COMBINE_DEVICE_TILING_KEY, KERNEL_TYPE_MIX_AIC_1_2);
        if (startSyncDebug != 0U) {
#ifdef HCCL_COMM
            if (tilingData->runtimeInfo.symmetricPtr != 0) {
                AscendC::SetHcclContext<HCCL_GROUP_ID_0>(
                    reinterpret_cast<__gm__ uint8_t *>(tilingData->runtimeInfo.symmetricPtr));
            }
            HcclShmem startSyncShmem;
            startSyncShmem.initHccl(tilingData);
#else
            HcclShmem startSyncShmem;
            startSyncShmem.initShmem(static_cast<GM_ADDR>(tilingData->runtimeInfo.symmetricPtr),
                                     tilingData->runtimeInfo.rank, tilingData->runtimeInfo.rankSize);
#endif
            startSyncShmem.CrossRankStartSyncAiv();
            startSyncShmem.CrossRankStartSyncAic();
            pipe_barrier(PIPE_ALL);
        }
        uint64_t tStart = get_sys_cnt();
        if (profileEntry != nullptr) {
            profileEntry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START] = tStart;
        }
        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM,
                tilingGM, stageProfile != 0U ? reinterpret_cast<GM_ADDR>(profileEntry) : nullptr);
        op.Process();

        pipe_barrier(PIPE_ALL);
        uint64_t tEnd = get_sys_cnt();
        if (profileEntry != nullptr) {
            profileEntry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END] = tEnd;
        }
    }
}
#endif

#if defined(__CCE_KT_TEST__) || !defined(__CCE_AICORE__)
extern "C" uint32_t GetAscendCoreSyncAddr(void **addr);
extern "C" uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
extern "C" uint32_t FreeAscendMemDevice(void *devMem);
uint32_t launch_and_profiling_dispatch_ffn_combine(uint64_t func_key, uint32_t blockDim, void *stream,
                                                   void **args, uint32_t size);
#endif

uint32_t launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream)
{
#if defined(__CCE_KT_TEST__) || !defined(__CCE_AICORE__)
    struct KernelArgs {
        alignas(((alignof(void*) + 3) >> 2) << 2) void *ffts_addr;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *w1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *w2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *expertId;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *scale1;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *scale2;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *probs;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *xActiveMask;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *c;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *expertTokenNums;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *workspaceGM;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *tilingGM;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *profileGM;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t stageProfile;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t startSyncDebug;
        alignas(((alignof(void*) + 3) >> 2) << 2) void *__ascendc_overflow;
    } kernel_args{};

    constexpr uint32_t overflow_status_size = 8;
    uint32_t ret = AllocAscendMemDevice(&kernel_args.__ascendc_overflow, overflow_status_size);
    if (ret != 0) {
        return ret;
    }

    ret = GetAscendCoreSyncAddr(&kernel_args.ffts_addr);
    if (ret != 0) {
        FreeAscendMemDevice(kernel_args.__ascendc_overflow);
        return ret;
    }

    kernel_args.x = args.x;
    kernel_args.w1 = args.weight1;
    kernel_args.w2 = args.weight2;
    kernel_args.expertId = args.expert_idx;
    kernel_args.scale1 = args.scale1;
    kernel_args.scale2 = args.scale2;
    kernel_args.probs = args.probs;
    kernel_args.xActiveMask = args.x_active_mask;
    kernel_args.c = args.out;
    kernel_args.expertTokenNums = args.expert_token_nums;
    kernel_args.workspaceGM = args.workspace;
    kernel_args.tilingGM = args.tiling;
    kernel_args.profileGM = args.profile_data;
    kernel_args.stageProfile = args.stage_profile;
    kernel_args.startSyncDebug = args.start_sync_debug;

    ret = launch_and_profiling_dispatch_ffn_combine(args.func_key, args.block_dim, stream,
                                                    reinterpret_cast<void **>(&kernel_args), sizeof(kernel_args));
    FreeAscendMemDevice(kernel_args.__ascendc_overflow);
    return ret;
#else
    dispatch_ffn_combine<<<args.block_dim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(args.x),
        reinterpret_cast<GM_ADDR>(args.weight1),
        reinterpret_cast<GM_ADDR>(args.weight2),
        reinterpret_cast<GM_ADDR>(args.expert_idx),
        reinterpret_cast<GM_ADDR>(args.scale1),
        reinterpret_cast<GM_ADDR>(args.scale2),
        reinterpret_cast<GM_ADDR>(args.probs),
        reinterpret_cast<GM_ADDR>(args.x_active_mask),
        reinterpret_cast<GM_ADDR>(args.out),
        reinterpret_cast<GM_ADDR>(args.expert_token_nums),
        reinterpret_cast<GM_ADDR>(args.workspace),
        reinterpret_cast<GM_ADDR>(args.tiling),
        reinterpret_cast<GM_ADDR>(args.profile_data),
        args.stage_profile,
        args.start_sync_debug);
    return 0;
#endif
}
