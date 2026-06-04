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
    GM_ADDR xActiveMask, GM_ADDR c, GM_ADDR expertTokenNums, GM_ADDR workspaceGM,  GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        const __gm__ DispatchFFNCombineTilingData *tilingData =
            reinterpret_cast<__gm__ DispatchFFNCombineTilingData *>(tilingGM);
        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM, tilingData);
        op.Process();
    }
}
#endif

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream)
{
#if defined(__CCE_KT_TEST__) || !defined(__CCE_AICORE__)
    dispatch_ffn_combine(
        args.block_dim,
        nullptr,
        stream,
        args.x,
        args.weight1,
        args.weight2,
        args.expert_idx,
        args.scale1,
        args.scale2,
        args.probs,
        args.x_active_mask,
        args.out,
        args.expert_token_nums,
        args.workspace,
        args.tiling);
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
        reinterpret_cast<GM_ADDR>(args.tiling));
#endif
}
