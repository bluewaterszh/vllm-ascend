#pragma once

#include <stdint.h>

constexpr uint64_t DISPATCH_FFN_COMBINE_DEVICE_TILING_KEY = 1000010UL;
constexpr uint64_t DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY = 0UL;
constexpr uint64_t DISPATCH_FFN_COMBINE_SYS_CNT_NS_PER_TICK = 20UL;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES = 256U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK = 3U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK =
    DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES * DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START = 0U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END = 1U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_STAGE_BASE = 2U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT = 7U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_BASE =
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_BASE + DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT * 2U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COUNT = 4U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_ENTRY_U64_COUNT =
    DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES / sizeof(uint64_t);

enum DispatchFFNCombineProfileStage : uint32_t {
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1 = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU = 3U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2 = 4U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE = 5U,
    DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE = 6U,
};

enum DispatchFFNCombineProfileReadyStage : uint32_t {
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM1 = 0U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_SWIGLU = 1U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM2 = 2U,
    DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COMBINE = 3U,
};

constexpr uint32_t DispatchFFNCombineProfileStageStartIndex(uint32_t stage)
{
    return DISPATCH_FFN_COMBINE_PROFILE_STAGE_BASE + stage * 2U;
}

constexpr uint32_t DispatchFFNCombineProfileStageEndIndex(uint32_t stage)
{
    return DispatchFFNCombineProfileStageStartIndex(stage) + 1U;
}

constexpr uint32_t DispatchFFNCombineProfileReadyStageStartIndex(uint32_t ready_stage)
{
    return DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_BASE + ready_stage;
}

static_assert(DispatchFFNCombineProfileStageEndIndex(DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT - 1U) <
              DISPATCH_FFN_COMBINE_PROFILE_ENTRY_U64_COUNT);
static_assert(DispatchFFNCombineProfileReadyStageStartIndex(DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COUNT - 1U) <
              DISPATCH_FFN_COMBINE_PROFILE_ENTRY_U64_COUNT);

struct DispatchFFNCombineLaunchArgs {
    void *x = nullptr;
    void *weight1 = nullptr;
    void *weight2 = nullptr;
    void *expert_idx = nullptr;
    void *scale1 = nullptr;
    void *scale2 = nullptr;
    void *probs = nullptr;
    void *x_active_mask = nullptr;
    void *out = nullptr;
    void *expert_token_nums = nullptr;
    void *workspace = nullptr;
    void *tiling = nullptr;
    void *profile_data = nullptr;
    uint32_t block_dim = 1;
    uint32_t stage_profile = 0;
    uint32_t start_sync_debug = 0;
    uint64_t func_key = DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY;
};

uint32_t launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream);
