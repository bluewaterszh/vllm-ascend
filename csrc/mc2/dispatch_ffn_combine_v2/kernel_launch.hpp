#pragma once

#include <stdint.h>

constexpr uint64_t DISPATCH_FFN_COMBINE_DEVICE_TILING_KEY = 1000010UL;
constexpr uint64_t DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY = 0UL;
constexpr uint64_t DISPATCH_FFN_COMBINE_SYS_CNT_NS_PER_TICK = 20UL;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES = 64U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK = 3U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK =
    DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES * DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START = 0U;
constexpr uint32_t DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END = 1U;

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
    uint64_t func_key = DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY;
};

uint32_t launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream);
