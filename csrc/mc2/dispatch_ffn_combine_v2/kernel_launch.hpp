#pragma once

#include <stdint.h>

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
    uint32_t block_dim = 1;
};

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream);
