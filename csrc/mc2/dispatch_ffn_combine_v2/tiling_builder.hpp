#pragma once

#include <cstdint>

#include "op_kernel/dispatch_ffn_combine_tiling.h"
#include "runtime_context.hpp"

struct CaseConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t topk = 0;
    uint32_t expert_per_rank = 0;
    uint32_t world_size = 0;
    uint32_t max_output_size = 0;
    uint32_t list_len = 1;
    double compare_atol = 1e-3;
    double compare_rtol = 1e-3;
    double input_tokens_all_ranks = 0.0;
    double routed_tokens_all_ranks = 0.0;
    double remote_routed_tokens_all_ranks = 0.0;
    double compute_flops_all_ranks = 0.0;
    double comm_bytes_all_ranks = 0.0;
};

struct DispatchFFNCombineBuildResult {
    DispatchFFNCombineTilingData tiling{};
    uint32_t block_dim = 1;
    uint64_t workspace_bytes = 0;
};

DispatchFFNCombineBuildResult BuildDispatchFFNCombineTiling(const CaseConfig &cfg,
                                                            const StandaloneRankRuntime &runtime);
