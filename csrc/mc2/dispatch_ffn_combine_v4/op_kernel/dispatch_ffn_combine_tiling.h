#pragma once

#include <cstdint>

namespace mc2::v4 {

enum class KernelMode : uint32_t {
    DispatchOnly = 1,
    ComputeOnly = 2,
    CombineOnly = 3,
    FullChain = 4,
};

struct KernelLaunchConfig {
    uint32_t m = 16;
    uint32_t k = 128;
    uint32_t n = 128;
    uint32_t topk = 2;
    uint32_t numExperts = 2;
    uint32_t expertsPerRank = 1;
    uint32_t worldSize = 2;
    uint32_t numExpertGroups = 1;
    uint32_t rowsPerGroup = 16;
    uint32_t maxOutputSize = 32;
};

struct WorkspaceLayoutConfig {
    uint64_t controlBytes = 0x4000;
    uint64_t dispatchBytes = 0x4000;
    uint64_t computeBytes = 0x4000;
    uint64_t combineBytes = 0x4000;
    uint64_t signalBytes = 0x4000;

    uint64_t TotalBytes() const {
        return controlBytes + dispatchBytes + computeBytes + combineBytes + signalBytes;
    }
};

struct StandaloneKernelTilingData {
    uint32_t mode = 0;
    uint32_t taskCount = 0;
    uint32_t hiddenBytes = 0;
    uint32_t outputBytes = 0;
};

struct ComputeOnlyParams {
    uint64_t input = 0;
    uint64_t weight1 = 0;
    uint64_t gmm1Out = 0;
    uint64_t swigluOut = 0;
    uint64_t weight2 = 0;
    uint64_t gmm2Out = 0;
};

struct CombineOnlyParams {
    uint64_t localPartial = 0;
    uint64_t restoredRows = 0;
    uint32_t phase = 0;
    uint32_t reserved = 0;
};

}  // namespace mc2::v4
