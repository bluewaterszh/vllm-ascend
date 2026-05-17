#pragma once

#include <cstdint>
#include <vector>

namespace mc2::v4 {

struct DispatchOnlyLaunchArgs {
    void* remoteWindow = nullptr;
    void* dispatchTasks = nullptr;
    void* tiling = nullptr;
    uint32_t blockDim = 1;
};

struct ComputeOnlyLaunchArgs {
    void* params = nullptr;
    void* tiling = nullptr;
    uint32_t blockDim = 1;
};

struct CombineOnlyLaunchArgs {
    void* remoteWindow = nullptr;
    void* params = nullptr;
    void* tiling = nullptr;
    uint32_t blockDim = 1;
};

struct ModeRunResult {
    bool pass = false;
    std::vector<unsigned char> bytes;
    std::vector<float> output;
};

void launchDispatchFFNCombine(const DispatchOnlyLaunchArgs& args, void* stream);
void launchComputeOnly(const ComputeOnlyLaunchArgs& args, void* stream);
void launchCombineOnly(const CombineOnlyLaunchArgs& args, void* stream);

inline int ExitCodeFromResult(const ModeRunResult& result) {
    return result.pass ? 0 : 1;
}

}  // namespace mc2::v4
