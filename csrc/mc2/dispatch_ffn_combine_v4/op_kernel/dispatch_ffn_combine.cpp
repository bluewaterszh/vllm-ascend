#include "kernel_operator.h"
#include "dispatch_ffn_combine_tiling.h"
#include "protocol/remote_window.hpp"
#include "protocol/task_plan.hpp"
#include "dispatch_ffn_combine.h"
#include "../kernel_launch.hpp"

using V4RemoteWindowContext = mc2::v4::protocol::RemoteWindowContext;
using V4DispatchPullTask = mc2::v4::protocol::DispatchPullTask;
using V4ComputeOnlyParams = mc2::v4::ComputeOnlyParams;
using V4CombineOnlyParams = mc2::v4::CombineOnlyParams;
using V4StandaloneKernelTilingData = mc2::v4::StandaloneKernelTilingData;

extern "C" __global__ __aicore__ void dispatch_ffn_combine(
    __gm__ V4RemoteWindowContext* remoteCtx,
    __gm__ uint8_t* modeArgs,
    __gm__ V4StandaloneKernelTilingData* tiling)
{
    if (tiling->mode == static_cast<uint32_t>(mc2::v4::KernelMode::DispatchOnly)) {
        mc2::v4::dispatch::RunDispatchOnlyKernel(remoteCtx,
                                                 reinterpret_cast<__gm__ V4DispatchPullTask*>(modeArgs),
                                                 tiling);
    } else if (tiling->mode == static_cast<uint32_t>(mc2::v4::KernelMode::ComputeOnly)) {
        mc2::v4::compute::RunComputeOnlyKernel(reinterpret_cast<__gm__ V4ComputeOnlyParams*>(modeArgs), tiling);
    } else if (tiling->mode == static_cast<uint32_t>(mc2::v4::KernelMode::CombineOnly)) {
        mc2::v4::combine::RunCombineOnlyKernel(remoteCtx,
                                               reinterpret_cast<__gm__ V4CombineOnlyParams*>(modeArgs),
                                               tiling);
    }
}

void mc2::v4::launchDispatchFFNCombine(const DispatchOnlyLaunchArgs& args, void* stream)
{
    dispatch_ffn_combine<<<args.blockDim, nullptr, stream>>>(
        static_cast<mc2::v4::protocol::RemoteWindowContext*>(args.remoteWindow),
        static_cast<uint8_t*>(args.dispatchTasks),
        static_cast<mc2::v4::StandaloneKernelTilingData*>(args.tiling));
}

void mc2::v4::launchComputeOnly(const ComputeOnlyLaunchArgs& args, void* stream)
{
    dispatch_ffn_combine<<<args.blockDim, nullptr, stream>>>(
        nullptr,
        static_cast<uint8_t*>(args.params),
        static_cast<mc2::v4::StandaloneKernelTilingData*>(args.tiling));
}

void mc2::v4::launchCombineOnly(const CombineOnlyLaunchArgs& args, void* stream)
{
    dispatch_ffn_combine<<<args.blockDim, nullptr, stream>>>(
        static_cast<mc2::v4::protocol::RemoteWindowContext*>(args.remoteWindow),
        static_cast<uint8_t*>(args.params),
        static_cast<mc2::v4::StandaloneKernelTilingData*>(args.tiling));
}
