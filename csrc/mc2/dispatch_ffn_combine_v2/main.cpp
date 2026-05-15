#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "runtime_context.hpp"
#include "tiling_builder.hpp"

extern "C" rtError_t rtSetDevice(int32_t device);

namespace {

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void *host_src = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed");
    }
    if (host_src != nullptr &&
        aclrtMemcpy(buffer.ptr, bytes, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMemcpy host->device failed");
    }
    return buffer;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("bf16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

bool ZeroWindowMemory(const StandaloneRankRuntime &runtime)
{
    const uint64_t window_bytes = runtime.hccl.host_ctx.winSize;
    for (uint32_t i = 0; i < runtime.hccl.host_ctx.rankNum; ++i) {
        void *window_ptr = reinterpret_cast<void *>(runtime.hccl.host_ctx.windowsIn[i]);
        if (aclrtMemset(window_ptr, window_bytes, 0, window_bytes) != ACL_SUCCESS) {
            return false;
        }
    }
    return true;
}

bool RunOneRank(int rank_id, int world_size, const std::string &case_dir, const HcclRootInfo &root_info)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        return false;
    }

    bool ok = false;
    try {
        const CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        const DispatchFFNCombineBuildResult build = BuildDispatchFFNCombineTiling(cfg, runtime);

        const std::vector<uint8_t> x = ReadBinaryFile(files.x);
        const std::vector<uint8_t> weight1 = ReadBinaryFile(files.weight1);
        const std::vector<uint8_t> weight2 = ReadBinaryFile(files.weight2);
        const std::vector<uint8_t> expert_idx = ReadBinaryFile(files.expert_idx);
        const std::vector<uint8_t> scale1 = ReadBinaryFile(files.scale1);
        const std::vector<uint8_t> scale2 = ReadBinaryFile(files.scale2);
        const std::vector<uint8_t> probs = ReadBinaryFile(files.probs);
        const std::vector<uint8_t> x_active_mask = ReadBinaryFile(files.x_active_mask);
        const std::vector<uint8_t> expected_out_bytes = ReadBinaryFile(files.expected_out);
        const std::vector<uint16_t> expected_out = BytesToU16(expected_out_bytes);

        DeviceBuffer x_dev = MakeDeviceBuffer(x.size(), x.data());
        DeviceBuffer weight1_dev = MakeDeviceBuffer(weight1.size(), weight1.data());
        DeviceBuffer weight2_dev = MakeDeviceBuffer(weight2.size(), weight2.data());
        DeviceBuffer expert_idx_dev = MakeDeviceBuffer(expert_idx.size(), expert_idx.data());
        DeviceBuffer scale1_dev = MakeDeviceBuffer(scale1.size(), scale1.data());
        DeviceBuffer scale2_dev = MakeDeviceBuffer(scale2.size(), scale2.data());
        DeviceBuffer probs_dev = MakeDeviceBuffer(probs.size(), probs.data());
        DeviceBuffer x_active_mask_dev = MakeDeviceBuffer(x_active_mask.size(), x_active_mask.data());
        DeviceBuffer out_dev = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t));
        DeviceBuffer expert_token_nums_dev = MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t));
        DeviceBuffer workspace_dev = MakeDeviceBuffer(build.workspace_bytes);
        DeviceBuffer tiling_dev = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling);

        if (!ZeroWindowMemory(runtime)) {
            throw std::runtime_error("failed to zero HCCL windows");
        }
        if (aclrtMemset(out_dev.ptr, out_dev.bytes, 0, out_dev.bytes) != ACL_SUCCESS) {
            throw std::runtime_error("failed to zero out buffer");
        }
        if (aclrtMemset(expert_token_nums_dev.ptr, expert_token_nums_dev.bytes, 0, expert_token_nums_dev.bytes) != ACL_SUCCESS) {
            throw std::runtime_error("failed to zero expert_token_nums");
        }
        if (aclrtMemset(workspace_dev.ptr, workspace_dev.bytes, 0, workspace_dev.bytes) != ACL_SUCCESS) {
            throw std::runtime_error("failed to zero workspace");
        }

        DispatchFFNCombineLaunchArgs args;
        args.block_dim = build.block_dim;
        args.tiling = tiling_dev.ptr;
        args.workspace = workspace_dev.ptr;
        args.x = x_dev.ptr;
        args.weight1 = weight1_dev.ptr;
        args.weight2 = weight2_dev.ptr;
        args.expert_idx = expert_idx_dev.ptr;
        args.scale1 = scale1_dev.ptr;
        args.scale2 = scale2_dev.ptr;
        args.probs = probs_dev.ptr;
        args.x_active_mask = x_active_mask_dev.ptr;
        args.out = out_dev.ptr;
        args.expert_token_nums = expert_token_nums_dev.ptr;

        launchDispatchFFNCombine(args, runtime.compute_stream);
        if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
            throw std::runtime_error("stream sync failed");
        }

        std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
        if (aclrtMemcpy(actual_out.data(), actual_out.size() * sizeof(uint16_t), out_dev.ptr,
                        actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host output copy failed");
        }

        WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin",
                        actual_out.data(), actual_out.size() * sizeof(uint16_t));
        ok = CompareBf16File(expected_out, actual_out, 1e-3f, 1e-3f);
        std::cout << (ok ? "PASS" : "FAIL") << " rank=" << rank_id << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "rank=" << rank_id << " error: " << ex.what() << std::endl;
        ok = false;
    }

    DestroyStandaloneRankRuntime(runtime);
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        return 1;
    }

    const int rank_id = CommMpiRank();
    const int world_size = CommMpiSize();
    const char *case_dir_env = std::getenv("DISPATCH_FFN_COMBINE_V2_CASE_DIR");
    const std::string case_dir = case_dir_env ? case_dir_env : "../out";

    if (aclInit(nullptr) != ACL_SUCCESS) {
        CommMpiFinalize();
        return 1;
    }
    if (rtSetDevice(rank_id) != 0) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    if (aclrtSetDevice(rank_id) != ACL_SUCCESS) {
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo root_info{};
    if (rank_id == 0 && HcclGetRootInfo(&root_info) != HCCL_SUCCESS) {
        aclrtResetDevice(rank_id);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    CommMpiBcast(&root_info, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    const bool ok = RunOneRank(rank_id, world_size, case_dir, root_info);

    CommMpiBarrier();
    aclFinalize();
    CommMpiFinalize();
    return ok ? 0 : 1;
}
