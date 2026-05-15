#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
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

constexpr int kDefaultWarmupIters = 3;
constexpr int kDefaultMeasureIters = 5;
constexpr double kMicrosecondsPerSecond = 1000.0 * 1000.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

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

struct EventHandle {
    aclrtEvent event = nullptr;

    ~EventHandle()
    {
        if (event != nullptr) {
            aclrtDestroyEvent(event);
        }
    }
};

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void *host_src = nullptr)
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    if (bytes == 0) {
        return buffer;
    }
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
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

int ParseEnvInt(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
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

void ZeroDeviceBuffer(const DeviceBuffer &buffer, const char *name)
{
    if (buffer.bytes == 0) {
        return;
    }
    if (aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void PrepareIterationState(const StandaloneRankRuntime &runtime,
                           const DeviceBuffer &out_dev,
                           const DeviceBuffer &expert_token_nums_dev,
                           const DeviceBuffer &workspace_dev)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL windows");
    }
    ZeroDeviceBuffer(out_dev, "out buffer");
    ZeroDeviceBuffer(expert_token_nums_dev, "expert_token_nums");
    ZeroDeviceBuffer(workspace_dev, "workspace");
}

PerfStats CalcStats(const std::vector<double> &samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }
    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

double ToTokensPerSecond(double tokens, double us)
{
    return us > 0.0 ? tokens * kMicrosecondsPerSecond / us : 0.0;
}

double ToTflops(double flops, double us)
{
    return us > 0.0 ? flops * kMicrosecondsPerSecond / us / 1e12 : 0.0;
}

double ToGbs(double bytes, double us)
{
    return us > 0.0 ? bytes * kMicrosecondsPerSecond / us / kBytesPerGiB : 0.0;
}

std::vector<double> GatherMaxSamplesToRoot(const std::vector<double> &local_samples,
                                           int rank_id,
                                           int world_size)
{
    if (local_samples.empty()) {
        return {};
    }
    const size_t sample_count = local_samples.size();
    const int bytes_per_rank = static_cast<int>(sample_count * sizeof(double));
    std::vector<double> gathered;
    if (rank_id == 0) {
        gathered.resize(sample_count * static_cast<size_t>(world_size));
    }
    CommMpiGather(local_samples.data(), bytes_per_rank, COMM_MPI_CHAR,
                  rank_id == 0 ? static_cast<void *>(gathered.data()) : nullptr,
                  bytes_per_rank, COMM_MPI_CHAR, 0);
    if (rank_id != 0) {
        return {};
    }

    std::vector<double> max_samples(sample_count, 0.0);
    for (size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        double max_value = gathered[sample_idx];
        for (int rank = 1; rank < world_size; ++rank) {
            max_value = std::max(max_value, gathered[static_cast<size_t>(rank) * sample_count + sample_idx]);
        }
        max_samples[sample_idx] = max_value;
    }
    return max_samples;
}

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport &report, double atol, double rtol)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << "rank=" << rank_id
       << " compare(total=" << report.total_count
       << ", mismatch=" << report.mismatch_count
       << ", nan_or_inf=" << report.nan_or_inf_count
       << ", max_abs_err=" << report.max_abs_err
       << ", max_rel_err=" << report.max_rel_err
       << ", mean_abs_err=" << report.mean_abs_err
       << ", rmse=" << report.rmse
       << ", atol=" << atol
       << ", rtol=" << rtol
       << ")";
    if (report.has_first_bad) {
        os << '\n'
           << std::fixed << std::setprecision(6)
           << "rank=" << rank_id
           << " first_bad(index=" << report.first_bad_index
           << ", expected=" << report.first_expected
           << ", actual=" << report.first_actual
           << ")";
    }
    return os.str();
}

void PrintOrderedByRank(int rank_id, int world_size, const std::string &text)
{
    for (int turn = 0; turn < world_size; ++turn) {
        CommMpiBarrier();
        if (turn == rank_id) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

void PrintPerfSummary(const CaseConfig &cfg,
                      int warmup_iters,
                      int measure_iters,
                      const std::vector<double> &kernel_samples_us,
                      const std::vector<double> &e2e_samples_us)
{
    if (kernel_samples_us.empty() || e2e_samples_us.empty()) {
        return;
    }
    const PerfStats kernel_stats = CalcStats(kernel_samples_us);
    const PerfStats e2e_stats = CalcStats(e2e_samples_us);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===============================================================\n";
    std::cout << "[PROFILE] dispatch_ffn_combine_v2\n";
    std::cout << "  shape: m=" << cfg.m
              << " k=" << cfg.k
              << " n=" << cfg.n
              << " topk=" << cfg.topk
              << " expert_per_rank=" << cfg.expert_per_rank
              << " world_size=" << cfg.world_size << '\n';
    std::cout << "  iters: warmup=" << warmup_iters
              << " measure=" << measure_iters << '\n';
    std::cout << "  logical work(all ranks): input_tokens=" << cfg.input_tokens_all_ranks
              << " routed_tokens=" << cfg.routed_tokens_all_ranks
              << " remote_routed_tokens=" << cfg.remote_routed_tokens_all_ranks
              << " compute_flops=" << cfg.compute_flops_all_ranks
              << " comm_bytes=" << cfg.comm_bytes_all_ranks << '\n';
    std::cout << "  kernel(max rank per iter): avg=" << kernel_stats.avg << " us"
              << " min=" << kernel_stats.min << " us"
              << " max=" << kernel_stats.max << " us"
              << " std=" << kernel_stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, kernel_stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, kernel_stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, kernel_stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, kernel_stats.avg) << " GB/s\n";
    std::cout << "  e2e(max rank per iter):    avg=" << e2e_stats.avg << " us"
              << " min=" << e2e_stats.min << " us"
              << " max=" << e2e_stats.max << " us"
              << " std=" << e2e_stats.stddev << " us\n";
    std::cout << "    input_tokens/s=" << ToTokensPerSecond(cfg.input_tokens_all_ranks, e2e_stats.avg)
              << " routed_tokens/s=" << ToTokensPerSecond(cfg.routed_tokens_all_ranks, e2e_stats.avg)
              << " eq_compute=" << ToTflops(cfg.compute_flops_all_ranks, e2e_stats.avg) << " TFLOPS"
              << " eq_comm=" << ToGbs(cfg.comm_bytes_all_ranks, e2e_stats.avg) << " GB/s\n";
    std::cout << "  note: equivalent compute/comm are derived from case.json logical workload, not hardware counters.\n";
    std::cout << "===============================================================\n" << std::endl;
}

bool RunOneRank(int rank_id, int world_size, const std::string &case_dir, const HcclRootInfo &root_info)
{
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        return false;
    }

    bool ok = false;
    try {
        const int warmup_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS", kDefaultWarmupIters);
        const int measure_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS", kDefaultMeasureIters);
        if (warmup_iters < 0 || measure_iters < 0) {
            throw std::runtime_error("warmup/measure iters must be non-negative");
        }

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

        EventHandle kernel_start;
        EventHandle kernel_end;
        if (measure_iters > 0) {
            if (aclrtCreateEvent(&kernel_start.event) != ACL_SUCCESS ||
                aclrtCreateEvent(&kernel_end.event) != ACL_SUCCESS) {
                throw std::runtime_error("failed to create ACL events");
            }
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

        auto launch_once = [&]() {
            launchDispatchFFNCombine(args, runtime.compute_stream);
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
        };

        std::vector<double> kernel_times_us;
        std::vector<double> e2e_times_us;
        kernel_times_us.reserve(static_cast<size_t>(measure_iters));
        e2e_times_us.reserve(static_cast<size_t>(measure_iters));

        CommMpiBarrier();
        for (int iter = 0; iter < warmup_iters; ++iter) {
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev);
            CommMpiBarrier();
            launch_once();
            CommMpiBarrier();
        }

        for (int iter = 0; iter < measure_iters; ++iter) {
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev);
            CommMpiBarrier();
            const auto host_start = std::chrono::high_resolution_clock::now();
            if (aclrtRecordEvent(kernel_start.event, runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("failed to record kernel start event");
            }
            launchDispatchFFNCombine(args, runtime.compute_stream);
            if (aclrtRecordEvent(kernel_end.event, runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("failed to record kernel end event");
            }
            if (aclrtSynchronizeStream(runtime.compute_stream) != ACL_SUCCESS) {
                throw std::runtime_error("stream sync failed");
            }
            CommMpiBarrier();
            const auto host_end = std::chrono::high_resolution_clock::now();

            float kernel_ms = 0.0f;
            if (aclrtEventElapsedTime(&kernel_ms, kernel_start.event, kernel_end.event) != ACL_SUCCESS) {
                throw std::runtime_error("failed to query kernel elapsed time");
            }
            kernel_times_us.push_back(static_cast<double>(kernel_ms) * 1000.0);
            e2e_times_us.push_back(std::chrono::duration<double, std::micro>(host_end - host_start).count());
        }

        const std::vector<double> kernel_max_samples = GatherMaxSamplesToRoot(kernel_times_us, rank_id, world_size);
        const std::vector<double> e2e_max_samples = GatherMaxSamplesToRoot(e2e_times_us, rank_id, world_size);
        if (rank_id == 0) {
            PrintPerfSummary(cfg, warmup_iters, measure_iters, kernel_max_samples, e2e_max_samples);
        }

        PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev);
        CommMpiBarrier();
        launch_once();
        CommMpiBarrier();

        std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
        if (aclrtMemcpy(actual_out.data(), actual_out.size() * sizeof(uint16_t), out_dev.ptr,
                        actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host output copy failed");
        }

        WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin",
                        actual_out.data(), actual_out.size() * sizeof(uint16_t));
        const AccuracyReport report = CompareFp16File(expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
        ok = report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildAccuracyReportText(rank_id, report, cfg.compare_atol, cfg.compare_rtol) + "\n" +
                           (ok ? "PASS" : "FAIL") + std::string(" rank=") + std::to_string(rank_id));
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
