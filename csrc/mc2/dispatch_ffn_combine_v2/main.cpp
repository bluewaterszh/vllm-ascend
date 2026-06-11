#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

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

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

std::string BuildAclError(const std::string &message, aclError ret);

bool TraceEnabled()
{
    const char *value = std::getenv("DISPATCH_FFN_COMBINE_V2_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void TraceLog(int rank_id, const std::string &message)
{
    if (!TraceEnabled()) {
        return;
    }

    std::ostringstream os;
    os << "[V2TRACE] rank=" << rank_id << " pid=" << static_cast<long long>(getpid()) << ' ' << message;
    const std::string line = os.str();
    std::cerr << line << std::endl;

    const char *trace_dir = std::getenv("DISPATCH_FFN_COMBINE_V2_TRACE_FILE_DIR");
    if (trace_dir != nullptr && trace_dir[0] != '\0') {
        std::ofstream trace_file(std::string(trace_dir) + "/rank" + std::to_string(rank_id) + ".log",
                                 std::ios::app);
        if (trace_file) {
            trace_file << line << '\n';
        }
    }
}

std::string RecentAclErrorText()
{
    const char *recent = aclGetRecentErrMsg();
    return recent != nullptr && recent[0] != '\0' ? std::string(recent) : std::string();
}

void LogFatalReturn(int rank_id, const std::string &message)
{
    std::cerr << "rank=" << rank_id << " error: " << message;
    const std::string recent = RecentAclErrorText();
    if (!recent.empty()) {
        std::cerr << ", recent=" << recent;
    }
    std::cerr << std::endl;
    TraceLog(rank_id, "fatal: " + message);
}

std::string PtrText(const void *ptr)
{
    std::ostringstream os;
    os << ptr;
    return os.str();
}

DeviceBuffer MakeDeviceBuffer(size_t bytes, const void *host_src = nullptr, const char *name = "device_buffer")
{
    DeviceBuffer buffer;
    buffer.bytes = bytes;
    const aclError malloc_ret = aclrtMalloc(&buffer.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (malloc_ret != ACL_SUCCESS) {
        throw std::runtime_error(BuildAclError(std::string("aclrtMalloc failed for ") + name +
                                                   " bytes=" + std::to_string(bytes),
                                               malloc_ret));
    }
    if (host_src != nullptr) {
        const aclError copy_ret = aclrtMemcpy(buffer.ptr, bytes, host_src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (copy_ret != ACL_SUCCESS) {
            throw std::runtime_error(BuildAclError(std::string("aclrtMemcpy host->device failed for ") + name +
                                                       " bytes=" + std::to_string(bytes),
                                                   copy_ret));
        }
    }
    return buffer;
}

DeviceBuffer MakeTensorList1(void *device_data, const char *name)
{
    const uint64_t tensor_list[] = {
        sizeof(uint64_t),
        reinterpret_cast<uint64_t>(device_data),
    };
    return MakeDeviceBuffer(sizeof(tensor_list), tensor_list, name);
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

std::string BuildAclError(const std::string &message, aclError ret)
{
    std::ostringstream os;
    os << message << ", ret=" << ret;
    const char *recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
        os << ", recent=" << recent;
    }
    return os.str();
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

std::string BuildAccuracyReportText(int rank_id, const AccuracyReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6)
       << "rank=" << rank_id
       << " max_diff=" << report.max_abs_err
       << " max_ratio=" << report.max_rel_err
       << " err=" << report.mismatch_count << "/" << report.total_count
       << " -> " << (report.pass ? "PASS" : "FAIL");
    return os.str();
}

std::string BuildIntVectorText(const std::string &name, const std::vector<int32_t> &values)
{
    std::ostringstream os;
    os << name << "=[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
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

bool ZeroWindowMemory(const StandaloneRankRuntime &runtime)
{
    const uint64_t window_bytes = runtime.hccl.host_ctx.winSize;
    const uint32_t rank_id = runtime.hccl.host_ctx.rankId;
    if (rank_id >= HCCL_STANDALONE_MAX_RANK_NUM) {
        return false;
    }
    void *window_ptr = reinterpret_cast<void *>(runtime.hccl.host_ctx.windowsIn[rank_id]);
    if (window_ptr == nullptr || window_bytes == 0) {
        return false;
    }
    if (aclrtMemset(window_ptr, window_bytes, 0, window_bytes) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

void ZeroDeviceBuffer(const DeviceBuffer &buffer, const char *name)
{
    if (aclrtMemset(buffer.ptr, buffer.bytes, 0, buffer.bytes) != ACL_SUCCESS) {
        throw std::runtime_error(std::string("failed to zero ") + name);
    }
}

void SynchronizeStream(aclrtStream stream)
{
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        throw std::runtime_error(BuildAclError("stream sync failed", sync_ret));
    }
}

void PrepareIterationState(const StandaloneRankRuntime &runtime,
                           const DeviceBuffer &out_dev,
                           const DeviceBuffer &expert_token_nums_dev,
                           const DeviceBuffer &workspace_dev,
                           const DeviceBuffer &profile_dev)
{
    if (!ZeroWindowMemory(runtime)) {
        throw std::runtime_error("failed to zero HCCL windows");
    }
    ZeroDeviceBuffer(out_dev, "out buffer");
    ZeroDeviceBuffer(expert_token_nums_dev, "expert_token_nums");
    ZeroDeviceBuffer(workspace_dev, "workspace");
    ZeroDeviceBuffer(profile_dev, "profile buffer");
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

double SysCntTicksToUs(uint64_t ticks)
{
    return static_cast<double>(ticks) * static_cast<double>(DISPATCH_FFN_COMBINE_SYS_CNT_NS_PER_TICK) / 1000.0;
}

const char *StageProfileName(uint32_t stage)
{
    static constexpr const char *kNames[DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT] = {
        "front", "dispatch", "gmm1", "swiglu", "gmm2", "combine", "unpermute"};
    return stage < DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT ? kNames[stage] : "unknown";
}

const char *ProfileEntryKind(uint32_t profile_idx)
{
    return profile_idx == 0U ? "aic" : "aiv";
}

uint32_t ProfileEntrySubblock(uint32_t profile_idx)
{
    return profile_idx == 0U ? 0U : profile_idx - 1U;
}

bool StageProfileEntryParticipates(uint32_t stage, uint32_t profile_idx)
{
    const bool is_aic = profile_idx == 0U;
    switch (stage) {
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2:
            return is_aic;
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE:
            return !is_aic;
        default:
            return false;
    }
}

struct StageProfileEnvelope {
    bool valid = false;
    uint32_t active_entries = 0;
    uint64_t start_min = 0;
    uint64_t end_max = 0;
    uint64_t max_core_ticks = 0;
};

bool ProfileIntervalValid(const uint64_t *entry, uint32_t stage)
{
    const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
    const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
    return start != 0U && end != 0U && end >= start;
}

double ReadKernelProfileUs(const DeviceBuffer &profile_dev, uint32_t block_dim, std::vector<uint8_t> *profile_host = nullptr)
{
    if (profile_dev.bytes == 0 || block_dim == 0) {
        return 0.0;
    }

    std::vector<uint8_t> profile(profile_dev.bytes, 0);
    if (aclrtMemcpy(profile.data(), profile.size(), profile_dev.ptr, profile_dev.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host profile copy failed");
    }
    if (profile_host != nullptr) {
        *profile_host = profile;
    }

    uint64_t start_min = std::numeric_limits<uint64_t>::max();
    uint64_t end_max = 0;
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (uint32_t profile_idx = 0; profile_idx < DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK; ++profile_idx) {
            const size_t offset =
                static_cast<size_t>(block) * DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK +
                static_cast<size_t>(profile_idx) * DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES;
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(profile.data() + offset);
            const uint64_t start = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START];
            const uint64_t end = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END];
            if (start == 0 || end == 0 || end < start) {
                continue;
            }
            start_min = std::min(start_min, start);
            end_max = std::max(end_max, end);
        }
    }
    if (start_min == std::numeric_limits<uint64_t>::max() || end_max < start_min) {
        return 0.0;
    }
    return SysCntTicksToUs(end_max - start_min);
}

std::string BuildStageProfileReportText(int rank_id, const std::vector<uint8_t> &profile, uint32_t block_dim)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    if (profile.empty() || block_dim == 0) {
        os << "rank=" << rank_id << " stageProfile EMPTY";
        return os.str();
    }

    uint64_t kernel_start_min = std::numeric_limits<uint64_t>::max();
    uint64_t kernel_end_max = 0;
    std::array<StageProfileEnvelope, DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT> envelopes{};

    for (uint32_t block = 0; block < block_dim; ++block) {
        for (uint32_t profile_idx = 0; profile_idx < DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK; ++profile_idx) {
            const size_t offset =
                static_cast<size_t>(block) * DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK +
                static_cast<size_t>(profile_idx) * DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES;
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(profile.data() + offset);
            const uint64_t kernel_start = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START];
            const uint64_t kernel_end = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END];
            if (kernel_start == 0U && kernel_end == 0U) {
                continue;
            }
            kernel_start_min = std::min(kernel_start_min, kernel_start);
            kernel_end_max = std::max(kernel_end_max, kernel_end);

            for (uint32_t stage = 0; stage < DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT; ++stage) {
                if (!StageProfileEntryParticipates(stage, profile_idx) || !ProfileIntervalValid(entry, stage)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
                const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
                StageProfileEnvelope &env = envelopes[stage];
                if (!env.valid) {
                    env.valid = true;
                    env.start_min = start;
                    env.end_max = end;
                } else {
                    env.start_min = std::min(env.start_min, start);
                    env.end_max = std::max(env.end_max, end);
                }
                env.active_entries += 1U;
                env.max_core_ticks = std::max(env.max_core_ticks, end - start);
            }
        }
    }

    if (kernel_start_min == std::numeric_limits<uint64_t>::max()) {
        os << "rank=" << rank_id << " stageProfile EMPTY";
        return os.str();
    }

    os << "rank=" << rank_id << " stageProfileEnvelope base=syscnt_min_kernel_start participant_cores_only=1"
       << " kernel_us=" << SysCntTicksToUs(kernel_end_max - kernel_start_min) << '\n';
    for (uint32_t stage = 0; stage < DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT; ++stage) {
        const StageProfileEnvelope &env = envelopes[stage];
        os << "  " << StageProfileName(stage);
        if (!env.valid) {
            os << " inactive\n";
            continue;
        }
        os << " active=" << env.active_entries
           << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min)
           << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min)
           << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min)
           << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks) << '\n';
    }

    os << "rank=" << rank_id << " stageProfileCore\n";
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (uint32_t profile_idx = 0; profile_idx < DISPATCH_FFN_COMBINE_PROFILE_ENTRIES_PER_BLOCK; ++profile_idx) {
            const size_t offset =
                static_cast<size_t>(block) * DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK +
                static_cast<size_t>(profile_idx) * DISPATCH_FFN_COMBINE_PROFILE_ENTRY_BYTES;
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(profile.data() + offset);
            const uint64_t kernel_start = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_START];
            const uint64_t kernel_end = entry[DISPATCH_FFN_COMBINE_PROFILE_KERNEL_END];
            if (kernel_start == 0U && kernel_end == 0U) {
                continue;
            }
            os << "  " << ProfileEntryKind(profile_idx) << " block=" << block;
            if (profile_idx != 0U) {
                os << " sub=" << ProfileEntrySubblock(profile_idx);
            }
            if (kernel_end >= kernel_start) {
                os << " kernel=(" << SysCntTicksToUs(kernel_start - kernel_start_min) << ','
                   << SysCntTicksToUs(kernel_end - kernel_start_min) << ','
                   << SysCntTicksToUs(kernel_end - kernel_start) << ')';
            }
            for (uint32_t stage = 0; stage < DISPATCH_FFN_COMBINE_PROFILE_STAGE_COUNT; ++stage) {
                os << ' ' << StageProfileName(stage) << '=';
                if (!ProfileIntervalValid(entry, stage)) {
                    os << '-';
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
                const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
                os << '(' << SysCntTicksToUs(start - kernel_start_min) << ','
                   << SysCntTicksToUs(end - kernel_start_min) << ','
                   << SysCntTicksToUs(end - start) << ')';
            }
            os << '\n';
        }
    }
    return os.str();
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

void PrintPerfSummary(int warmup_iters,
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
    std::cout << "  iters: warmup=" << warmup_iters << " measure=" << measure_iters << '\n';
    std::cout << "  kernel(syscnt max rank per iter): avg=" << kernel_stats.avg << " us"
              << " min=" << kernel_stats.min << " us"
              << " max=" << kernel_stats.max << " us"
              << " std=" << kernel_stats.stddev << " us\n";
    std::cout << "  e2e(max rank per iter):    avg=" << e2e_stats.avg << " us"
              << " min=" << e2e_stats.min << " us"
              << " max=" << e2e_stats.max << " us"
              << " std=" << e2e_stats.stddev << " us\n";
    std::cout << "===============================================================\n" << std::endl;
}

bool RunOneRank(int rank_id, int world_size, const std::string &case_dir, const HcclRootInfo &root_info)
{
    TraceLog(rank_id, "RunOneRank enter world_size=" + std::to_string(world_size) + " case_dir=" + case_dir);
    StandaloneRankRuntime runtime;
    if (!InitStandaloneRankRuntime(runtime, rank_id, world_size, root_info)) {
        LogFatalReturn(rank_id, "InitStandaloneRankRuntime failed");
        return false;
    }
    TraceLog(rank_id, "runtime initialized device_id=" + std::to_string(runtime.hccl.device_id) +
                          " winSize=" + std::to_string(runtime.hccl.host_ctx.winSize) +
                          " windowInLocal=0x" +
                          [&]() {
                              std::ostringstream os;
                              os << std::hex << runtime.hccl.host_ctx.windowsIn[rank_id];
                              return os.str();
                          }());

    bool ok = false;
    try {
        TraceLog(rank_id, "parse env begin");
        const int warmup_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_WARMUP_ITERS", kDefaultWarmupIters);
        const int measure_iters = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_MEASURE_ITERS", kDefaultMeasureIters);
        const bool skip_accuracy = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_SKIP_ACCURACY", 0) != 0;
        const bool stage_profile = ParseEnvInt("DISPATCH_FFN_COMBINE_V2_STAGE_PROFILE", 0) != 0;
        if (warmup_iters < 0 || measure_iters < 0) {
            throw std::runtime_error("warmup/measure iters must be non-negative");
        }
        TraceLog(rank_id, "env parsed warmup=" + std::to_string(warmup_iters) +
                              " measure=" + std::to_string(measure_iters) +
                              " skip_accuracy=" + std::to_string(skip_accuracy ? 1 : 0) +
                              " stage_profile=" + std::to_string(stage_profile ? 1 : 0));

        TraceLog(rank_id, "load case config begin");
        const CaseConfig cfg = LoadCaseConfig(case_dir + "/case.json");
        TraceLog(rank_id, "case config loaded m=" + std::to_string(cfg.m) +
                              " k=" + std::to_string(cfg.k) +
                              " n=" + std::to_string(cfg.n) +
                              " topk=" + std::to_string(cfg.topk) +
                              " expert_per_rank=" + std::to_string(cfg.expert_per_rank) +
                              " world_size=" + std::to_string(cfg.world_size) +
                              " max_output_size=" + std::to_string(cfg.max_output_size));
        const RankFileSet files = BuildRankFileSet(case_dir, rank_id);
        TraceLog(rank_id, "build tiling begin");
        const DispatchFFNCombineBuildResult build = BuildDispatchFFNCombineTiling(cfg, runtime);
        TraceLog(rank_id, "build tiling done block_dim=" + std::to_string(build.block_dim) +
                              " workspace_bytes=" + std::to_string(build.workspace_bytes));

        TraceLog(rank_id, "read input files begin");
        const std::vector<uint8_t> x = ReadBinaryFile(files.x);
        const std::vector<uint8_t> weight1 = ReadBinaryFile(files.weight1);
        const std::vector<uint8_t> weight2 = ReadBinaryFile(files.weight2);
        const std::vector<uint8_t> expert_idx = ReadBinaryFile(files.expert_idx);
        const std::vector<uint8_t> scale1 = ReadBinaryFile(files.scale1);
        const std::vector<uint8_t> scale2 = ReadBinaryFile(files.scale2);
        const std::vector<uint8_t> probs = ReadBinaryFile(files.probs);
        const std::vector<uint8_t> x_active_mask = ReadBinaryFile(files.x_active_mask);
        std::vector<uint16_t> expected_out;
        if (!skip_accuracy) {
            const std::vector<uint8_t> expected_out_bytes = ReadBinaryFile(files.expected_out);
            expected_out = BytesToU16(expected_out_bytes);
        }
        TraceLog(rank_id, "read input files done x=" + std::to_string(x.size()) +
                              " weight1=" + std::to_string(weight1.size()) +
                              " weight2=" + std::to_string(weight2.size()) +
                              " expert_idx=" + std::to_string(expert_idx.size()));
        const bool all_tokens_active = std::all_of(x_active_mask.begin(), x_active_mask.end(),
                                                   [](uint8_t value) { return value != 0; });

        TraceLog(rank_id, "device allocations begin");
        DeviceBuffer x_dev = MakeDeviceBuffer(x.size(), x.data(), "x");
        DeviceBuffer weight1_dev = MakeDeviceBuffer(weight1.size(), weight1.data(), "weight1");
        DeviceBuffer weight2_dev = MakeDeviceBuffer(weight2.size(), weight2.data(), "weight2");
        DeviceBuffer expert_idx_dev = MakeDeviceBuffer(expert_idx.size(), expert_idx.data(), "expert_idx");
        DeviceBuffer scale1_dev = MakeDeviceBuffer(scale1.size(), scale1.data(), "scale1");
        DeviceBuffer scale2_dev = MakeDeviceBuffer(scale2.size(), scale2.data(), "scale2");
        DeviceBuffer weight1_list_dev = MakeTensorList1(weight1_dev.ptr, "weight1_list");
        DeviceBuffer weight2_list_dev = MakeTensorList1(weight2_dev.ptr, "weight2_list");
        DeviceBuffer scale1_list_dev = MakeTensorList1(scale1_dev.ptr, "scale1_list");
        DeviceBuffer scale2_list_dev = MakeTensorList1(scale2_dev.ptr, "scale2_list");
        DeviceBuffer probs_dev = MakeDeviceBuffer(probs.size(), probs.data(), "probs");
        DeviceBuffer x_active_mask_dev = MakeDeviceBuffer(x_active_mask.size(), x_active_mask.data(), "x_active_mask");
        DeviceBuffer out_dev = MakeDeviceBuffer(static_cast<size_t>(cfg.m) * cfg.k * sizeof(uint16_t), nullptr, "out");
        DeviceBuffer expert_token_nums_dev =
            MakeDeviceBuffer(static_cast<size_t>(cfg.expert_per_rank) * sizeof(int32_t), nullptr, "expert_token_nums");
        DeviceBuffer workspace_dev = MakeDeviceBuffer(build.workspace_bytes, nullptr, "workspace");
        DeviceBuffer tiling_dev = MakeDeviceBuffer(sizeof(build.tiling), &build.tiling, "tiling");
        const size_t profile_bytes =
            static_cast<size_t>(build.block_dim) * DISPATCH_FFN_COMBINE_PROFILE_BYTES_PER_BLOCK;
        DeviceBuffer profile_dev = MakeDeviceBuffer(profile_bytes, nullptr, "profile");
        TraceLog(rank_id, "device allocations done workspace=" + std::to_string(workspace_dev.bytes) +
                              " profile=" + std::to_string(profile_dev.bytes));

        DispatchFFNCombineLaunchArgs args;
        args.block_dim = build.block_dim;
        args.func_key = DISPATCH_FFN_COMBINE_STANDALONE_FUNC_KEY;
        args.tiling = tiling_dev.ptr;
        args.workspace = workspace_dev.ptr;
        args.x = x_dev.ptr;
        args.weight1 = weight1_list_dev.ptr;
        args.weight2 = weight2_list_dev.ptr;
        args.expert_idx = expert_idx_dev.ptr;
        args.scale1 = scale1_list_dev.ptr;
        args.scale2 = scale2_list_dev.ptr;
        args.probs = probs_dev.ptr;
        args.x_active_mask = all_tokens_active ? nullptr : x_active_mask_dev.ptr;
        args.out = out_dev.ptr;
        args.expert_token_nums = expert_token_nums_dev.ptr;
        args.profile_data = profile_dev.ptr;
        args.stage_profile = stage_profile ? 1U : 0U;
        TraceLog(rank_id, "launch args ready block_dim=" + std::to_string(args.block_dim) +
                              " func_key=" + std::to_string(args.func_key) +
                              " stream=" + PtrText(runtime.compute_stream));

        auto launch_once = [&]() {
            TraceLog(rank_id, "kernel launch begin");
            const uint32_t launch_ret = launchDispatchFFNCombine(args, runtime.compute_stream);
            if (launch_ret != 0) {
                throw std::runtime_error("kernel launch failed, ret=" + std::to_string(launch_ret));
            }
            TraceLog(rank_id, "kernel launch returned, sync begin");
            SynchronizeStream(runtime.compute_stream);
            TraceLog(rank_id, "kernel sync done");
        };

        std::vector<double> kernel_times_us;
        std::vector<double> e2e_times_us;
        kernel_times_us.reserve(static_cast<size_t>(measure_iters));
        e2e_times_us.reserve(static_cast<size_t>(measure_iters));
        std::vector<uint8_t> last_profile_host;

        TraceLog(rank_id, "pre-warmup barrier begin");
        CommMpiBarrier();
        TraceLog(rank_id, "pre-warmup barrier done");
        for (int iter = 0; iter < warmup_iters; ++iter) {
            TraceLog(rank_id, "warmup iter=" + std::to_string(iter) + " prepare begin");
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
            TraceLog(rank_id, "warmup iter=" + std::to_string(iter) + " pre-launch barrier begin");
            CommMpiBarrier();
            TraceLog(rank_id, "warmup iter=" + std::to_string(iter) + " pre-launch barrier done");
            launch_once();
            TraceLog(rank_id, "warmup iter=" + std::to_string(iter) + " post-launch barrier begin");
            CommMpiBarrier();
            TraceLog(rank_id, "warmup iter=" + std::to_string(iter) + " done");
        }

        for (int iter = 0; iter < measure_iters; ++iter) {
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " prepare begin");
            PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " pre-launch barrier begin");
            CommMpiBarrier();
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " pre-launch barrier done");
            const auto host_start = std::chrono::high_resolution_clock::now();
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " kernel launch begin");
            const uint32_t launch_ret = launchDispatchFFNCombine(args, runtime.compute_stream);
            if (launch_ret != 0) {
                throw std::runtime_error("kernel launch failed, ret=" + std::to_string(launch_ret));
            }
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " kernel launch returned, sync begin");
            SynchronizeStream(runtime.compute_stream);
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " sync done, post barrier begin");
            CommMpiBarrier();
            const auto host_end = std::chrono::high_resolution_clock::now();
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " post barrier done");

            std::vector<uint8_t> profile_host;
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " read profile begin");
            kernel_times_us.push_back(ReadKernelProfileUs(profile_dev, build.block_dim,
                                                          stage_profile ? &profile_host : nullptr));
            if (stage_profile) {
                last_profile_host = std::move(profile_host);
            }
            e2e_times_us.push_back(std::chrono::duration<double, std::micro>(host_end - host_start).count());
            TraceLog(rank_id, "measure iter=" + std::to_string(iter) + " done");
        }

        TraceLog(rank_id, "gather perf samples begin");
        const std::vector<double> kernel_max_samples = GatherMaxSamplesToRoot(kernel_times_us, rank_id, world_size);
        const std::vector<double> e2e_max_samples = GatherMaxSamplesToRoot(e2e_times_us, rank_id, world_size);
        TraceLog(rank_id, "gather perf samples done");
        if (rank_id == 0) {
            PrintPerfSummary(warmup_iters, measure_iters, kernel_max_samples, e2e_max_samples);
        }
        if (stage_profile) {
            if (measure_iters > 0) {
                PrintOrderedByRank(rank_id, world_size,
                                   BuildStageProfileReportText(rank_id, last_profile_host, build.block_dim));
            } else {
                PrintOrderedByRank(rank_id, world_size,
                                   "rank=" + std::to_string(rank_id) + " stageProfile EMPTY measure_iters=0");
            }
        }

        TraceLog(rank_id, "final accuracy launch prepare begin");
        PrepareIterationState(runtime, out_dev, expert_token_nums_dev, workspace_dev, profile_dev);
        TraceLog(rank_id, "final accuracy launch barrier begin");
        CommMpiBarrier();
        TraceLog(rank_id, "final accuracy launch barrier done");
        launch_once();
        TraceLog(rank_id, "final accuracy post barrier begin");
        CommMpiBarrier();
        TraceLog(rank_id, "final accuracy post barrier done");

        if (skip_accuracy) {
            ok = true;
            PrintOrderedByRank(rank_id, world_size,
                               "rank=" + std::to_string(rank_id) + " accuracy=SKIP\nPASS rank=" +
                                   std::to_string(rank_id));
        } else {
            std::vector<uint16_t> actual_out(static_cast<size_t>(cfg.m) * cfg.k);
            if (aclrtMemcpy(actual_out.data(), actual_out.size() * sizeof(uint16_t), out_dev.ptr,
                            actual_out.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                throw std::runtime_error("device->host output copy failed");
            }
            std::vector<int32_t> expert_token_nums(cfg.expert_per_rank);
            if (aclrtMemcpy(expert_token_nums.data(), expert_token_nums.size() * sizeof(int32_t),
                            expert_token_nums_dev.ptr, expert_token_nums.size() * sizeof(int32_t),
                            ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                throw std::runtime_error("device->host expert_token_nums copy failed");
            }

            WriteBinaryFile(case_dir + "/output_rank" + std::to_string(rank_id) + ".bin",
                            actual_out.data(), actual_out.size() * sizeof(uint16_t));
            const AccuracyReport report = CompareFp16File(expected_out, actual_out, cfg.compare_atol, cfg.compare_rtol);
            ok = report.pass;
            std::string report_text = BuildAccuracyReportText(rank_id, report);
            if (!ok) {
                report_text += "\n" + BuildIntVectorText("expert_token_nums", expert_token_nums);
            }
            PrintOrderedByRank(rank_id, world_size, report_text + "\n" +
                                                         (ok ? "PASS" : "FAIL") + std::string(" rank=") +
                                                             std::to_string(rank_id));
        }
    } catch (const std::exception &ex) {
        std::cerr << "rank=" << rank_id << " error: " << ex.what() << std::endl;
        TraceLog(rank_id, std::string("exception: ") + ex.what());
        ok = false;
    }

    TraceLog(rank_id, "destroy runtime begin");
    DestroyStandaloneRankRuntime(runtime);
    TraceLog(rank_id, "RunOneRank exit ok=" + std::to_string(ok ? 1 : 0));
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "rank=? error: CommMpiInit failed" << std::endl;
        return 1;
    }

    const int rank_id = CommMpiRank();
    const int world_size = CommMpiSize();
    TraceLog(rank_id, "main enter world_size=" + std::to_string(world_size));
    const char *case_dir_env = std::getenv("DISPATCH_FFN_COMBINE_V2_CASE_DIR");
    const std::string case_dir = case_dir_env ? case_dir_env : "../out";
    TraceLog(rank_id, "case_dir=" + case_dir);

    TraceLog(rank_id, "aclInit begin");
    const aclError acl_init_ret = aclInit(nullptr);
    if (acl_init_ret != ACL_SUCCESS) {
        LogFatalReturn(rank_id, BuildAclError("aclInit failed", acl_init_ret));
        CommMpiFinalize();
        return 1;
    }
    TraceLog(rank_id, "aclInit done");
    TraceLog(rank_id, "rtSetDevice begin device=" + std::to_string(rank_id));
    const rtError_t rt_set_device_ret = rtSetDevice(rank_id);
    if (rt_set_device_ret != 0) {
        LogFatalReturn(rank_id, "rtSetDevice failed, ret=" + std::to_string(rt_set_device_ret));
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    TraceLog(rank_id, "rtSetDevice done");
    TraceLog(rank_id, "aclrtSetDevice begin device=" + std::to_string(rank_id));
    const aclError acl_set_device_ret = aclrtSetDevice(rank_id);
    if (acl_set_device_ret != ACL_SUCCESS) {
        LogFatalReturn(rank_id, BuildAclError("aclrtSetDevice failed", acl_set_device_ret));
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    TraceLog(rank_id, "aclrtSetDevice done");

    HcclRootInfo root_info{};
    if (rank_id == 0) {
        TraceLog(rank_id, "HcclGetRootInfo begin");
    }
    if (rank_id == 0 && HcclGetRootInfo(&root_info) != HCCL_SUCCESS) {
        LogFatalReturn(rank_id, "HcclGetRootInfo failed");
        aclrtResetDevice(rank_id);
        aclFinalize();
        CommMpiFinalize();
        return 1;
    }
    if (rank_id == 0) {
        TraceLog(rank_id, "HcclGetRootInfo done");
    }
    TraceLog(rank_id, "root info bcast begin");
    CommMpiBcast(&root_info, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    TraceLog(rank_id, "root info bcast done");
    TraceLog(rank_id, "pre RunOneRank barrier begin");
    CommMpiBarrier();
    TraceLog(rank_id, "pre RunOneRank barrier done");

    const bool ok = RunOneRank(rank_id, world_size, case_dir, root_info);

    TraceLog(rank_id, "post RunOneRank barrier begin ok=" + std::to_string(ok ? 1 : 0));
    CommMpiBarrier();
    TraceLog(rank_id, "post RunOneRank barrier done");
    TraceLog(rank_id, "aclFinalize begin");
    aclFinalize();
    TraceLog(rank_id, "aclFinalize done");
    CommMpiFinalize();
    TraceLog(rank_id, "main exit ok=" + std::to_string(ok ? 1 : 0));
    return ok ? 0 : 1;
}
