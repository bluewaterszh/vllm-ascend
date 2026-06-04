#include "data_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

std::string GetJsonScalar(const std::string &text, const std::string &key)
{
    const std::string token = "\"" + key + "\":";
    const size_t pos = text.find(token);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing key: " + key);
    }
    size_t begin = pos + token.size();
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    if (begin >= text.size()) {
        throw std::runtime_error("invalid value for key: " + key);
    }
    if (text[begin] == '"') {
        const size_t end = text.find('"', begin + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("unterminated string for key: " + key);
        }
        return text.substr(begin + 1, end - begin - 1);
    }
    const size_t end = text.find_first_of(",}\n", begin);
    return text.substr(begin, end - begin);
}

bool TryGetJsonScalar(const std::string &text, const std::string &key, std::string &value)
{
    try {
        value = GetJsonScalar(text, key);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

uint32_t ParseJsonUInt(const std::string &text, const std::string &key)
{
    return static_cast<uint32_t>(std::stoul(GetJsonScalar(text, key)));
}

double ParseJsonDouble(const std::string &text, const std::string &key, double default_value)
{
    std::string value;
    if (!TryGetJsonScalar(text, key, value)) {
        return default_value;
    }
    return std::stod(value);
}

} // namespace

std::vector<uint8_t> ReadBinaryFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open: " + path);
    }
    file.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(bytes);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

void WriteBinaryFile(const std::string &path, const void *data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open for write: " + path);
    }
    file.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

CaseConfig LoadCaseConfig(const std::string &case_json_path)
{
    const std::vector<uint8_t> raw = ReadBinaryFile(case_json_path);
    const std::string text(raw.begin(), raw.end());
    CaseConfig cfg;
    cfg.m = ParseJsonUInt(text, "m");
    cfg.k = ParseJsonUInt(text, "k");
    cfg.n = ParseJsonUInt(text, "n");
    cfg.topk = ParseJsonUInt(text, "topk");
    cfg.expert_per_rank = ParseJsonUInt(text, "expert_per_rank");
    cfg.world_size = ParseJsonUInt(text, "world_size");
    cfg.max_output_size = ParseJsonUInt(text, "max_output_size");
    cfg.list_len = 1;
    cfg.compare_atol = ParseJsonDouble(text, "compare_atol", 1e-4);
    cfg.compare_rtol = ParseJsonDouble(text, "compare_rtol", 1e-3);
    return cfg;
}

RankFileSet BuildRankFileSet(const std::string &case_dir, int rank)
{
    const std::string prefix = case_dir + "/rank" + std::to_string(rank) + "_";
    return RankFileSet{
        prefix + "x.bin",
        prefix + "weight1.bin",
        prefix + "weight2.bin",
        prefix + "expert_idx.bin",
        prefix + "scale1.bin",
        prefix + "scale2.bin",
        prefix + "probs.bin",
        prefix + "x_active_mask.bin",
        prefix + "expected_out.bin",
    };
}

float Fp16ToFloat(uint16_t value)
{
    const double sign = (value & 0x8000U) != 0 ? -1.0 : 1.0;
    const uint32_t exponent = (value >> 10U) & 0x1FU;
    const uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            return static_cast<float>(sign * 0.0);
        }
        return static_cast<float>(sign * std::ldexp(static_cast<double>(mantissa), -24));
    }
    if (exponent == 0x1FU) {
        if (mantissa == 0U) {
            return static_cast<float>(sign * std::numeric_limits<float>::infinity());
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(sign * std::ldexp(static_cast<double>(1024U + mantissa),
                                                static_cast<int>(exponent) - 25));
}

AccuracyReport CompareFp16File(const std::vector<uint16_t> &expected,
                               const std::vector<uint16_t> &actual,
                               double atol,
                               double rtol)
{
    AccuracyReport report;
    report.total_count = expected.size();
    if (expected.size() != actual.size()) {
        report.pass = false;
        report.mismatch_count = 1;
        return report;
    }
    report.err_threshold = static_cast<size_t>(static_cast<double>(report.total_count) * rtol);

    double sum_abs_err = 0.0;
    double sum_squared_err = 0.0;
    double min_abs_err = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < expected.size(); ++i) {
        const float expected_value = Fp16ToFloat(expected[i]);
        const float actual_value = Fp16ToFloat(actual[i]);
        const bool invalid = std::isnan(actual_value) || std::isinf(actual_value) ||
                             std::isnan(expected_value) || std::isinf(expected_value);
        const double abs_err = invalid ? std::numeric_limits<double>::infinity()
                                       : std::fabs(static_cast<double>(actual_value) - expected_value);
        const double tolerance = atol + rtol * std::fabs(static_cast<double>(expected_value));
        const double rel_denom = std::max(std::fabs(static_cast<double>(expected_value)), 1e-7);
        const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / rel_denom;

        if (invalid) {
            report.nan_or_inf_count += 1;
        }
        if (invalid || abs_err > tolerance) {
            report.mismatch_count += 1;
            if (!report.has_first_bad) {
                report.has_first_bad = true;
                report.first_bad_index = i;
                report.first_expected = expected_value;
                report.first_actual = actual_value;
            }
        }

        if (!invalid) {
            min_abs_err = std::min(min_abs_err, abs_err);
            report.max_abs_err = std::max(report.max_abs_err, abs_err);
            report.max_rel_err = std::max(report.max_rel_err, rel_err);
            sum_abs_err += abs_err;
            sum_squared_err += abs_err * abs_err;
        }
    }

    if (report.total_count > 0) {
        report.min_abs_err = std::isfinite(min_abs_err) ? min_abs_err : 0.0;
        report.mean_abs_err = sum_abs_err / static_cast<double>(report.total_count);
        report.rmse = std::sqrt(sum_squared_err / static_cast<double>(report.total_count));
    }
    report.pass = (report.mismatch_count <= report.err_threshold) && (report.nan_or_inf_count == 0);
    return report;
}
