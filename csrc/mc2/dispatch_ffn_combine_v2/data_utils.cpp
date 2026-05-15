#include "data_utils.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

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

static uint32_t ParseJsonUInt(const std::string &text, const std::string &key)
{
    const std::string token = "\"" + key + "\":";
    const size_t pos = text.find(token);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing key: " + key);
    }
    const size_t begin = pos + token.size();
    const size_t end = text.find_first_of(",}\n", begin);
    return static_cast<uint32_t>(std::stoul(text.substr(begin, end - begin)));
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

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool CompareBf16File(const std::vector<uint16_t> &expected,
                     const std::vector<uint16_t> &actual,
                     float atol,
                     float rtol)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const float exp = Bf16ToFloat(expected[i]);
        const float act = Bf16ToFloat(actual[i]);
        const float diff = std::fabs(exp - act);
        const float limit = atol + rtol * std::fabs(exp);
        if (std::isnan(act) || std::isinf(act) || diff > limit) {
            return false;
        }
    }
    return true;
}
