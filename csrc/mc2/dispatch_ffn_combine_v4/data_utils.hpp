#pragma once

#include <cstdint>
#include <string>

namespace mc2::v4 {

struct PerfRecord {
    std::string mode;
    std::string stage;
    uint32_t rank = 0;
    double elapsedMs = 0.0;
    uint64_t bytes = 0;
    uint32_t items = 0;
};

double NowMs();
void PrintPerfRecord(const PerfRecord& record);

}  // namespace mc2::v4
