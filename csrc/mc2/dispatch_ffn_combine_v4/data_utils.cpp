#include "data_utils.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

namespace mc2::v4 {

double NowMs()
{
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

void PrintPerfRecord(const PerfRecord& record)
{
    const double bandwidthGbps = record.elapsedMs > 0.0
        ? (static_cast<double>(record.bytes) / 1.0e9) / (record.elapsedMs / 1.0e3)
        : 0.0;
    std::ostringstream oss;
    oss << "perf"
        << " mode=" << record.mode
        << " stage=" << record.stage
        << " rank=" << record.rank
        << " elapsed_ms=" << record.elapsedMs
        << " bytes=" << record.bytes
        << " items=" << record.items
        << " bandwidth_GBps=" << bandwidthGbps;
    std::cout << oss.str() << '\n';
}

}  // namespace mc2::v4
