#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace mc2::v4::routing {

template <typename T>
inline std::vector<T> ExclusiveScan(const std::vector<T>& input) {
    std::vector<T> output(input.size(), 0);
    T running = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = running;
        running += input[i];
    }
    return output;
}

}  // namespace mc2::v4::routing
