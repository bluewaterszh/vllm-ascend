#pragma once

namespace mc2::v4::protocol {

inline void PublishPayloadFence() {
    __sync_synchronize();
}

inline void ConsumePayloadFence() {
    __sync_synchronize();
}

}  // namespace mc2::v4::protocol
