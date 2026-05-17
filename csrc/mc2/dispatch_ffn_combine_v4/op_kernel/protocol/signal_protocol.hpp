#pragma once

#include <cstdint>

namespace mc2::v4::protocol {

#if defined(__CCE_AICORE__)
#define V4_PROTOCOL_INLINE inline __attribute__((always_inline)) __aicore__
#else
#define V4_PROTOCOL_INLINE inline
#endif

struct SignalLayoutConfig {
    uint32_t maxPeers = 64;
    uint32_t maxGroups = 64;
    uint32_t dispatchReadyBase = 0;
    uint32_t dispatchDoneBase = 4096;
    uint32_t computeDoneBase = 8192;
    uint32_t combineReadyBase = 12288;
    uint32_t combineDoneBase = 16384;
    uint32_t summaryBase = 20480;
};

V4_PROTOCOL_INLINE uint32_t FlatPeerGroupIndex(uint32_t peer, uint32_t group, uint32_t maxGroups = 64) {
    return peer * maxGroups + group;
}

V4_PROTOCOL_INLINE uint32_t DispatchReadyIndex(uint32_t peer, uint32_t group) {
    return FlatPeerGroupIndex(peer, group);
}

V4_PROTOCOL_INLINE uint32_t DispatchDoneIndex(uint32_t peer, uint32_t group) {
    return 4096 + FlatPeerGroupIndex(peer, group);
}

V4_PROTOCOL_INLINE uint32_t ComputeDoneIndex(uint32_t peer, uint32_t group) {
    return 8192 + FlatPeerGroupIndex(peer, group);
}

V4_PROTOCOL_INLINE uint32_t CombineReadyIndex(uint32_t peer, uint32_t group) {
    return 12288 + FlatPeerGroupIndex(peer, group);
}

V4_PROTOCOL_INLINE uint32_t CombineDoneIndex(uint32_t peer, uint32_t group) {
    return 16384 + FlatPeerGroupIndex(peer, group);
}

V4_PROTOCOL_INLINE uint32_t SummaryIndex(uint32_t peer, uint32_t lane = 0) {
    return 20480 + FlatPeerGroupIndex(peer, lane);
}

}  // namespace mc2::v4::protocol
