#pragma once

#include <cstdint>

namespace mc2::v4::protocol {

static constexpr uint32_t kMaxRemoteRanks = 64;

enum class RemoteRegion : uint32_t {
    Control,
    Dispatch,
    Compute,
    Combine,
    Signal,
};

struct RegionSlice {
    uint64_t base = 0;
    uint64_t bytes = 0;
};

struct RemoteWindowContext {
    uint64_t workspaceBase = 0;
    uint64_t workspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t segmentBytes = 0;
    uint64_t windowIn[kMaxRemoteRanks] = {};
    uint64_t windowOut[kMaxRemoteRanks] = {};

    uint64_t controlRegionOffset = 0;
    uint64_t dispatchRegionOffset = 0;
    uint64_t computeRegionOffset = 0;
    uint64_t combineRegionOffset = 0;
    uint64_t signalRegionOffset = 0;

    uint64_t controlRegionBytes = 0;
    uint64_t dispatchRegionBytes = 0;
    uint64_t computeRegionBytes = 0;
    uint64_t combineRegionBytes = 0;
    uint64_t signalRegionBytes = 0;

    uint64_t RegionOffset(RemoteRegion region) const {
        switch (region) {
            case RemoteRegion::Control:
                return controlRegionOffset;
            case RemoteRegion::Dispatch:
                return dispatchRegionOffset;
            case RemoteRegion::Compute:
                return computeRegionOffset;
            case RemoteRegion::Combine:
                return combineRegionOffset;
            case RemoteRegion::Signal:
                return signalRegionOffset;
        }
        return 0;
    }

    uint64_t RegionBase(RemoteRegion region) const {
        return workspaceBase + RegionOffset(region);
    }

    uint64_t RegionBytes(RemoteRegion region) const {
        switch (region) {
            case RemoteRegion::Control:
                return controlRegionBytes;
            case RemoteRegion::Dispatch:
                return dispatchRegionBytes;
            case RemoteRegion::Compute:
                return computeRegionBytes;
            case RemoteRegion::Combine:
                return combineRegionBytes;
            case RemoteRegion::Signal:
                return signalRegionBytes;
        }
        return 0;
    }

    uint64_t PeerRegionBase(uint32_t peer, RemoteRegion region) const {
        return windowIn[peer] + RegionOffset(region);
    }

    RegionSlice Slice(RemoteRegion region) const {
        return RegionSlice{RegionBase(region), RegionBytes(region)};
    }
};

}  // namespace mc2::v4::protocol
