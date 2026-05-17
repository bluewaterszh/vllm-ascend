#include "../op_kernel/protocol/remote_window.hpp"

#include <cassert>
#include <cstdint>

int main() {
    using namespace mc2::v4::protocol;
    RemoteWindowContext ctx{};
    ctx.workspaceBase = 0x100000;
    ctx.controlRegionOffset = 0x0000;
    ctx.dispatchRegionOffset = 0x4000;
    ctx.computeRegionOffset = 0x8000;
    ctx.combineRegionOffset = 0xC000;
    ctx.signalRegionOffset = 0x10000;

    assert(ctx.RegionBase(RemoteRegion::Control) == 0x100000);
    assert(ctx.RegionBase(RemoteRegion::Dispatch) == 0x104000);
    assert(ctx.RegionBase(RemoteRegion::Compute) == 0x108000);
    assert(ctx.RegionBase(RemoteRegion::Combine) == 0x10C000);
    assert(ctx.RegionBase(RemoteRegion::Signal) == 0x110000);
    return 0;
}
