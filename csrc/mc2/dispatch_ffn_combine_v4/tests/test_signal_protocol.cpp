#include "../op_kernel/protocol/signal_protocol.hpp"

#include <cassert>

int main() {
    using namespace mc2::v4::protocol;
    assert(DispatchReadyIndex(1, 2) != DispatchReadyIndex(2, 1));
    assert(DispatchReadyIndex(0, 0) == 0);
    assert(CombineReadyIndex(3, 1) > DispatchReadyIndex(3, 1));
    assert(DispatchDoneIndex(0, 0) > DispatchReadyIndex(0, 0));
    return 0;
}
