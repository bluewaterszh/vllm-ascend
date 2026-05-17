#include "../case_io.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main()
{
    const auto oracle = mc2::v4::BuildHostComputeOracle();
    assert(!oracle.quantPayload.empty());
    assert(!oracle.scale1.empty());
    assert(!oracle.scale2.empty());
    assert(oracle.gmm1Out.size() == 4);
    assert(oracle.swigluOut.size() == 2);
    assert(oracle.gmm2Out.size() == 2);
    assert(mc2::v4::CompareFloats(oracle.gmm1Out, std::vector<float>({2.0f, 0.0f, 2.0f, 0.0f}), 1e-6f));
    assert(mc2::v4::CompareFloats(oracle.swigluOut, std::vector<float>({2.0f, 0.0f}), 1e-6f));
    assert(mc2::v4::CompareFloats(oracle.gmm2Out, std::vector<float>({2.0f, 0.0f}), 1e-6f));
    return 0;
}
