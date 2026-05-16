#ifndef SELECT_HELPER_HPP
#define SELECT_HELPER_HPP

#include "dispatch_policy_custom.hpp"

using namespace AscendC;

template <typename Layout, typename ElementType, typename = void>
struct LayoutBInitializer {
    PTO_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout{k, n};
    }
};

template <typename Layout, typename ElementType>
struct LayoutBInitializer<Layout, ElementType,
    std::enable_if_t<Layout::kTileLayout == pto::TileLayoutCustom::ZN>>
{
    PTO_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout::template MakeLayout<ElementType>(k, n);
    }
};

#endif // SELECT_HELPER_HPP
