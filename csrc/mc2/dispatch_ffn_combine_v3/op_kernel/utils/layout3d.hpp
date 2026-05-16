#ifndef LAYOUT_3D_HPP
#define LAYOUT_3D_HPP

#include "kernel_operator.h"

#include <cstdint>

#if defined(__CCE_AICORE__)
#define DISPATCH_FFN_COMBINE_DEVICE __aicore__ inline
#else
#define DISPATCH_FFN_COMBINE_DEVICE inline
#endif

class Layout3D {
    int64_t strides[2];
public:
    DISPATCH_FFN_COMBINE_DEVICE Layout3D() {}

    DISPATCH_FFN_COMBINE_DEVICE Layout3D(int64_t stride0, int64_t stride1)
    {
        strides[0] = stride0;
        strides[1] = stride1;
    }

    DISPATCH_FFN_COMBINE_DEVICE int64_t operator()(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * strides[0] + dim1 * strides[1] + dim2;
    }
};

#undef DISPATCH_FFN_COMBINE_DEVICE

#endif // LAYOUT_3D_HPP
