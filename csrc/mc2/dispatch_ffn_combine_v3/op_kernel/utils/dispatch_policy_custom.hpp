#ifndef DISPATH_POLICY_CUSTOM_HPP
#define DISPATH_POLICY_CUSTOM_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#define PTO_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define PTO_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define PTO_HOST_DEVICE
#endif
#define PTO_GLOBAL __global__ __aicore__

namespace pto_ext {

template <class...>
inline constexpr bool DEPENDENT_FALSE = false;

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;
constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;
constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;
constexpr uint32_t BYTE_PER_BLK_FP = 128;

using PtoShape1D = pto::Shape<pto::DYNAMIC, 1, 1, 1, 1>;
using PtoShape2D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, 1, 1, 1>;
using PtoShape3D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1, 1>;
using PtoShape4D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using PtoStride1D = pto::Stride<pto::DYNAMIC, 1, 1, 1, 1>;
using PtoStride2D = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, 1, 1, 1>;
using PtoStride4D = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using PtoCoord1D = PtoShape1D;
using PtoCoord2D = PtoShape2D;
using PtoCoord3D = PtoShape3D;

PTO_HOST_DEVICE PtoCoord1D MakePtoCoord1D(int64_t dim0)
{
    return PtoCoord1D(dim0);
}

PTO_HOST_DEVICE PtoCoord2D MakePtoCoord2D(int64_t dim0, int64_t dim1)
{
    return PtoCoord2D(dim0, dim1);
}

PTO_HOST_DEVICE PtoCoord3D MakePtoCoord3D(int64_t dim0, int64_t dim1, int64_t dim2)
{
    return PtoCoord3D(dim0, dim1, dim2);
}

PTO_HOST_DEVICE PtoShape3D MakePtoShape3D(int64_t dim0, int64_t dim1, int64_t dim2)
{
    return PtoShape3D(dim0, dim1, dim2);
}

PTO_HOST_DEVICE PtoCoord2D MulPtoCoord2D(PtoCoord2D const &lhs, PtoCoord2D const &rhs)
{
    return PtoCoord2D(lhs.shape[0] * rhs.shape[0], lhs.shape[1] * rhs.shape[1]);
}

PTO_HOST_DEVICE uint32_t GetPtoShapeM(PtoShape3D const &shape)
{
    return static_cast<uint32_t>(shape.shape[0]);
}

PTO_HOST_DEVICE uint32_t GetPtoShapeN(PtoShape3D const &shape)
{
    return static_cast<uint32_t>(shape.shape[1]);
}

PTO_HOST_DEVICE uint32_t GetPtoShapeK(PtoShape3D const &shape)
{
    return static_cast<uint32_t>(shape.shape[2]);
}

PTO_HOST_DEVICE PtoShape2D GetPtoShapeMN(PtoShape3D const &shape)
{
    return PtoShape2D(shape.shape[0], shape.shape[1]);
}

PTO_HOST_DEVICE PtoShape2D GetPtoShapeMK(PtoShape3D const &shape)
{
    return PtoShape2D(shape.shape[0], shape.shape[2]);
}

PTO_HOST_DEVICE PtoShape2D GetPtoShapeKN(PtoShape3D const &shape)
{
    return PtoShape2D(shape.shape[2], shape.shape[1]);
}

class EmptyClass {};

template <uint32_t Align, class T>
PTO_HOST_DEVICE constexpr T CeilDiv(T value)
{
    return (value + static_cast<T>(Align) - 1) / static_cast<T>(Align);
}

template <class T, class U>
PTO_HOST_DEVICE constexpr auto CeilDiv(T lhs, U rhs)
{
    using Common = std::common_type_t<T, U>;
    Common lhsValue = static_cast<Common>(lhs);
    Common rhsValue = static_cast<Common>(rhs);
    return (lhsValue + rhsValue - 1) / rhsValue;
}

template <uint32_t Align, class T>
PTO_HOST_DEVICE constexpr T RoundUp(T value)
{
    return CeilDiv<Align>(value) * static_cast<T>(Align);
}

template <class T, class U>
PTO_HOST_DEVICE constexpr auto AlignUp(T value, U align)
{
    using Common = std::common_type_t<T, U>;
    Common alignValue = static_cast<Common>(align);
    return CeilDiv(static_cast<Common>(value), alignValue) * alignValue;
}

template <
    int RANK_,
    class Index_ = uint32_t,
    class LongIndex_ = int64_t>
struct Coord {
    static constexpr int RANK = RANK_;
    using Index = Index_;
    using LongIndex = LongIndex_;

    Index idx[RANK]{};

    PTO_HOST_DEVICE constexpr explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    PTO_HOST_DEVICE constexpr Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    PTO_HOST_DEVICE explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != 0) {
                return true;
            }
        }
        return false;
    }

    PTO_HOST_DEVICE bool operator!() const
    {
        return !static_cast<bool>(*this);
    }

    PTO_HOST_DEVICE Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] + b[i];
        }
        return c;
    }

    PTO_HOST_DEVICE Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] - b[i];
        }
        return c;
    }

    PTO_HOST_DEVICE Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] * b[i];
        }
        return c;
    }

    PTO_HOST_DEVICE Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] / b[i];
        }
        return c;
    }

    PTO_HOST_DEVICE Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b[i];
        }
        return *this;
    }

    PTO_HOST_DEVICE bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    PTO_HOST_DEVICE Index &operator[](int dim)
    {
        return idx[dim];
    }

    PTO_HOST_DEVICE Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    template <int DIM>
    PTO_HOST_DEVICE Index &At()
    {
        return idx[DIM];
    }

    PTO_HOST_DEVICE Index &At(int dim)
    {
        return idx[dim];
    }

    template <int DIM>
    PTO_HOST_DEVICE Index const &At() const
    {
        return idx[DIM];
    }

    PTO_HOST_DEVICE Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    PTO_HOST_DEVICE auto GetCoordByAxis() const
    {
        Index values[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{values};
    }
};

template <class... Ts>
PTO_HOST_DEVICE constexpr auto MakeCoord(Ts... values)
{
    using Index = std::common_type_t<Ts...>;
    Index data[sizeof...(Ts)]{static_cast<Index>(values)...};
    return Coord<sizeof...(Ts), Index>{data};
}

template <int RANK, class Index, class LongIndex>
PTO_HOST_DEVICE Coord<RANK, Index, LongIndex> CeilDiv(
    Coord<RANK, Index, LongIndex> const &lhs,
    Coord<RANK, Index, LongIndex> const &rhs)
{
    Coord<RANK, Index, LongIndex> out;
    for (int i = 0; i < RANK; ++i) {
        out[i] = pto_ext::CeilDiv(lhs[i], rhs[i]);
    }
    return out;
}

template <uint32_t ROW_ = 1, uint32_t COLUMN_ = 1>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;
    static constexpr int64_t COUNT = ROW * COLUMN;

    PTO_HOST_DEVICE static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }

    PTO_HOST_DEVICE static PtoShape2D ToPtoShape()
    {
        return PtoShape2D(ROW, COLUMN);
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<2, Index>;
    using LongIndex = typename Base::LongIndex;

    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;

    PTO_HOST_DEVICE MatrixCoord() = default;
    PTO_HOST_DEVICE MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}
    PTO_HOST_DEVICE MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}
    PTO_HOST_DEVICE MatrixCoord(LongIndex row, LongIndex column)
        : Base(MakeCoord(Index(row), Index(column))) {}

    PTO_HOST_DEVICE Index const &row() const { return this->At(ROW_INDEX); }
    PTO_HOST_DEVICE Index &row() { return this->At(ROW_INDEX); }
    PTO_HOST_DEVICE Index const &column() const { return this->At(COLUMN_INDEX); }
    PTO_HOST_DEVICE Index &column() { return this->At(COLUMN_INDEX); }

    PTO_HOST_DEVICE MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    PTO_HOST_DEVICE MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }

    PTO_HOST_DEVICE PtoShape2D ToPtoShape() const
    {
        return PtoShape2D(row(), column());
    }
};

PTO_HOST_DEVICE MatrixCoord CeilDiv(MatrixCoord const &lhs, MatrixCoord const &rhs)
{
    return MatrixCoord(pto_ext::CeilDiv(lhs.row(), rhs.row()), pto_ext::CeilDiv(lhs.column(), rhs.column()));
}

PTO_HOST_DEVICE PtoShape2D CeilDiv(PtoShape2D const &lhs, PtoShape2D const &rhs)
{
    return PtoShape2D(
        pto_ext::CeilDiv(lhs.shape[0], rhs.shape[0]),
        pto_ext::CeilDiv(lhs.shape[1], rhs.shape[1]));
}

template <uint32_t M_ = 1, uint32_t N_ = 1, uint32_t K_ = 1>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;
    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;
    static constexpr int64_t COUNT = MNK;

    PTO_HOST_DEVICE static Coord<3> ToCoord() { return MakeCoord(M, N, K); }
    PTO_HOST_DEVICE static Coord<2> ToCoordMN() { return MakeCoord(M, N); }
    PTO_HOST_DEVICE static Coord<2> ToCoordMK() { return MakeCoord(M, K); }
    PTO_HOST_DEVICE static Coord<2> ToCoordKN() { return MakeCoord(K, N); }
    PTO_HOST_DEVICE static PtoShape3D ToPtoShape() { return PtoShape3D(M, N, K); }
    PTO_HOST_DEVICE static PtoShape2D ToPtoShapeMN() { return PtoShape2D(M, N); }
    PTO_HOST_DEVICE static PtoShape2D ToPtoShapeMK() { return PtoShape2D(M, K); }
    PTO_HOST_DEVICE static PtoShape2D ToPtoShapeKN() { return PtoShape2D(K, N); }
};



namespace layout {

struct ND {
    static constexpr int RANK = 2;
    static constexpr pto::Layout kPtoLayout = pto::Layout::ND;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ND;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape2D;
    using Stride = PtoStride2D;

    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE ND(Index rows = 0, Index cols = 0)
        : shape_(rows, cols), stride_(LongIndex(cols), LongIndex(1)) {}

    PTO_HOST_DEVICE ND(Index rows, Index cols, LongIndex ldm)
        : shape_(rows, cols), stride_(ldm, LongIndex(1)) {}

    PTO_HOST_DEVICE ND(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    PTO_HOST_DEVICE static ND MakeLayout(Index rows, Index cols)
    {
        return ND(rows, cols);
    }

    PTO_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_.stride[0] + LongIndex(coord.column());
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) * stride_.stride[0] + LongIndex(coord.shape[1]);
    }

    PTO_HOST_DEVICE ND GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ND(tileShape.ToPtoShape(), stride());
    }

    PTO_HOST_DEVICE ND GetTileLayout(PtoShape2D const &tileShape) const
    {
        return ND(tileShape, stride());
    }

    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

struct DN {
    static constexpr int RANK = 2;
    static constexpr pto::Layout kPtoLayout = pto::Layout::DN;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::DN;
    static constexpr pto::BLayout kBLayout = pto::BLayout::ColMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape2D;
    using Stride = PtoStride2D;

    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE DN(Index rows = 0, Index cols = 0)
        : shape_(rows, cols), stride_(LongIndex(1), LongIndex(rows)) {}

    PTO_HOST_DEVICE DN(Index rows, Index cols, LongIndex ldm)
        : shape_(rows, cols), stride_(LongIndex(1), ldm) {}

    PTO_HOST_DEVICE DN(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    PTO_HOST_DEVICE static DN MakeLayout(Index rows, Index cols)
    {
        return DN(rows, cols);
    }

    PTO_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_.stride[1];
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) + LongIndex(coord.shape[1]) * stride_.stride[1];
    }

    PTO_HOST_DEVICE DN GetTileLayout(MatrixCoord const &tileShape) const
    {
        return DN(tileShape.ToPtoShape(), stride());
    }

    PTO_HOST_DEVICE DN GetTileLayout(PtoShape2D const &tileShape) const
    {
        return DN(tileShape, stride());
    }

    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

struct VectorLayout {
    static constexpr int RANK = 1;
    static constexpr pto::Layout kPtoLayout = pto::Layout::SCALE;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::NONE;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape1D;
    using Stride = PtoStride1D;
    using TensorCoord = Coord<RANK, Index>;

    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE VectorLayout(Index size = 0)
        : shape_(size), stride_(LongIndex(1)) {}

    PTO_HOST_DEVICE VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    PTO_HOST_DEVICE LongIndex GetOffset(TensorCoord const &coord) const
    {
        return stride_.stride[0] * coord[0];
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord1D const &coord) const
    {
        return stride_.stride[0] * coord.shape[0];
    }

    PTO_HOST_DEVICE VectorLayout GetTileLayout(TensorCoord const &tileShape) const
    {
        return VectorLayout(Shape(tileShape[0]), stride());
    }

    PTO_HOST_DEVICE VectorLayout GetTileLayout(PtoShape1D const &tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

struct Nz {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::NZ;
    static constexpr pto::BLayout kBLayout = pto::BLayout::ColMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::RowMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE Nz(
        Index orgRows = 0,
        Index orgCols = 0,
        Index rowsInFractal = 0,
        Index rowsByFractal = 0,
        Index colsInFractal = 0,
        Index colsByFractal = 0,
        LongIndex strideRowsInFractal = 0,
        LongIndex strideRowsByFractal = 0,
        LongIndex strideColsInFractal = 0,
        LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal) {}

    PTO_HOST_DEVICE Nz(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    PTO_HOST_DEVICE static Nz MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return Nz(orgRows,
                  orgCols,
                  ELE_NUM_PER_C0,
                  rowsRound / ELE_NUM_PER_C0,
                  C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL,
                  1,
                  colsRound * ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  BYTE_PER_FRACTAL / sizeof(Element));
    }

    PTO_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3] +
            (LongIndex(coord.row()) % shape_.shape[0]) * stride_.stride[0] +
            (LongIndex(coord.column()) % shape_.shape[2]) * stride_.stride[2];
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3] +
            (LongIndex(coord.shape[0]) % shape_.shape[0]) * stride_.stride[0] +
            (LongIndex(coord.shape[1]) % shape_.shape[2]) * stride_.stride[2];
    }

    PTO_HOST_DEVICE Nz GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return Nz(tileOriShape.ToPtoShape(), tileShape, stride());
    }

    PTO_HOST_DEVICE Nz GetTileLayout(PtoShape2D const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.shape[0], shape(0)), shape(2), CeilDiv(tileOriShape.shape[1], shape(2)));
        return Nz(tileOriShape, tileShape, stride());
    }

    PTO_HOST_DEVICE static Nz MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return MakeLayoutInL0C(shape.ToPtoShape());
    }

    PTO_HOST_DEVICE static Nz MakeLayoutInL0C(PtoShape2D const &shape)
    {
        return Nz(shape.shape[0],
                  shape.shape[1],
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[0]),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[1]),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.shape[0]) * C0_NUM_PER_FRACTAL);
    }

    PTO_HOST_DEVICE int64_t orgShape(int idx) const { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &orgShape(int idx) { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

struct Zn {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ZN;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::ColMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE Zn(
        Index orgRows = 0,
        Index orgCols = 0,
        Index rowsInFractal = 0,
        Index rowsByFractal = 0,
        Index colsInFractal = 0,
        Index colsByFractal = 0,
        LongIndex strideRowsInFractal = 0,
        LongIndex strideRowsByFractal = 0,
        LongIndex strideColsInFractal = 0,
        LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal) {}

    PTO_HOST_DEVICE Zn(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    PTO_HOST_DEVICE static Zn MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return Zn(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  BYTE_PER_FRACTAL / sizeof(Element),
                  1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    PTO_HOST_DEVICE static Zn MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return MakeLayoutInL0C(shape.ToPtoShape());
    }

    PTO_HOST_DEVICE static Zn MakeLayoutInL0C(PtoShape2D const &shape)
    {
        return Zn(shape.shape[0],
                  shape.shape[1],
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[0]),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[1]),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.shape[0]) * C0_NUM_PER_FRACTAL);
    }

    PTO_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3] +
            (LongIndex(coord.row()) % shape_.shape[0]) * stride_.stride[0] +
            (LongIndex(coord.column()) % shape_.shape[2]) * stride_.stride[2];
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3] +
            (LongIndex(coord.shape[0]) % shape_.shape[0]) * stride_.stride[0] +
            (LongIndex(coord.shape[1]) % shape_.shape[2]) * stride_.stride[2];
    }

    PTO_HOST_DEVICE Zn GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return Zn(tileOriShape.ToPtoShape(), tileShape, stride());
    }

    PTO_HOST_DEVICE Zn GetTileLayout(PtoShape2D const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.shape[0], shape(0)), shape(2), CeilDiv(tileOriShape.shape[1], shape(2)));
        return Zn(tileOriShape, tileShape, stride());
    }

    PTO_HOST_DEVICE int64_t orgShape(int idx) const { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &orgShape(int idx) { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

struct Zz {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ZZ;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::RowMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    PTO_HOST_DEVICE Zz(
        Index orgRows = 0,
        Index orgCols = 0,
        Index rowsInFractal = 0,
        Index rowsByFractal = 0,
        Index colsInFractal = 0,
        Index colsByFractal = 0,
        LongIndex strideRowsInFractal = 0,
        LongIndex strideRowsByFractal = 0,
        LongIndex strideColsInFractal = 0,
        LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal) {}

    PTO_HOST_DEVICE Zz(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    PTO_HOST_DEVICE static Zz MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return Zz(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  colsRound * C0_NUM_PER_FRACTAL,
                  1,
                  BYTE_PER_FRACTAL / sizeof(Element));
    }

    PTO_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3];
    }

    PTO_HOST_DEVICE LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
            LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3];
    }

    PTO_HOST_DEVICE int64_t orgShape(int idx) const { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &orgShape(int idx) { return orgShape_.shape[idx]; }
    PTO_HOST_DEVICE Shape shape() const { return shape_; }
    PTO_HOST_DEVICE Shape &shape() { return shape_; }
    PTO_HOST_DEVICE int64_t shape(int idx) const { return shape_.shape[idx]; }
    PTO_HOST_DEVICE int64_t &shape(int idx) { return shape_.shape[idx]; }
    PTO_HOST_DEVICE Stride stride() const { return stride_; }
    PTO_HOST_DEVICE Stride &stride() { return stride_; }
    PTO_HOST_DEVICE int64_t stride(int idx) const { return stride_.stride[idx]; }
    PTO_HOST_DEVICE int64_t &stride(int idx) { return stride_.stride[idx]; }
};

}  // namespace layout

namespace Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct LocalTensorBufferBase {
    template <class Element = half>
    PTO_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    PTO_DEVICE LocalTensorBufferBase() = default;
    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L1_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0A_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0B_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::BIAS_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0C_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::UB_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
    PTO_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2PIPE2GM> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::FIXBUF_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct Resource {
    AscendC::TPipe pipe;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf;

    PTO_DEVICE Resource()
    {
        pipe.Destroy();
    }
};

}  // namespace Arch

namespace Gemm {

template <class Element_, class Layout_, AscendC::TPosition Position_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition Position = Position_;
};

struct MmadAtlasA2 {
    using ArchTag = Arch::AtlasA2;
    static constexpr bool ASYNC = false;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsync {
    using ArchTag = Arch::AtlasA2;
    static constexpr bool ASYNC = true;
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2PreloadFixpipeQuant : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsyncFixpipe
    : public MmadAtlasA2PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

namespace helper {

template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>, "Unsupported accumulator selector");
};

template <>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template <>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template <class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported L1 alignment helper");
};

template <class Element>
struct L1AlignHelper<Element, layout::ND> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::Zn> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>, "Unsupported L1A type selector");
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::VectorLayout>> {
    using L1AType = GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::ND>> {
    using L1AType = GemmType<Element, layout::Zn, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::Zn>> {
    using L1AType = GemmType<Element, layout::Zn, AscendC::TPosition::A1>;
};

template <class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>, "Unsupported L1B type selector");
};

template <class Element>
struct L1BTypeSelector<GemmType<Element, layout::Zn>> {
    using L1BType = GemmType<Element, layout::Zn, AscendC::TPosition::A1>;
};

}  // namespace helper

namespace Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <class ArchTag, class ElementSrc, class ElementDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c quant mode");
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, float, half, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, float, bfloat16_t, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <class ArchTag, class GmType, class L1Type = typename helper::L1ATypeSelector<GmType>::L1AType>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1");
};

template <class Element>
struct CopyGmToL1<Arch::AtlasA2,
                  GemmType<Element, layout::ND>,
                  GemmType<Element, layout::Zn, AscendC::TPosition::A1>> {
    using LayoutDst = layout::Zn;
    using LayoutSrc = layout::ND;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    PTO_DEVICE CopyGmToL1() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                                   AscendC::GlobalTensor<Element> const &srcTensor,
                                   LayoutDst const &layoutDst,
                                   LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
                  GemmType<Element, layout::Zn>,
                  GemmType<Element, layout::Zn, AscendC::TPosition::A1>> {
    using LayoutDst = layout::Zn;
    using LayoutSrc = layout::Zn;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    PTO_DEVICE CopyGmToL1() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                                   AscendC::GlobalTensor<Element> const &srcTensor,
                                   LayoutDst const &layoutDst,
                                   LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
        uint32_t blockLen = layoutSrc.orgShape(0);
        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag,
                  GemmType<Element, layout::VectorLayout>,
                  GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    PTO_DEVICE CopyGmToL1() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                                   AscendC::GlobalTensor<Element> const &srcTensor,
                                   LayoutDst const &layoutDst,
                                   LayoutSrc const &layoutSrc)
    {
        (void)layoutSrc;
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0a");
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag,
                   GemmType<Element, layout::Zn, AscendC::TPosition::A1>,
                   GemmType<Element, layout::Zz, AscendC::TPosition::A2>> {
    using LayoutDst = layout::Zz;
    using LayoutSrc = layout::Zn;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    PTO_DEVICE CopyL1ToL0A() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                                   AscendC::LocalTensor<Element> const &srcTensor,
                                   LayoutDst const &layoutDst,
                                   LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, GemmType<Element, layout::Zn, AscendC::TPosition::A1>>
    : CopyL1ToL0A<ArchTag,
                  GemmType<Element, layout::Zn, AscendC::TPosition::A1>,
                  GemmType<Element, layout::Zz, AscendC::TPosition::A2>> {};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0b");
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag,
                   GemmType<int8_t, layout::Zn, AscendC::TPosition::A1>,
                   GemmType<int8_t, layout::Nz, AscendC::TPosition::B2>> {
    using Element = int8_t;
    using LayoutDst = layout::Nz;
    using LayoutSrc = layout::Zn;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    PTO_DEVICE CopyL1ToL0B() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
                                   AscendC::LocalTensor<Element> const &srcTensor,
                                   LayoutDst const &layoutDst,
                                   LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, GemmType<int8_t, layout::Zn, AscendC::TPosition::A1>>
    : CopyL1ToL0B<ArchTag,
                  GemmType<int8_t, layout::Zn, AscendC::TPosition::A1>,
                  GemmType<int8_t, layout::Nz, AscendC::TPosition::B2>> {};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToFP {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to fixpipe buffer");
};

template <class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToFP<ArchTag,
                  GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
                  GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_FP = BYTE_PER_BLK_FP / sizeof(ElementSrc);

    PTO_DEVICE CopyL1ToFP() = default;

    PTO_DEVICE void operator()(AscendC::LocalTensor<ElementDst> dstTensor,
                                   AscendC::LocalTensor<ElementSrc> srcTensor,
                                   LayoutDst layoutDst,
                                   LayoutSrc layoutSrc)
    {
        (void)layoutSrc;
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = (layoutDst.shape(0) + ELE_NUM_PER_FP - 1) / ELE_NUM_PER_FP;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
          bool ReluEnable = false>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm");
};

template <class ElementAccumulator_, class ElementDst_, ScaleGranularity Granularity_, bool ReluEnable_>
struct CopyL0CToGm<Arch::AtlasA2,
                   ElementAccumulator_,
                   GemmType<ElementDst_, layout::ND>,
                   Granularity_,
                   ReluEnable_> {
    using ArchTag = Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = layout::Zn;
    using LayoutDst = layout::ND;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, Granularity_>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0f;
        PTO_HOST_DEVICE Params() = default;
        PTO_HOST_DEVICE explicit Params(float scale_) : scale(scale_) {}
    };

    Params params;

    PTO_DEVICE CopyL0CToGm() = default;
    PTO_DEVICE CopyL0CToGm(Params const &params_) : params(params_) {}

    PTO_DEVICE void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
                                   AscendC::LocalTensor<ElementSrc> const &src,
                                   LayoutDst const &dstLayout,
                                   LayoutSrc const &srcLayout,
                                   uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }

    PTO_DEVICE void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
                                   AscendC::LocalTensor<ElementSrc> const &src,
                                   AscendC::LocalTensor<uint64_t> cbufWorkspace,
                                   LayoutDst const &dstLayout,
                                   LayoutSrc const &srcLayout,
                                   uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, cbufWorkspace, intriParams);
    }
};

template <class ArchTag, class AType, class BType, class CType, class BiasType = void>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyGmToL1A = Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};

template <class ArchTag, class AType, class BType, class CType, class BiasType = void,
          ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR>
struct QuantTileCopy : public TileCopyGemm<ArchTag, AType, BType, CType, BiasType> {
    using Base = TileCopyGemm<ArchTag, AType, BType, CType, BiasType>;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using CopyL0CToGm = Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;
    using CopyL1ToFP = Tile::CopyL1ToFP<ArchTag,
        GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
        GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>>;
};

template <class ArchTag, class AType, class BType, class BiasType = void>
struct TileMmad {};

}  // namespace Tile

namespace Block {

template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    PtoShape3D problemShape;
    PtoShape2D tileMN;
    PtoShape2D loopsMN;

    PTO_DEVICE GemmIdentityBlockSwizzle() = default;

    PTO_DEVICE GemmIdentityBlockSwizzle(PtoShape3D const &problemShape_, PtoShape2D const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = pto_ext::CeilDiv(GetPtoShapeMN(problemShape), tileMN);
    }

    PTO_DEVICE void Update(PtoShape3D const &problemShape_, PtoShape2D const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = pto_ext::CeilDiv(GetPtoShapeMN(problemShape), tileMN);
    }

    PTO_DEVICE uint32_t GetCoreLoops() const
    {
        return static_cast<uint32_t>(loopsMN.shape[0] * loopsMN.shape[1]);
    }

    PTO_DEVICE PtoCoord2D GetBlockCoordMN(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        uint32_t loopM = static_cast<uint32_t>(loopsMN.shape[0]);
        uint32_t loopN = static_cast<uint32_t>(loopsMN.shape[1]);
        uint32_t mIdx = 0;
        uint32_t nIdx = 0;
        if constexpr (SwizzleDirection == 0) {
            uint32_t tileBlockLoop = pto_ext::CeilDiv(loopM, SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopN);
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopN);
            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopM - SwizzleOffset * tileBlockIdx;
            }
            mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopN - nIdx - 1;
            }
        } else {
            uint32_t tileBlockLoop = pto_ext::CeilDiv(loopN, SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopM);
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopM);
            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopN - SwizzleOffset * tileBlockIdx;
            }
            mIdx = inTileBlockIdx / nCol;
            nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopM - mIdx - 1;
            }
        }
        return MakePtoCoord2D(mIdx, nIdx);
    }

    PTO_DEVICE PtoShape2D GetActualBlockShapeMN(PtoCoord2D const &blockCoord)
    {
        uint32_t tileM = static_cast<uint32_t>(tileMN.shape[0]);
        uint32_t tileN = static_cast<uint32_t>(tileMN.shape[1]);
        uint32_t loopM = static_cast<uint32_t>(loopsMN.shape[0]);
        uint32_t loopN = static_cast<uint32_t>(loopsMN.shape[1]);
        uint32_t blockM = static_cast<uint32_t>(blockCoord.shape[0]);
        uint32_t blockN = static_cast<uint32_t>(blockCoord.shape[1]);
        uint32_t mActual = (blockM == (loopM - 1)) ?
            (GetPtoShapeM(problemShape) - blockM * tileM) : tileM;
        uint32_t nActual = (blockN == (loopN - 1)) ?
            (GetPtoShapeN(problemShape) - blockN * tileN) : tileN;
        return PtoShape2D(mActual, nActual);
    }
};

template <class DispatchPolicy,
          class L1TileShape,
          class L0TileShape,
          class AType,
          class BType,
          class CType,
          class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopyGemm<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

}  // namespace Block

}  // namespace Gemm

namespace Epilogue {

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2UnQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantSwigluQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantV2 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

namespace Tile {

template <class ArchTag, class CType, class ScaleType, class PerTokenScaleType, class DType>
struct TileCopy {};

template <class ArchTag, class ElementMulType, int Dummy = 0>
struct TileElemWiseMuls {};

}  // namespace Tile

namespace Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Block

}  // namespace Epilogue

}  // namespace pto_ext

namespace pto_ext::support {

constexpr uint64_t kL2Offset = pto_ext::L2_OFFSET;

struct NoopCallback {
    PTO_DEVICE void operator()() const {}
};

}  // namespace pto_ext::support

#endif // DISPATH_POLICY_CUSTOM_HPP
