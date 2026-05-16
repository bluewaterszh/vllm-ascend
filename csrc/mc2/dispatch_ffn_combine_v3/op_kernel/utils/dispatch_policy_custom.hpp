#ifndef DISPATH_POLICY_CUSTOM_HPP
#define DISPATH_POLICY_CUSTOM_HPP

#include "kernel_operator.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#define CATLASS_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define CATLASS_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__ __aicore__

namespace Catlass {

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

class EmptyClass {};

template <uint32_t Align, class T>
CATLASS_HOST_DEVICE constexpr T CeilDiv(T value)
{
    return (value + static_cast<T>(Align) - 1) / static_cast<T>(Align);
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto CeilDiv(T lhs, U rhs)
{
    using Common = std::common_type_t<T, U>;
    Common lhsValue = static_cast<Common>(lhs);
    Common rhsValue = static_cast<Common>(rhs);
    return (lhsValue + rhsValue - 1) / rhsValue;
}

template <uint32_t Align, class T>
CATLASS_HOST_DEVICE constexpr T RoundUp(T value)
{
    return CeilDiv<Align>(value) * static_cast<T>(Align);
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto AlignUp(T value, U align)
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

    CATLASS_HOST_DEVICE constexpr explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    CATLASS_HOST_DEVICE constexpr Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    CATLASS_HOST_DEVICE explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != 0) {
                return true;
            }
        }
        return false;
    }

    CATLASS_HOST_DEVICE bool operator!() const
    {
        return !static_cast<bool>(*this);
    }

    CATLASS_HOST_DEVICE Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] + b[i];
        }
        return c;
    }

    CATLASS_HOST_DEVICE Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] - b[i];
        }
        return c;
    }

    CATLASS_HOST_DEVICE Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] * b[i];
        }
        return c;
    }

    CATLASS_HOST_DEVICE Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] / b[i];
        }
        return c;
    }

    CATLASS_HOST_DEVICE Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b[i];
        }
        return *this;
    }

    CATLASS_HOST_DEVICE bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    CATLASS_HOST_DEVICE Index &operator[](int dim)
    {
        return idx[dim];
    }

    CATLASS_HOST_DEVICE Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    template <int DIM>
    CATLASS_HOST_DEVICE Index &At()
    {
        return idx[DIM];
    }

    CATLASS_HOST_DEVICE Index &At(int dim)
    {
        return idx[dim];
    }

    template <int DIM>
    CATLASS_HOST_DEVICE Index const &At() const
    {
        return idx[DIM];
    }

    CATLASS_HOST_DEVICE Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    CATLASS_HOST_DEVICE auto GetCoordByAxis() const
    {
        Index values[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{values};
    }
};

template <class... Ts>
CATLASS_HOST_DEVICE constexpr auto MakeCoord(Ts... values)
{
    using Index = std::common_type_t<Ts...>;
    Index data[sizeof...(Ts)]{static_cast<Index>(values)...};
    return Coord<sizeof...(Ts), Index>{data};
}

template <int RANK, class Index, class LongIndex>
CATLASS_HOST_DEVICE Coord<RANK, Index, LongIndex> CeilDiv(
    Coord<RANK, Index, LongIndex> const &lhs,
    Coord<RANK, Index, LongIndex> const &rhs)
{
    Coord<RANK, Index, LongIndex> out;
    for (int i = 0; i < RANK; ++i) {
        out[i] = Catlass::CeilDiv(lhs[i], rhs[i]);
    }
    return out;
}

template <uint32_t ROW_ = 1, uint32_t COLUMN_ = 1>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;
    static constexpr int64_t COUNT = ROW * COLUMN;

    CATLASS_HOST_DEVICE static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<2, Index>;
    using LongIndex = typename Base::LongIndex;

    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;

    CATLASS_HOST_DEVICE MatrixCoord() = default;
    CATLASS_HOST_DEVICE MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}
    CATLASS_HOST_DEVICE MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}
    CATLASS_HOST_DEVICE MatrixCoord(LongIndex row, LongIndex column)
        : Base(MakeCoord(Index(row), Index(column))) {}

    CATLASS_HOST_DEVICE Index const &row() const { return this->At(ROW_INDEX); }
    CATLASS_HOST_DEVICE Index &row() { return this->At(ROW_INDEX); }
    CATLASS_HOST_DEVICE Index const &column() const { return this->At(COLUMN_INDEX); }
    CATLASS_HOST_DEVICE Index &column() { return this->At(COLUMN_INDEX); }

    CATLASS_HOST_DEVICE MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    CATLASS_HOST_DEVICE MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

CATLASS_HOST_DEVICE MatrixCoord CeilDiv(MatrixCoord const &lhs, MatrixCoord const &rhs)
{
    return MatrixCoord(Catlass::CeilDiv(lhs.row(), rhs.row()), Catlass::CeilDiv(lhs.column(), rhs.column()));
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

    CATLASS_HOST_DEVICE static Coord<3> ToCoord() { return MakeCoord(M, N, K); }
    CATLASS_HOST_DEVICE static Coord<2> ToCoordMN() { return MakeCoord(M, N); }
    CATLASS_HOST_DEVICE static Coord<2> ToCoordMK() { return MakeCoord(M, K); }
    CATLASS_HOST_DEVICE static Coord<2> ToCoordKN() { return MakeCoord(K, N); }
};

struct GemmCoord : public Coord<3, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<3, Index>;

    static constexpr int M_INDEX = 0;
    static constexpr int N_INDEX = 1;
    static constexpr int K_INDEX = 2;

    CATLASS_HOST_DEVICE GemmCoord() = default;
    CATLASS_HOST_DEVICE GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}
    CATLASS_HOST_DEVICE GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    CATLASS_HOST_DEVICE Index const &m() const { return this->At(M_INDEX); }
    CATLASS_HOST_DEVICE Index &m() { return this->At(M_INDEX); }
    CATLASS_HOST_DEVICE Index const &n() const { return this->At(N_INDEX); }
    CATLASS_HOST_DEVICE Index &n() { return this->At(N_INDEX); }
    CATLASS_HOST_DEVICE Index const &k() const { return this->At(K_INDEX); }
    CATLASS_HOST_DEVICE Index &k() { return this->At(K_INDEX); }

    CATLASS_HOST_DEVICE auto GetCoordMN() const { return this->template GetCoordByAxis<M_INDEX, N_INDEX>(); }
    CATLASS_HOST_DEVICE auto GetCoordMK() const { return this->template GetCoordByAxis<M_INDEX, K_INDEX>(); }
    CATLASS_HOST_DEVICE auto GetCoordKN() const { return this->template GetCoordByAxis<K_INDEX, N_INDEX>(); }
};

CATLASS_HOST_DEVICE GemmCoord CeilDiv(GemmCoord const &lhs, GemmCoord const &rhs)
{
    return GemmCoord(
        Catlass::CeilDiv(lhs.m(), rhs.m()),
        Catlass::CeilDiv(lhs.n(), rhs.n()),
        Catlass::CeilDiv(lhs.k(), rhs.k()));
}

namespace layout {

struct RowMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

    CATLASS_HOST_DEVICE RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

    CATLASS_HOST_DEVICE RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayout(Index rows, Index cols)
    {
        return RowMajor(rows, cols);
    }

    CATLASS_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    CATLASS_HOST_DEVICE RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
    }

    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
};

struct ColumnMajor {
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

    CATLASS_HOST_DEVICE ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

    CATLASS_HOST_DEVICE ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static ColumnMajor MakeLayout(Index rows, Index cols)
    {
        return ColumnMajor(rows, cols);
    }

    CATLASS_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    CATLASS_HOST_DEVICE ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
    }

    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
};

struct VectorLayout {
    static constexpr int RANK = 1;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
    using TensorCoord = Coord<RANK, Index>;

    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE VectorLayout(Index size = 0)
        : shape_(MakeCoord(size)), stride_(MakeCoord(LongIndex(1))) {}

    CATLASS_HOST_DEVICE VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    CATLASS_HOST_DEVICE LongIndex GetOffset(TensorCoord const &coord) const
    {
        return stride_[0] * coord[0];
    }

    CATLASS_HOST_DEVICE VectorLayout GetTileLayout(TensorCoord const &tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
};

struct nZ {
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE constexpr nZ(
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
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    CATLASS_HOST_DEVICE constexpr nZ(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static constexpr nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(orgRows,
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

    CATLASS_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    CATLASS_HOST_DEVICE nZ GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return nZ(tileOriShape, tileShape, stride());
    }

    CATLASS_HOST_DEVICE static nZ MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return nZ(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    CATLASS_HOST_DEVICE typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE typename OrgShape::Index &orgShape(int idx) { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
};

struct zN {
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE constexpr zN(
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
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    CATLASS_HOST_DEVICE constexpr zN(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static constexpr zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows,
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

    CATLASS_HOST_DEVICE static zN MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return zN(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    CATLASS_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    CATLASS_HOST_DEVICE zN GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2)));
        return zN(tileOriShape, tileShape, stride());
    }

    CATLASS_HOST_DEVICE typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE typename OrgShape::Index &orgShape(int idx) { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
};

struct zZ {
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    CATLASS_HOST_DEVICE constexpr zZ(
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
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    CATLASS_HOST_DEVICE constexpr zZ(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static constexpr zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(orgRows,
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

    CATLASS_HOST_DEVICE LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    CATLASS_HOST_DEVICE typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE typename OrgShape::Index &orgShape(int idx) { return orgShape_[idx]; }
    CATLASS_HOST_DEVICE Shape shape() const { return shape_; }
    CATLASS_HOST_DEVICE Shape &shape() { return shape_; }
    CATLASS_HOST_DEVICE typename Shape::Index shape(int idx) const { return shape_[idx]; }
    CATLASS_HOST_DEVICE typename Shape::Index &shape(int idx) { return shape_[idx]; }
    CATLASS_HOST_DEVICE Stride stride() const { return stride_; }
    CATLASS_HOST_DEVICE Stride &stride() { return stride_; }
    CATLASS_HOST_DEVICE typename Stride::Index stride(int idx) const { return stride_[idx]; }
    CATLASS_HOST_DEVICE typename Stride::Index &stride(int idx) { return stride_[idx]; }
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
    CATLASS_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE LocalTensorBufferBase() = default;
    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L1_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0A_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0B_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::BIAS_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::L0C_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        GetTPipePtr()->InitBuffer(buf, ArchTag::UB_SIZE);
        tensor = buf.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> : LocalTensorBufferBase {
    CATLASS_DEVICE LocalTensorBuffer()
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

    CATLASS_DEVICE Resource()
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
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::zN> {
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
struct L1ATypeSelector<GemmType<Element, layout::RowMajor>> {
    using L1AType = GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::zN>> {
    using L1AType = GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>, "Unsupported L1B type selector");
};

template <class Element>
struct L1BTypeSelector<GemmType<Element, layout::zN>> {
    using L1BType = GemmType<Element, layout::zN, AscendC::TPosition::A1>;
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
                  GemmType<Element, layout::RowMajor>,
                  GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE CopyGmToL1() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
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
                  GemmType<Element, layout::zN>,
                  GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE CopyGmToL1() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
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

    CATLASS_DEVICE CopyGmToL1() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
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
                   GemmType<Element, layout::zN, AscendC::TPosition::A1>,
                   GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE CopyL1ToL0A() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
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
struct CopyL1ToL0A<ArchTag, GemmType<Element, layout::zN, AscendC::TPosition::A1>>
    : CopyL1ToL0A<ArchTag,
                  GemmType<Element, layout::zN, AscendC::TPosition::A1>,
                  GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0b");
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag,
                   GemmType<int8_t, layout::zN, AscendC::TPosition::A1>,
                   GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE CopyL1ToL0B() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<Element> const &dstTensor,
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
struct CopyL1ToL0B<ArchTag, GemmType<int8_t, layout::zN, AscendC::TPosition::A1>>
    : CopyL1ToL0B<ArchTag,
                  GemmType<int8_t, layout::zN, AscendC::TPosition::A1>,
                  GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>> {};

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

    CATLASS_DEVICE CopyL1ToFP() = default;

    CATLASS_DEVICE void operator()(AscendC::LocalTensor<ElementDst> dstTensor,
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
                   GemmType<ElementDst_, layout::RowMajor>,
                   Granularity_,
                   ReluEnable_> {
    using ArchTag = Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, Granularity_>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0f;
        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE explicit Params(float scale_) : scale(scale_) {}
    };

    Params params;

    CATLASS_DEVICE CopyL0CToGm() = default;
    CATLASS_DEVICE CopyL0CToGm(Params const &params_) : params(params_) {}

    CATLASS_DEVICE void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
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

    CATLASS_DEVICE void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
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
    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    CATLASS_DEVICE GemmIdentityBlockSwizzle() = default;

    CATLASS_DEVICE GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = Catlass::CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = Catlass::CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) {
            uint32_t tileBlockLoop = Catlass::CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());
            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else {
            uint32_t tileBlockLoop = Catlass::CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());
            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
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

}  // namespace Catlass

namespace DispatchFFNCombineCompat {

constexpr uint64_t kL2Offset = Catlass::L2_OFFSET;

struct NoopCallback {
    CATLASS_DEVICE void operator()() const {}
};

}  // namespace DispatchFFNCombineCompat

#endif // DISPATH_POLICY_CUSTOM_HPP
