/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_fullload_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_H
#define MOE_V2_FULL_LOAD_QUANT_H

#include "moe_v2_fullload_quant_base.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadQuant : public MoeV2FullLoadQuantBase {
 public:
  __aicore__ inline MoeV2FullLoadQuant(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX,
                              GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                              const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void Compute(int64_t xLocalLength);
  __aicore__ inline void LoadXRows(const LocalTensor<T>& xLocal, int64_t startXRow, int64_t rowCount,
                                   int64_t inFactor);
  __aicore__ inline void StoreExpandedXRow(int32_t outIndex, const LocalTensor<int8_t>& outLocal,
                                           int64_t localOffset);
  __aicore__ inline void CopyOutX();

 private:
  TQue<QuePosition::VECOUT, 1> floatQueue;
  TQue<QuePosition::VECOUT, 1> halfQueue;
  TQue<QuePosition::VECOUT, 1> inputXCopyOutQueue;

  GlobalTensor<T> xGm;
  GlobalTensor<float> scaleGm;
  GlobalTensor<float> offsetGm;

  float scale;
  float offset;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Compute(int64_t xLocalLength) {
  LocalTensor<T> inLocal = xCopyInQueue.DeQue<T>();
  LocalTensor<int8_t> outLocal = inputXCopyOutQueue.AllocTensor<int8_t>();
  LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
  LocalTensor<half> halfLocal = halfQueue.AllocTensor<half>();

  uint32_t elements = Align(this->cols, sizeof(int8_t)) * xLocalLength;
  if constexpr (IsSameType<T, bfloat16_t>::value) {
    pto_detail::PtoCastVector(floatLocal, inLocal, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoCastVector(halfLocal, floatLocal, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoMulVector(halfLocal, halfLocal, elements, static_cast<half>(this->scale));
    pto_detail::PtoAddScalarVector(halfLocal, halfLocal, elements, static_cast<half>(this->offset));
    LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
    pto_detail::PtoCastVector(intLocal, halfLocal, elements, pto::RoundMode::CAST_RINT);
    SetDeqScale((half)1.000000e+00f);
    pto_detail::PtoPipeBarrier<PIPE_V>();
    pto_detail::PtoCastVector(halfLocal, intLocal, elements, pto::RoundMode::CAST_RINT);
    pto_detail::PtoCastVector(outLocal, halfLocal, elements, pto::RoundMode::CAST_RINT);
  } else if constexpr (IsSameType<T, float>::value) {
    pto_detail::PtoCastVector(halfLocal, inLocal, elements, pto::RoundMode::CAST_NONE);
    pto_detail::PtoMulVector(halfLocal, halfLocal, elements, static_cast<half>(this->scale));
    pto_detail::PtoAddScalarVector(halfLocal, halfLocal, elements, static_cast<half>(this->offset));
    pto_detail::PtoCastVector(outLocal, halfLocal, elements, pto::RoundMode::CAST_RINT);
  } else {
    pto_detail::PtoMulVector(inLocal, inLocal, elements, static_cast<T>(this->scale));
    pto_detail::PtoAddScalarVector(inLocal, inLocal, elements, static_cast<T>(this->offset));
    pto_detail::PtoCastVector(outLocal, inLocal, elements, pto::RoundMode::CAST_RINT);
  }
  inputXCopyOutQueue.EnQue(outLocal);
  xCopyInQueue.FreeTensor(inLocal);
  floatQueue.FreeTensor(floatLocal);
  halfQueue.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::LoadXRows(const LocalTensor<T>& xLocal,
                                                        int64_t startXRow,
                                                        int64_t rowCount,
                                                        int64_t inFactor) {
  uint32_t dstStride = (inFactor * sizeof(T) - AlignBytes(this->cols, sizeof(T))) / BLOCK_BYTES;
  DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(rowCount),
                                    static_cast<uint32_t>(this->cols * sizeof(T)), 0, dstStride, 0};
  DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, 0};
  DataCopyPad(xLocal, xGm[startXRow * this->cols], dataXCopyParams, dataXCopyPadParams);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::StoreExpandedXRow(int32_t outIndex,
                                                                const LocalTensor<int8_t>& outLocal,
                                                                int64_t localOffset) {
  DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};
  DataCopyPad(expandedXGm[outIndex * this->cols], outLocal[localOffset], intriParams);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::CopyOutX() {
  LocalTensor<T> xLocal = xCopyInQueue.AllocTensor<T>();
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.DeQue<int32_t>();
  int64_t inFactor = Align(this->cols, sizeof(int8_t));
  int64_t curRowsStart = this->blockIdx * this->perCoreRows;
  int64_t startXRow = curRowsStart / this->k;
  int64_t endXRow = (curRowsStart + this->coreRows - 1) / this->k;

  LoadXRows(xLocal, startXRow, endXRow - startXRow + 1, inFactor);
  xCopyInQueue.EnQue(xLocal);
  Compute(endXRow - startXRow + 1);
  LocalTensor<int8_t> outLocal = inputXCopyOutQueue.DeQue<int8_t>();
  int64_t k = 0;
  for (int64_t i = startXRow; i <= endXRow; i++) {
    for (; k < this->perCoreRows && curRowsStart / this->k == i; curRowsStart++, k++) {
      int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
      if (outIndex < this->activateRows) {
        StoreExpandedXRow(outIndex, outLocal, (i - startXRow) * inFactor);
      }
    }
  }
  expandedRowIdxCopyOutQueue.FreeTensor(expandedRowIdx);
  inputXCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                   GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                   GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                                   const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe) {
  this->InitBase(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, tPipe);
  xGm.SetGlobalBuffer((__gm__ T*)x);
  scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);
  offsetGm.SetGlobalBuffer((__gm__ float*)offset, 1);
  this->scale = scaleGm.GetValue(0);
  this->offset = offsetGm.GetValue(0);

  int64_t curRowsStart = this->blockIdx * this->perCoreRows;
  int64_t rowLength = (curRowsStart + this->coreRows - 1) / this->k - curRowsStart / this->k + 1;
  int64_t xAlignedCount = Align(this->cols, sizeof(int8_t));
  pipe->InitBuffer(xCopyInQueue, bufferNum, xAlignedCount * sizeof(T) * rowLength);
  pipe->InitBuffer(inputXCopyOutQueue, 1, xAlignedCount * sizeof(int8_t) * rowLength);
  pipe->InitBuffer(floatQueue, 1, xAlignedCount * sizeof(float) * rowLength);
  pipe->InitBuffer(halfQueue, 1, xAlignedCount * sizeof(half) * rowLength);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadQuant<T>::Process() {
  if (this->blockIdx < this->needCoreNum) {
    this->ProcessBase();
    CopyOutX();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif