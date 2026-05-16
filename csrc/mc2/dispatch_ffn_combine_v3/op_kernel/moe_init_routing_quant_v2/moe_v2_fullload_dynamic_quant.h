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
 * \file moe_v2_fullload_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H
#define MOE_V2_FULL_LOAD_DYNAMIC_QUANT_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_pto_sort.h"
#include "moe_v2_sort_base.h"
namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
template <typename T>
class MoeV2FullLoadDynamicQuant : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2FullLoadDynamicQuant(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                              GM_ADDR workspace, const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void CopyOutXQuant1H();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();
  __aicore__ inline void Compute(LocalTensor<float>& smoothLocal);

 private:
  int64_t sortNum_;
  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData_;
  int64_t blockIdx_;
  int64_t needCoreNum_;
  int64_t coreRows_;
  int64_t perCoreRows_;
  int64_t k_;
  int64_t n_;
  int64_t cols_;
  int64_t cols_scale_;
  int64_t activateRows_;
  int64_t expertNum;
  int64_t expertCapacity;
  int64_t smoothType;
  int64_t colsAlign;

  TQue<QuePosition::VECIN, 1> xCopyInQueue_;
  TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
  TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;
  TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;
  TQue<QuePosition::VECIN, 1> smoothInQueue;
  TQue<QuePosition::VECOUT, 1> calcQueue;
  TQue<QuePosition::VECOUT, 1> inputXOutQueue;

  GlobalTensor<T> xGm_;
  GlobalTensor<int32_t> expertIdxGm_;
  GlobalTensor<float> quantSmoothGm;

  GlobalTensor<int8_t> expandedXGm_;
  GlobalTensor<int32_t> expandedRowIdxGm_;
  GlobalTensor<int32_t> expandedExpertIdxGm_;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t dropPadMode = 0;

  LocalTensor<uint32_t> expandDstToSrcRowLocal;
  LocalTensor<int32_t> expandedExpertIdxLocal;
};

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  pto_detail::PtoLoadVector(inLocal[0], expertIdxGm_, this->totalLength);
  ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
  sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::SortCompute() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertIdxLocal = inLocal[0];
  LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
  LocalTensor<float> packedSortLocal = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
  LocalTensor<float> mergeTmpLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum_));

  expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<int32_t>();
  expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
  pto_detail::PtoSortInt32AscendingUB(expertIdxLocal,
                                      rowIdxLocal,
                                      expandedExpertIdxLocal,
                                      expandDstToSrcRowLocal,
                                      packedSortLocal,
                                      mergeTmpLocal,
                                      this->totalLength);
  expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocal);

  LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<uint32_t>();
  LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
  LocalTensor<int32_t> expandDstToSrcRowLocalInt32 = expandDstToSrcRowLocal.ReinterpretCast<int32_t>();
  LocalTensor<int32_t> rowSortScratchLocal = expertIdxLocal;
  ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
  pto_detail::PtoPipeBarrier<PIPE_V>();
  pto_detail::PtoSortInt32AscendingUB(expandDstToSrcRowLocalInt32,
                                      rowIdxLocal,
                                      rowSortScratchLocal,
                                      expandedRowIdxU32,
                                      packedSortLocal,
                                      mergeTmpLocal,
                                      this->totalLength);
  expandedRowIdxCopyOutQueue_.EnQue<uint32_t>(expandedRowIdx);
  sortDataCopyInQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutIdx() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  pto_detail::PtoStoreVector(expandedRowIdxGm_, expandedRowIdx, this->totalLength);
  expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::ComputeExpertTokenCountOrCumsum() {
  expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
  LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue_.AllocTensor<int32_t>();

  int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
  Duplicate(expertTokensCount, 0, expertNumAlign);
  pto_detail::PtoSetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

  int32_t lastExpertId = expandedExpertIdxLocal.GetValue(0);
  int64_t tokenCount = 0;
  int64_t lastExpertCount = 0;
  for (int64_t i = 0; i < this->totalLength; i++) {
    int32_t curExpertId = expandedExpertIdxLocal.GetValue(i);
    tokenCount++;
    while (lastExpertId < curExpertId) {
      expertTokensCount.SetValue(lastExpertId, tokenCount - 1);
      if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_COUNT) {
        tokenCount = 1;
      }
      lastExpertId++;
    }
  }
  expertTokensCount.SetValue(lastExpertId, tokenCount);
  if (this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM) {
    lastExpertId++;
    while (lastExpertId < this->expertNum) {
      expertTokensCount.SetValue(lastExpertId, tokenCount);
      lastExpertId++;
    }
  }
  if (this->expertTokensCountOrCumsumFlag > 0) {
    pto_detail::PtoStoreVector(expertTokensCountOrCumsumGm, expertTokensCount, this->expertNum);
  }
  expertTokensCopyOutQueue_.FreeTensor(expertTokensCount);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutEmpty() {
  expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Compute(LocalTensor<float>& smoothLocal) {
  LocalTensor<float> inLocal = xCopyInQueue_.DeQue<float>();

  LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
  LocalTensor<int8_t> outLocal = inputXOutQueue.AllocTensor<int8_t>();
  LocalTensor<float> dynamicQuantLocal = outLocal[this->cols_].template ReinterpretCast<float>();

  if constexpr (!IsSameType<T, float>::value) {
    pto_detail::PtoCastVector(inLocal, inLocal.ReinterpretCast<T>()[colsAlign], this->cols_, pto::RoundMode::CAST_NONE);
  }

  if (smoothType != 0) {
    pto_detail::PtoMulElementwiseVector(inLocal, inLocal, smoothLocal, this->cols_);
  }

  pto_detail::PtoAbsVector(tempLocal, inLocal, this->cols_);

  pto_detail::PtoReduceMaxVector(dynamicQuantLocal, tempLocal, tempLocal, this->cols_);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;

  Duplicate<float>(dynamicQuantLocal, maxValue, 8);
  Duplicate<float>(tempLocal, maxValue, this->cols_);
  pto_detail::PtoPipeBarrier<PIPE_V>();

  pto_detail::PtoDivVector(tempLocal, inLocal, tempLocal, this->cols_);

  pto_detail::PtoCastVector(tempLocal.ReinterpretCast<half>(), tempLocal, this->cols_, pto::RoundMode::CAST_TRUNC);
  pto_detail::PtoCastVector(outLocal, tempLocal.ReinterpretCast<half>(), this->cols_, pto::RoundMode::CAST_ROUND);

  calcQueue.FreeTensor(tempLocal);
  inputXOutQueue.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::CopyOutXQuant1H() {
  expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
  expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdxLocal);

  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t curRowsEnd = curRowsStart + this->coreRows_ - 1;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = curRowsEnd / this->k_;

  LocalTensor<float> smoothLocal;
  if (smoothType == 1) {
    smoothLocal = smoothInQueue.AllocTensor<float>();
    pto_detail::PtoLoadVector(smoothLocal, quantSmoothGm, this->cols_);
    smoothInQueue.EnQue(smoothLocal);
    smoothLocal = smoothInQueue.DeQue<float>();
  }
  for (int64_t row = startXRow; row <= endXRow; row++) {
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
      pto_detail::PtoLoadVector(xLocal, xGm_[row * this->cols_], this->cols_);
    } else {
      pto_detail::PtoLoadVector(xLocal[colsAlign], xGm_[row * this->cols_], this->cols_);
    }

    xCopyInQueue_.EnQue<T>(xLocal);
    Compute(smoothLocal);

    LocalTensor<int8_t> outLocal = inputXOutQueue.DeQue<int8_t>();
    while (curRowsStart <= curRowsEnd && curRowsStart / this->k_ == row) {
      int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
      curRowsStart++;
      if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows_)) {
        continue;
      }
      pto_detail::PtoStoreVector(expandedXGm_[outIndex * this->cols_scale_], outLocal, this->cols_scale_);
    }

    xCopyInQueue_.FreeTensor(xLocal);
    inputXOutQueue.FreeTensor(outLocal);
  }
  expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR quantSmooth, GM_ADDR dynamicQuantScale,
                                                          GM_ADDR workspace,
                                                          const MoeInitRoutingQuantV2TilingData* tilingData,
                                                          TPipe* tPipe) {
  this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
  //this->blockIdx_ = GetBlockIdx();
  this->blockIdx_ = get_block_idx() + get_subblockid() * get_block_num();
  this->k_ = tilingData->k;
  this->n_ = tilingData->n;
  this->cols_ = tilingData->cols;
  this->cols_scale_ = this->cols_ + UB_ALIGN;
  this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;
  this->perCoreRows_ = this->gatherOutTilingData_->perCoreRows;
  this->activateRows_ = this->gatherOutTilingData_->activateRows;
  if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
    this->coreRows_ = this->gatherOutTilingData_->lastCoreRows;
  } else {
    this->coreRows_ = this->gatherOutTilingData_->perCoreRows;
  }
  this->expertNum = tilingData->expertNum;
  this->dropPadMode = tilingData->dropPadMode;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum_ = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->smoothType = tilingData->smoothType;
  this->colsAlign = Align(this->cols_, sizeof(T));
  this->pipe = tPipe;

  xGm_.SetGlobalBuffer((__gm__ T*)x);
  expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);

  expandedXGm_.SetGlobalBuffer((__gm__ int8_t*)expandedX);
  expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, this->tileLength);
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                Align(this->expertNum, sizeof(int32_t)));
  }
  quantSmoothGm.SetGlobalBuffer((__gm__ float*)quantSmooth);

  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum_ * sizeof(int32_t);

  int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
  int64_t startXRow = curRowsStart / this->k_;
  int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;

  pipe->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum, buffSize);
  pipe->InitBuffer(expertTokensCopyOutQueue_, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
  pipe->InitBuffer(expandDstToSrcRowQueue_, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
  pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
  pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);

  if constexpr (IsSameType<T, float>::value) {
    pipe->InitBuffer(xCopyInQueue_, 1, AlignBytes(this->cols_, sizeof(float)));
  } else {
    pipe->InitBuffer(xCopyInQueue_, 1, 2 * AlignBytes(this->cols_, sizeof(T)));
  }
  pipe->InitBuffer(smoothInQueue, 1, AlignBytes(this->cols_, sizeof(float)));
  pipe->InitBuffer(calcQueue, 1, AlignBytes(this->cols_, sizeof(float)));
  pipe->InitBuffer(inputXOutQueue, 1, AlignBytes(this->cols_scale_, sizeof(int8_t)));
}

template <typename T>
__aicore__ inline void MoeV2FullLoadDynamicQuant<T>::Process() {
  if (this->blockIdx_ < this->needCoreNum_) {
    CopyIn();
    SortCompute();
    if (this->blockIdx_ == 0) {
      CopyOutIdx();
    }
    if (this->blockIdx_ == this->needCoreNum_ - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
      ComputeExpertTokenCountOrCumsum();
    } else {
      CopyOutEmpty();
    }
    CopyOutXQuant1H();
  }
}
}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_DYNAMIC_QUANT_FULL_LOAD_H