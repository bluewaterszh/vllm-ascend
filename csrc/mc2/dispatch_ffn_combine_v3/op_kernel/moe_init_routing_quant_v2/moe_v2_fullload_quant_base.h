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
 * \file moe_v2_fullload_quant_base.h
 * \brief
 */
#ifndef MOE_V2_FULL_LOAD_QUANT_BASE_H
#define MOE_V2_FULL_LOAD_QUANT_BASE_H

#include "kernel_operator.h"

#include "moe_v2_pto_sort.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2FullLoadQuantBase {
 public:
  __aicore__ inline MoeV2FullLoadQuantBase(){};

 protected:
  __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                  GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                  const MoeInitRoutingQuantV2TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void ProcessBase();
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOutIdx();
  __aicore__ inline void CopyOutEmpty();
  __aicore__ inline void ComputeExpertTokenCountOrCumsum();

 protected:
  const InnerMoeV2GatherOutComputeTilingData* gatherOutTilingData;

  TPipe* pipe;
  int64_t tileLength;
  int64_t bufferNum = 1;
  int64_t totalLength;
  int64_t coreNum;
  int64_t sortNum;
  int64_t blockIdx;
  int64_t needCoreNum;
  int64_t coreRows;
  int64_t perCoreRows;
  int64_t k;
  int64_t n;
  int64_t cols;
  int64_t activateRows;
  int64_t expertNum;
  int64_t expertCapacity;

  TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
  TBuf<TPosition::VECCALC> tempBuffer;
  TBuf<TPosition::VECCALC> sortedBuffer;
  TQue<QuePosition::VECIN, 1> xCopyInQueue;
  TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue;
  TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue;
  TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue;
  TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue;

  GlobalTensor<int32_t> expertIdxGm;
  GlobalTensor<int8_t> expandedXGm;
  GlobalTensor<int32_t> expandedRowIdxGm;
  GlobalTensor<int32_t> expandedExpertIdxGm;
  GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
  GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t dropPadMode = 0;
  static constexpr int64_t DST_BLK_STRIDE = 1;
  static constexpr int64_t DST_REP_STRIDE = 8;
  static constexpr int64_t FOUR_BLOCK_BYTES = 128;
};

__aicore__ inline void MoeV2FullLoadQuantBase::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  pto_detail::PtoLoadVector(inLocal[0], expertIdxGm, this->totalLength);
  ArithProgression<int32_t>(inLocal[this->sortNum], 0, 1, this->totalLength);
  sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::SortCompute() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertIdxLocal = inLocal[0];
  LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum].template ReinterpretCast<uint32_t>();
  LocalTensor<float> packedSortLocal = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  LocalTensor<float> mergeTmpLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));

  LocalTensor<int32_t> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue.AllocTensor<int32_t>();
  LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue.AllocTensor<uint32_t>();
  pto_detail::PtoSortInt32AscendingUB(expertIdxLocal,
                                      rowIdxLocal,
                                      expandedExpertIdxLocal,
                                      expandDstToSrcRowLocal,
                                      packedSortLocal,
                                      mergeTmpLocal,
                                      this->totalLength);
  expandedExpertIdxCopyOutQueue.EnQue<int32_t>(expandedExpertIdxLocal);

  LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.AllocTensor<uint32_t>();
  LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
  LocalTensor<int32_t> expandDstToSrcRowLocalInt32 = expandDstToSrcRowLocal.ReinterpretCast<int32_t>();
  LocalTensor<int32_t> rowSortScratchLocal = expertIdxLocal;
  ArithProgression<int32_t>(inLocal[this->sortNum], 0, 1, this->totalLength);
  AscendC::PipeBarrier<PIPE_V>();
  pto_detail::PtoSortInt32AscendingUB(expandDstToSrcRowLocalInt32,
                                      rowIdxLocal,
                                      rowSortScratchLocal,
                                      expandedRowIdxU32,
                                      packedSortLocal,
                                      mergeTmpLocal,
                                      this->totalLength);
  expandedRowIdxCopyOutQueue.EnQue<uint32_t>(expandedRowIdx);
  sortDataCopyInQueue.FreeTensor(inLocal);

  expandDstToSrcRowQueue.FreeTensor(expandDstToSrcRowLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutIdx() {
  LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue.DeQue<int32_t>();
  pto_detail::PtoStoreVector(expandedRowIdxGm, expandedRowIdx, this->totalLength);
  expandedRowIdxCopyOutQueue.EnQue(expandedRowIdx);
}

__aicore__ inline void MoeV2FullLoadQuantBase::ComputeExpertTokenCountOrCumsum() {
  LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue.AllocTensor<int32_t>();

  int64_t expertNumAlign = Align(this->expertNum, sizeof(int32_t));
  Duplicate(expertTokensCount, 0, expertNumAlign);
  SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

  int32_t lastExpertId = expandedExpertIdx.GetValue(0);
  int64_t tokenCount = 0;
  int64_t lastExpertCount = 0;
  for (int64_t i = 0; i < this->totalLength; i++) {
    int32_t curExpertId = expandedExpertIdx.GetValue(i);
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
  expertTokensCopyOutQueue.FreeTensor(expertTokensCount);
  expandedExpertIdxCopyOutQueue.FreeTensor(expandedExpertIdx);
}

__aicore__ inline void MoeV2FullLoadQuantBase::CopyOutEmpty() {
  LocalTensor<int32_t> outLocal = expandedExpertIdxCopyOutQueue.DeQue<int32_t>();
  expandedExpertIdxCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeV2FullLoadQuantBase::InitBase(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                        GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                        GM_ADDR workspace,
                                                        const MoeInitRoutingQuantV2TilingData* tilingData,
                                                        TPipe* tPipe) {
  this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
  this->k = tilingData->k;
  this->n = tilingData->n;
  this->cols = tilingData->cols;
  this->needCoreNum = this->gatherOutTilingData->needCoreNum;
  this->perCoreRows = this->gatherOutTilingData->perCoreRows;
  this->activateRows = this->gatherOutTilingData->activateRows;
  if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
    this->coreRows = this->gatherOutTilingData->lastCoreRows;
  } else {
    this->coreRows = this->gatherOutTilingData->perCoreRows;
  }
  this->expertNum = tilingData->expertNum;
  this->dropPadMode = tilingData->dropPadMode;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;

  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->pipe = tPipe;

  expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);

  expandedXGm.SetGlobalBuffer((__gm__ int8_t*)expandedX);
  expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx, this->tileLength);
  if (this->expertTokensCountOrCumsumFlag > 0) {
    // dropless
    expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                Align(this->expertNum, sizeof(int32_t)));
  }

  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum * sizeof(int32_t);

  pipe->InitBuffer(expandedRowIdxCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(expandedExpertIdxCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(expertTokensCopyOutQueue, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
  pipe->InitBuffer(expandDstToSrcRowQueue, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);
  pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
  pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);
}

__aicore__ inline void MoeV2FullLoadQuantBase::ProcessBase() {
  if (this->blockIdx < this->needCoreNum) {
    CopyIn();
    SortCompute();
    if (this->blockIdx == 0) {
      CopyOutIdx();
    }
    if (this->blockIdx == this->needCoreNum - 1 && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
      ComputeExpertTokenCountOrCumsum();
    } else {
      CopyOutEmpty();
    }
  }
}

}  // namespace MoeInitRoutingQuantV2
#endif  // MOE_V2_FULL_LOAD_QUANT_BASE_H