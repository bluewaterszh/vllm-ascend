/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v2_sort_one_core.h
 * \brief
 */
#ifndef INNER_MOE_V2_SORT_ONE_CORE_H
#define INNER_MOE_V2_SORT_ONE_CORE_H

#include "moe_v2_mrgsort.h"
#include "moe_v2_pto_sort.h"
#include "moe_v2_sort_base.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
using namespace optiling;
class MoeV2SortOneCore : public MoeV2SortBase {
 public:
  __aicore__ inline MoeV2SortOneCore(){};
  template <typename TilingData>
  __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                              GM_ADDR workspace, const TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void SortCompute();
  __aicore__ inline void CopyOut();

 private:
  int64_t sortNum;
  int64_t blockIdx;
};

__aicore__ inline void MoeV2SortOneCore::CopyIn() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
  pto_detail::PtoLoadVector(inLocal[0], expertIdxGm, this->totalLength);

  LocalTensor<int32_t> rowIdxLocal = inLocal[this->sortNum];
  ArithProgression<int32_t>(rowIdxLocal, 0, 1, this->sortNum);
  sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::SortCompute() {
  LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
  LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
  LocalTensor<uint32_t> sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();

  LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.AllocTensor<int32_t>();
  LocalTensor<int32_t> sortedExpertForSourceRowLocal = outLocal[0];
  LocalTensor<uint32_t> expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
  LocalTensor<float> packedSortLocal = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
  LocalTensor<float> mergeTmpLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));

  pto_detail::PtoSortInt32AscendingUB(expertForSourceRowLocal,
                                      sourceRowLocal,
                                      sortedExpertForSourceRowLocal,
                                      expandDstToSrcRowLocal,
                                      packedSortLocal,
                                      mergeTmpLocal,
                                      this->totalLength);
  sortDataCopyOutQueue.EnQue<int32_t>(outLocal);
  sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2SortOneCore::CopyOut() {
  LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
  pto_detail::PtoStoreVector(sortedexpertIdxGm, outLocal[0], this->totalLength);
  pto_detail::PtoStoreVector(expandDstToSrcRowGm, outLocal[this->sortNum], this->totalLength);
  sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename TilingData>
__aicore__ inline void MoeV2SortOneCore::Init(GM_ADDR expertIdx, GM_ADDR expertTokensCountOrCumsum,
                                              GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                              const TilingData* tilingData, TPipe* tPipe) {
  this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
  this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
  this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
  this->totalLength = tilingData->n * tilingData->k;
  this->coreNum = tilingData->coreNum;
  this->pipe = tPipe;
  this->n = tilingData->n;
  this->k = tilingData->k;
  this->expertNum = tilingData->expertNum;
  this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
  this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

  expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx, this->tileLength);
  sortedexpertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), this->tileLength);
  expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace) + this->tileLength,
                                      this->tileLength);

  if (this->blockIdx == this->coreNum - 1) {
    if (this->expertTokensCountOrCumsumFlag > 0) {
      expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensCountOrCumsum,
                                                  Align(this->expertNum, sizeof(int32_t)));
      InitGlobalMemory(expertTokensCountOrCumsumGm, this->expertNum, 0);
    }
    if (this->expertTokensBeforeCapacityFlag == 1) {
      expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t*)expertTokensBeforeCapacity,
                                                   Align(this->expertNum, sizeof(int32_t)));
      InitGlobalMemory(expertTokensBeforeCapacityGm, this->expertNum, 0);
    }
  }
  // key and value
  int64_t kvFactor = 2;
  int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
  pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
  pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
  pipe->InitBuffer(tempBuffer, buffSize);
  pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeV2SortOneCore::Process() {
  if (get_block_idx() + get_subblockid() * get_block_num() < 1) {
    CopyIn();
    SortCompute();
    CopyOut();
  }
  this->SyncAll();
}
}  // namespace MoeInitRoutingQuantV2
#endif  // INNER_MOE_V2_SORT_ONE_CORE_H