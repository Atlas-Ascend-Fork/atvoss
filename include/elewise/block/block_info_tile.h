/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_BLOCK_TENSOR_H
#define ATVOSS_BLOCK_TENSOR_H
#if !defined(__ATVOSS_HOST_ONLY__)
#include "operators/tensor_evaluator.h"
#endif

#include "utils/layout/layout.h"

namespace Atvoss::Ele {

template <typename T, typename L = Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<1, 1, 1>>>
class BlockTensor {
public:
    using PrimType = T;
    using LayoutType = L;

#if !defined(__ATVOSS_HOST_ONLY__)
    __aicore__ inline BlockTensor() = default;

    __aicore__ inline BlockTensor(__gm__ uint8_t* gm)
    {
        gmAddr_ = gm;
    }

    __aicore__ inline AscendC::LocalTensor<T>& GetUbTensor()
    {
        return ubTensor_;
    }

    __aicore__ inline void CopyIn(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        Atvoss::Tile::CopyIn(ubTensor_, gmTensor[curGmOffset], copyLen);
    }

    __aicore__ inline void CopyOut(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        Atvoss::Tile::CopyOut(gmTensor[curGmOffset], ubTensor_, copyLen);
    }

private:
    AscendC::LocalTensor<T> ubTensor_;
    __gm__ uint8_t* gmAddr_;
#endif
};
} // namespace Atvoss::Ele
#endif