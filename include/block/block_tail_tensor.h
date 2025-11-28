/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_BLOCK_TILE_TENSOR_H
#define Atvoss_BLOCK_TILE_TENSOR_H

#include "tile/tile_operator.h"
#include "utils/layout/layout.h"

namespace Atvoss::Block {

template <typename T, typename L = Atvoss::Layout::TailLayout<Atvoss::Layout::VariableRankExtents<1>>>
class TailTensor {
public:
    using PrimType = T;
    using LayoutType = L;

    __aicore__ inline TailTensor() = default;

    __aicore__ inline TailTensor(AscendC::GlobalTensor<T> gmTensor, LayoutType layout,
                             ParamUsage usage = Atvoss::ParamUsage::in, int index = 0)
    {
        SetGmTensor(gmTensor);
        index_ = index;
        usage_ = usage;
        layout_ = layout;
    }

    __aicore__ inline TailTensor(int index = 0)
    {
        index_ = index;
    }

    __aicore__ inline void SetSize(std::size_t size)
    {
        size_ = size;
    }

    __aicore__ inline uint64_t

    GetSize() const
    {
        return size_;
    }

    __aicore__ inline uint64_t

    GetCurGmOffset() const
    {
        return curGmOffset_;
    }

    __aicore__ inline AscendC::LocalTensor<T>&

    GetUbTensor()
    {
        return ubTensor_;
    }

    __aicore__ inline AscendC::GlobalTensor<T>

    GetGmTensor() const
    {
        return gmTensor_;
    }

    __aicore__ inline ParamUsage

    GetParamUsage() const
    {
        return usage_;
    }

    __aicore__ inline void SetGmTensor(AscendC::GlobalTensor<T> gmTensor)
    {
        gmTensor_ = gmTensor;
    }

    __aicore__ inline void SetUbTensor(AscendC::LocalTensor<T> ubTensor)
    {
        ubTensor_ = ubTensor;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t copyCnt)
    {
        Atvoss::Tile::CopyIn(ubTensor_, gmTensor_[offset], copyCnt);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t copyCnt)
    {
        Atvoss::Tile::CopyIn(gmTensor_[offset], ubTensor_, copyCnt);
    }

    __aicore__ inline LayoutType GetLayout()
    {
        return layout_;
    }

private:
    AscendC::LocalTensor<T> ubTensor_;
    AscendC::GlobalTensor<T> gmTensor_;
    uint64_t size_;
    uint64_t curGmOffset_;
    int index_ = 0;
    ParamUsage usage_;
    LayoutType layout_;
};
}  // namespace Atvoss::Block
#endif  //Atvoss_BLOCK_TILE_TENSOR_H
