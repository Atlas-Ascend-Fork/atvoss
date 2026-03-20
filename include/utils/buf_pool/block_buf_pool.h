/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVOSS_BLOCK_BUFFER_POOL_H
#define ATVOSS_BLOCK_BUFFER_POOL_H

namespace Atvoss {

template <int TILE_NUM, int TILE_SIZE>
class BlockBufferEx {
private:
    static_assert(TILE_SIZE % 32 == 0, "[ERROR]: [Atvoss][LoopBuffer] Tile size must be a multiple of 32\n");
    static_assert(TILE_NUM > 0, "[ERROR]: [Atvoss][LoopBuffer] Tile number must bigger than 0\n");

public:
    static constexpr uint64_t BLOCK_LEN = TILE_SIZE;
    __aicore__ inline BlockBufferEx()
    {}
    __aicore__ inline ~BlockBufferEx()
    {}

    __aicore__ inline void Init()
    {
        GetTPipePtr()->InitBuffer(tbuf_, TILE_SIZE * TILE_NUM);
        tensorPool_ = tbuf_.Get<uint8_t>();
    }

    template <typename T>
    __aicore__ inline void AllocTensor(AscendC::LocalTensor<T>& inTensor, uint32_t bufferId)
    {
        inTensor = tensorPool_[bufferId * BLOCK_LEN].template ReinterpretCast<T>();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_;
    AscendC::LocalTensor<uint8_t> tensorPool_;
};
} // namespace Atvoss

#endif // ATVOSS_BLOCK_BUFFER_POOL_H