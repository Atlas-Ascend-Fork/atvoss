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
    __aicore__ inline BlockBufferEx() : actualBufSize_(TILE_SIZE)
    {}
    __aicore__ inline ~BlockBufferEx()
    {}

    __aicore__ inline void Init()
    {
        actualBufSize_ = TILE_SIZE;
        GetTPipePtr()->InitBuffer(tbuf_, TILE_SIZE * TILE_NUM);
        tensorPool_ = tbuf_.Get<uint8_t>();
    }

    // Reduce 初始化
    __aicore__ inline void Init(uint32_t actualBufSize, uint32_t preBufSize, uint32_t postBufSize, uint32_t resBufSize)
    {
        actualBufSize_ = actualBufSize;
        GetTPipePtr()->InitBuffer(tbuf_, actualBufSize_);
        preBufPool_ = tbuf_.Get<uint8_t>();
        postBufPool_ = preBufPool_[preBufSize];
        resBuf_ = preBufPool_[preBufSize + postBufSize];
        cache_ = preBufPool_[preBufSize + postBufSize + resBufSize];
    }

    template <typename T>
    __aicore__ inline void AllocTensor(AscendC::LocalTensor<T>& inTensor, uint32_t bufferId)
    {
        inTensor = tensorPool_[bufferId * actualBufSize_].template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T>& AllocPreTensor(uint8_t bufId)
    {
        int32_t offset = 0;  // TODO
        return preBufPool_[offset].template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T>& AllocPostTensor(uint8_t bufId)
    {
        // int32_t offset = bufId * tiling_->resultBlock; // TODO
        int32_t offset = 0;
        return postBufPool_[offset].template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T>& AllocResTensor()
    {
        return resBuf_.template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T>& AllocCacheTensor()
    {
        return cache_.template ReinterpretCast<T>();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf_;
    AscendC::LocalTensor<uint8_t> tensorPool_;
    // reduce 使用
    uint32_t actualBufSize_;   // 运行时实际使用的buf大小
    AscendC::LocalTensor<uint8_t> preBufPool_;
    AscendC::LocalTensor<uint8_t> postBufPool_;
    AscendC::LocalTensor<uint8_t> resBuf_;
    AscendC::LocalTensor<uint8_t> cache_;
};
} // namespace Atvoss

#endif // ATVOSS_BLOCK_BUFFER_POOL_H