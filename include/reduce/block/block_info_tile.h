/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_BLOCK_TENSOR_H
#define ATVOSS_REDUCE_BLOCK_TENSOR_H

#include "utils/layout/layout.h"

namespace Atvoss::Reduce {

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

    // 普通连续CopyIn
    __aicore__ inline void CopyIn(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        AscendC::DataCopy(ubTensor_, gmTensor[curGmOffset], static_cast<uint32_t>(copyLen));
    }

    // ARAR→AARR 转置 CopyIn (DataCopyPad)
    //
    // GM(ARAR交织布局):
    //   [A₀R₀..R_{k-1} | 间隔srcStride | A₁R₀..R_{k-1} | ...]
    //   segCount 个 A-分段, 每段 segSize 个 R 元素, 段间 stride = srcStride
    //
    // UB(AARR分块布局):
    //   [A₀R₀..R_{k-1} | A₁R₀..R_{k-1} | ...]  连续存放, dstStride = segSize
    //
    // DataCopyPad 用硬件 DMA 一次完成: segCount 个 block, 每 block segSize 元素,
    //   GM 读取 stride = srcStride, UB 写入 stride = segSize (连续)
    //
    // 参数来源 (通过 ReduceSchLoopInfo 确定 split axis, 通过 cfg.shape 确定 stride):
    //   segCount  = ubFactorA  (A维UB切分数)
    //   segSize   = ubFactorR  (R维UB切分数)
    //   srcStride = cfg.srcStride[loopAAxis[loopACount-1]]  (A切分轴在GM中的stride)
    //   dstStride = segSize    (UB中AARR连续存放)
    //
    __aicore__ inline void CopyIn(uint64_t curGmOffset, uint64_t copyLen,
                                  uint64_t segCount, uint64_t segSize,
                                  uint64_t srcStride, uint64_t dstStride)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));

        if (segCount <= 1 || srcStride == 0 || srcStride == segSize) {
            // 无转置: 连续搬运
            AscendC::DataCopy(ubTensor_, gmTensor[curGmOffset], static_cast<uint32_t>(copyLen));
        } else {
            // ARAR→AARR: DataCopyPad 硬件加速分段搬运
            AscendC::DataCopyExtParams copyParams(
                static_cast<uint16_t>(segCount),          // blockCount
                static_cast<uint32_t>(segSize),           // blockLen (每段元素数)
                static_cast<int64_t>(srcStride),          // srcStride (GM段间stride)
                static_cast<int64_t>(dstStride),          // dstStride (UB段间stride)
                0);                                       // rsv

            // DataCopyPad params: 不启用填充 (填充由后续 Padding 步骤处理)
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = false;

            AscendC::DataCopyPad(ubTensor_, gmTensor[curGmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOut(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        AscendC::DataCopy(gmTensor[curGmOffset], ubTensor_, static_cast<uint32_t>(copyLen));
    }

    __aicore__ inline __gm__ uint8_t* GmAddr()
    {
        return gmAddr_;
    }

private:
    AscendC::LocalTensor<T> ubTensor_;
    __gm__ uint8_t* gmAddr_;
#endif
};
} // namespace Atvoss::Reduce
#endif
