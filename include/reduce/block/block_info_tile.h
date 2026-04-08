/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file block_info_tile.h
 * @brief Reduce Block 层 Tensor 封装: UB Tensor + GM 地址绑定
 */

#ifndef ATVOSS_REDUCE_BLOCK_TENSOR_H
#define ATVOSS_REDUCE_BLOCK_TENSOR_H

#include "utils/layout/layout.h"

namespace Atvoss::Reduce {

/**
 * @brief Block 层 Tensor 封装, 同时持有 GM 地址和 UB LocalTensor
 * @tparam T 数据类型 (如 float, half)
 * @tparam L Layout 类型, 编码 tile 的形状信息
 */
template <typename T, typename L = Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<1, 1, 1>>>
class BlockTensor {
public:
    using PrimType = T;
    using LayoutType = L;

#if !defined(__ATVOSS_HOST_ONLY__)
    __aicore__ inline BlockTensor() = default;

    /**
     * @brief 构造 BlockTensor 并绑定 GM 地址
     * @param gm GM 数据指针
     */
    __aicore__ inline BlockTensor(__gm__ uint8_t* gm)
    {
        gmAddr_ = gm;
    }

    /**
     * @brief 获取 UB LocalTensor 引用
     * @return LocalTensor<T>&
     */
    __aicore__ inline AscendC::LocalTensor<T>& GetUbTensor()
    {
        return ubTensor_;
    }

    /**
     * @brief 普通连续 CopyIn: GM → UB 整块搬运
     * @param curGmOffset GM 偏移 (元素偏移)
     * @param copyLen 拷贝元素数
     */
    __aicore__ inline void CopyIn(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        AscendC::DataCopy(ubTensor_, gmTensor[curGmOffset], static_cast<uint32_t>(copyLen));
    }

    /**
     * @brief ARAR→AARR 转置 CopyIn (DataCopyPad 硬件加速)
     *
     * GM(ARAR交织布局):
     *   [A₀R₀..R_{k-1} | 间隔srcStride | A₁R₀..R_{k-1} | ...]
     *   segCount 个 A-分段, 每段 segSize 个 R 元素, 段间 stride = srcStride
     *
     * UB(AARR分块布局):
     *   [A₀R₀..R_{k-1} | A₁R₀..R_{k-1} | ...]  连续存放, dstStride = segSize
     *
     * 参数来源 (通过 ReduceSchLoopInfo 确定 split axis):
     *   segCount  = ubFactorA  (A维UB切分数)
     *   segSize   = ubFactorR  (R维UB切分数)
     *   srcStride = cfg.srcStride[loopAAxis[loopACount-1]]  (A切分轴在GM中的stride)
     *   dstStride = segSize    (UB中AARR连续布局)
     *
     * @param curGmOffset GM 偏移
     * @param copyLen 拷贝元素数
     * @param segCount A维度分段数
     * @param segSize 每段R轴元素数
     * @param srcStride GM中相邻A段间的元素stride
     * @param dstStride UB中相邻A段间的stride
     */
    __aicore__ inline void CopyIn(uint64_t curGmOffset, uint64_t copyLen,
                                  uint64_t segCount, uint64_t segSize,
                                  uint64_t srcStride, uint64_t dstStride)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));

        if (segCount <= 1 || srcStride == 0 || srcStride == segSize) {
            AscendC::DataCopy(ubTensor_, gmTensor[curGmOffset], static_cast<uint32_t>(copyLen));
        } else {
            AscendC::DataCopyExtParams copyParams(
                static_cast<uint16_t>(segCount),          // blockCount
                static_cast<uint32_t>(segSize),           // blockLen
                static_cast<int64_t>(srcStride),          // srcStride
                static_cast<int64_t>(dstStride),          // dstStride
                0);                                       // rsv

            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = false;

            AscendC::DataCopyPad(ubTensor_, gmTensor[curGmOffset], copyParams, padParams);
        }
    }

    /**
     * @brief CopyOut: UB → GM 整块写回
     * @param curGmOffset GM 偏移
     * @param copyLen 拷贝元素数
     */
    __aicore__ inline void CopyOut(uint64_t curGmOffset, uint64_t copyLen)
    {
        AscendC::GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmAddr_));
        AscendC::DataCopy(gmTensor[curGmOffset], ubTensor_, static_cast<uint32_t>(copyLen));
    }

    /**
     * @brief 获取 GM 地址指针
     * @return __gm__ uint8_t*
     */
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
