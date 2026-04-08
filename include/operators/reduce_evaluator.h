/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_ASCENDC_REDUCE_H
#define ATVOSS_TILE_ASCENDC_REDUCE_H

#include "adv_api/reduce/reduce.h"

namespace Atvoss::Tile {

constexpr int32_t REDUCE_OP_SUM = 1;
constexpr int32_t REDUCE_OP_PROD = 2;
constexpr int32_t REDUCE_OP_MIN = 3;
constexpr int32_t REDUCE_OP_MAX = 4;

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

template <class Pattern, typename T, typename OperationShape, int32_t Operator>
__aicore__ inline void DoCaching(const LocalTensor<T>& ubDst, const LocalTensor<T>& ubSrc, int64_t index,
                                 OperationShape& shape)
{
    constexpr int64_t vLElems = Ops::Base::GetVRegSize() / sizeof(T);
    const int64_t cacheID = GetCacheID(index);
    const int64_t dimA = Pattern::TailA ? shape.axis1 : shape.axis0;
    const int64_t stride = Ops::Base::CeilAlign(dimA, vLElems);
    // count A轴的大小 * vLElems
    const uint16_t outerLoopTimes = Ops::Base::CeilDiv(static_cast<int64_t>(dimA), static_cast<int64_t>(vLElems));
    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);
    const uint32_t innerLoopStride = static_cast<uint32_t>(stride);  // cacahe的每一个idex的块的大小， A轴的大小
    LocalTensor<T> dstTensor = ubDst;
    LocalTensor<T> srcTensor = ubSrc;

    __VEC_SCOPE__
    {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* cah = (__local_mem__ T*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(dimA);
        AscendC::MicroAPI::RegTensor<T> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {  // outerLoopTimes是dimA的大小
            pMask = AscendC::MicroAPI::UpdateMask<T>(sreg);
            AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ T*)src + i * vLElems);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::MicroAPI::DataCopy(bReg, (__local_mem__ T*)dst + i * vLElems + j * innerLoopStride);
                if constexpr (Operator == REDUCE_OP_SUM) {
                    AscendC::MicroAPI::Add<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_PROD) {
                    AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_MIN) {
                    AscendC::MicroAPI::Min<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                } else if constexpr (Operator == REDUCE_OP_MAX) {
                    AscendC::MicroAPI::Max<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
            }
            AscendC::MicroAPI::DataCopy((__local_mem__ T*)cah + i * vLElems, aReg, pMask);
        }
    }
}

/*!
 * \brief Sum all input elements
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, Atvoss::Pattern pattern, typename T>
__aicore__ inline void ReduceSumAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    uint32_t shape[] = {operationShape.axis0, operationShape.axis1};
    if constexpr (pattern == Atvoss::Pattern::AR) {
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(dst, src, shape, true);
    } else if constexpr (pattern == Atvoss::Pattern::RA) {
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(dst, src, shape, true);
    }
}

/*!
 * \brief ReduceSum calculation based on the expression
 * \param[in] args, LocalTensor data of input tensors & output tensor
 * \param[in] localVar, LocalTensor data of temp tensors
 * \param[in] tail, Calculate the length of the data
 * \return ReduceSum Expression calculation
 */
template <typename T, typename U, Atvoss::Pattern pattern>
struct Evaluator<OpAssign<T, OpReduceSum<pattern, U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpReduceSum<pattern, U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Binary>(context.argsTensors);
        return Atvoss::Tile::ReduceSumAssign<OperationShape, pattern, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief ReduceSum calculation based on the expression
 * \param[in] args, LocalTensor data of input tensors & output tensor
 * \param[in] localVar, LocalTensor data of temp tensors
 * \param[in] tail, Calculate the length of the data
 * \return ReduceSum Expression calculation
 */
template <typename T, typename U>
struct Evaluator<OpAssign<T, OpReduceSum<U>>> {
    using Type = void;

public:
    template<typename OperationShape, class Pattern, typename IsSameV<Pattern, Pattern::RA>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(const LocalTensor<T>& dst, const LocalTensor<T>& src, OperationShape& shape) {
        uint32_t srcShape[2] = {shape.axis0, shape.axis1};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(dst, src, srcShape, false);
    }

    template<typename OperationShape, class Pattern, typename IsSameV<Pattern, Pattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(const LocalTensor<T>& dst, const LocalTensor<T>& src, OperationShape& shape) {
        uint32_t srcShape[2] = {shape.axis0, shape.axis1};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, false);
    }

    template<typename OperationShape, class Pattern>
    __aicore__ inline void UpdateCache(const LocalTensor<T>& cache, const LocalTensor<T>& src, OperationShape& shape,
                                       int64_t index) {
        DoCaching<Pattern, T, OperationShape, REDUCE_OP_SUM>(cache, src, index, shape);
    }
    
    template<typename OperationShape>
    __aicore__ inline void BinaryAccPreProcess(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                               const LocalTensor<T>& src1, OperationShape& shape) {
        int32_t calCount = static_cast<int32_t>(shape.axis0 * shape.axis1);
        AscendC::Add(dst, src0, src1, calCount);
    }

    template<typename OperationShape, class Pattern, class P>
    __aicore__ inline void DoPadding(const LocalTensor<T>& dst, OperationShape& shape, P& padding) {
        return;
    }

private:
    __aicore__ inline T GetPaddingValue() {
        T paddingValue = 0;
        return paddingValue;
    }
};
} // namespace Atvoss::Tile

#endif // ATVOSS_TILE_ASCENDC_REDUCE_H
