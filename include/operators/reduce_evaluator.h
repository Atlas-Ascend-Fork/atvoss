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
        return;
    }

    template<typename OperationShape, class Pattern, typename IsSameV<Pattern, Pattern::AR>::Type* dummpy = nullptr>
    __aicore__ inline void Compute(const LocalTensor<T>& dst, const LocalTensor<T>& src, OperationShape& shape) {
        return;
    }

    template<typename OperationShape, class Pattern>
    __aicore__ inline void UpdateCache(const LocalTensor<T>& cache, const LocalTensor<T>& src, OperationShape& shape,
                                       int64_t index) {
        return;
    }
    
    template<typename OperationShape>
    __aicore__ inline void BinaryAccPreProcess(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                               const LocalTensor<T>& src1, OperationShape& shape) {
        return;
    }

    template<typename OperationShape, class Pattern, class P>
    __aicore__ inline void DoPadding(const LocalTensor<T>& dst, OperationShape& shape, P& padding) {
        return;
    }

private:
    __aicore__ inline T GetPaddingValue() {
        return;
    }
};
} // namespace Atvoss::Tile

#endif // ATVOSS_TILE_ASCENDC_REDUCE_H
