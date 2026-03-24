/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_ASCENDC_TRANSCENDENTAL_H
#define ATVOSS_TILE_ASCENDC_TRANSCENDENTAL_H

#include "adv_api/reduce/reduce.h"
#include "adv_api/broadcast/broadcast.h"

namespace Atvoss::Tile {

/*!
 * \brief Broadcast all input elements
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, Pattern pattern, typename T>
__aicore__ inline void BroadcastAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    uint32_t dstShape[] = {operationShape.axis0, operationShape.axis1};
    if constexpr (pattern == Pattern::AB) {
        uint32_t srcShape_[] = {operationShape.axis0, 1};
        AscendC::Broadcast<T, 2, 1>(dst, src, dstShape, srcShape_);
    } else if constexpr (pattern == Pattern::BA) {
        uint32_t srcShape_[] = {1, operationShape.axis1};
        AscendC::Broadcast<T, 2, 0>(dst, src, dstShape, srcShape_);
    } else {
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
    } else {
    }
}

/*!
 * \brief Broadcast calculation based on the expression
 * \param[in] args, LocalTensor data of input tensors & output tensor
 * \param[in] localVar, LocalTensor data of temp tensors
 * \param[in] tail, Calculate the length of the data
 * \return Broadcast Expression calculation
 */
template <typename T, typename U, Atvoss::Pattern pattern>
struct Evaluator<OpAssign<T, OpBroadcast<pattern, U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpBroadcast<pattern, U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Binary>(context.argsTensors);
        return Atvoss::Tile::BroadcastAssign<OperationShape, pattern, Dtype>(
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
} // namespace Atvoss::Tile

#endif // ATVOSS_TILE_ASCENDC_MATH_H
