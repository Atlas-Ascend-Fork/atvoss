/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_TILE_TRANSCENDENTAL_H
#define Atvoss_DEV_TILE_TRANSCENDENTAL_H

#include "tile_evaluator_common.h"

namespace Atvoss::Tile::Eval {

/*!
 * \brief Broadcast calculation based on the expression
 * \param[in] args, LocalTensor data of input tensors & output tensor
 * \param[in] localVar, LocalTensor data of temp tensors
 * \param[in] tail, Calculate the length of the data
 * \return Broadcast Expression calculation
 */
template<typename T, typename U, Atvoss::Patterns::Pattern pattern>
struct Evaluator<OpAssign<T, OpBroadcast<pattern, U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpBroadcast<pattern, U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Binary>(args);
        return BroadcastAssign<OperationShape, pattern, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief ReduceSum calculation based on the expression
 * \param[in] args, LocalTensor data of input tensors & output tensor
 * \param[in] localVar, LocalTensor data of temp tensors
 * \param[in] tail, Calculate the length of the data
 * \return ReduceSum Expression calculation
 */
template<typename T, typename U, Atvoss::Patterns::Pattern pattern>
struct Evaluator<OpAssign<T, OpReduceSum<pattern, U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpReduceSum<pattern, U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Binary>(args);
        return ReduceSumAssign<OperationShape, pattern, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};
}
#endif //Atvoss_DEV_TILE_TRANSCENDENTAL_H
