/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_TILE_MATH_H
#define Atvoss_DEV_TILE_MATH_H

#include "tile_evaluator_common.h"
#include "utils/layout/layout.h"

namespace Atvoss::Tile::Eval {

using OperationShape = Atvoss::Layout::OperationShape;

/*!
 * \brief Add calculation based on the expression
 * \param[in] src1, Input data
 * \param[in] src2, Output data
 * \return src1 + src2
 */
template<typename T, typename U>
__aicore__ auto Add(const T &src1, const U &src2) {
    return src1 + src2;
}

/*!
 * \brief Add calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Add expression
*/
template<typename T, typename U>
struct Evaluator<OpAdd<T, U>> {
    using Type = decltype(Add(std::declval<typename Evaluator<T>::Type>(),
                              std::declval<typename Evaluator<U>::Type>()));

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAdd<T, U> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               int tail = 0,
                               Arguments&... arguments) const {
        return Add(Evaluator<T>{}(op.GetLhs(), args, localVars),
                   Evaluator<U>{}(op.GetRhs(), args, localVars));
    }
};

/*!
 * \brief Add calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Add expression
*/
template<typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpAdd<U, V>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpAdd<U, V>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return AddAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Sub calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sub expression
*/
template<typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpSub<U, V>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpSub<U, V>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return SubAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Mul calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Mul expression
 */
template<typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpMul<U, V>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpMul<U, V>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return MulAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Div calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Div expression
 */
template<typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpDiv<U, V>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpDiv<U, V>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return DivAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Divs calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Divs expression
 */
template<typename T, typename U, int scalarValue>
struct Evaluator<OpAssign<T, OpDivs<scalarValue, U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpDivs<scalarValue, U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return DivsAssign<OperationShape, scalarValue, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Exp calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Exp expression
 */
template<typename T, typename U>
struct Evaluator<OpAssign<T, OpExp<U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpExp<U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return ExpAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Sqrt calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sqrt expression
 */
template<typename T, typename U>
struct Evaluator<OpAssign<T, OpSqrt<U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto
    operator()(const OpAssign<T, OpSqrt<U>> &op, ArgTup &args, LocalVarTup &localVars, Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return SqrtAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Sqrt calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sqrt expression
 */
template<typename T, typename U, int scalarValue>
struct Evaluator<OpAssign<T, OpPower<scalarValue, U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpPower<scalarValue, U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return PowerAssign<OperationShape, scalarValue, Dtype>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

/*!
 * \brief Cast calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Cast expression
 */
template<typename T, typename U, CastMode castMode>
struct Evaluator<OpAssign<T, OpCast<castMode, U>>> {
    using Type = void;

    template<typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAssign<T, OpCast<castMode, U>> &op,
                               ArgTup &args,
                               LocalVarTup &localVars,
                               Arguments&... arguments) const {
        using DstType = Dtype_t<T>;
        using SrcType = Dtype_t<U>;
        OperationShape operationShape = GetShape<Operation::Unary>(args);
        return CastAssign<OperationShape, castMode, DstType, SrcType>(
                Evaluator<T>{}(op.GetLhs(), args, localVars).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetData(), args, localVars).GetUbTensor(),
                operationShape);
    }
};

}
#endif //Atvoss_DEV_TILE_MATH_H
