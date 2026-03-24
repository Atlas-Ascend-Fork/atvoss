/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_ASCENDC_MATH_H
#define ATVOSS_TILE_ASCENDC_MATH_H
#include "lib/math/power.h"
#include "common/arch.h"
#include "tile_shape.h"

namespace Atvoss::Tile {

/*!
 * \brief dst[i] = src0[i] + src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void AddAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0, const AscendC::LocalTensor<T>& src1,
    OperationShape& operationShape)
{
    AscendC::Add(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src[i] + scalar
 * \param[in] src, Input LocalTensor
 * \param[in] scalar, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void AddsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, T scalar, OperationShape& operationShape)
{
    AscendC::Adds(dst, src, scalar, operationShape.axis0);
}

/*!
 * \brief dst[i] = src0[i] - src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SubAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0, const AscendC::LocalTensor<T>& src1,
    OperationShape& operationShape)
{
    AscendC::Sub(dst, src0, src1, operationShape.axis0);
}

#if _ATVOSS_ARCH35_
/*!
 * \brief dst[i] = src[i] - scalar
 * \param[in] src, Input LocalTensor
 * \param[in] scalar, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SubsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, T scalar, OperationShape& operationShape)
{
    AscendC::Subs(dst, src, scalar, operationShape.axis0);
}

/*!
 * \brief dst[i] = scalar - src[i]
 * \param[in] src, Input LocalTensor
 * \param[in] scalar, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SubsAssign(
    AscendC::LocalTensor<T>& dst, T scalar, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Subs(dst, scalar, src, operationShape.axis0);
}
#endif

/*!
 * \brief dst[i] = src0[i] * src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void MulAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0, const AscendC::LocalTensor<T>& src1,
    OperationShape& operationShape)
{
    AscendC::Mul(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src[i] * src1
 * \param[in] src, Input LocalTensor
 * \param[in] src1, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void MulsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, T src1, OperationShape& operationShape)
{
    AscendC::Muls(dst, src, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src0[i] / src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void DivAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0, const AscendC::LocalTensor<T>& src1,
    OperationShape& operationShape)
{
    AscendC::Div(dst, src0, src1, operationShape.axis0);
}

#if _ATVOSS_ARCH35_
/*!
 * \brief dst[i] = src[i] / scalar
 * \param[in] src, Input LocalTensor
 * \param[in] scalar, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void DivsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, T scalar, OperationShape& operationShape)
{
    AscendC::Divs(dst, src, scalar, operationShape.axis0);
}

/*!
 * \brief dst[i] = scalar / src[i]
 * \param[in] src, Input LocalTensor
 * \param[in] scalar, Input scalar
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void DivsAssign(
    AscendC::LocalTensor<T>& dst, T scalar, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Divs(dst, scalar, src, operationShape.axis0);
}
#endif

/*!
 * \brief dst[i] = src[i] / scalarValue
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void DivsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    T src1 = T{1} / scalarValue;
    AscendC::Muls(dst, src, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = exp(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void ExpAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Exp(dst, src, operationShape.axis0);
}

/*!
 * \brief dst[i] = abs(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void AbsAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Abs(dst, src, operationShape.axis0);
}

/*!
 * \brief dst[i] = sqrt(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SqrtAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Sqrt(dst, src, operationShape.axis0);
}

/*!
 * \brief dst[i] = power(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void PowerAssign(
    AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, OperationShape& operationShape)
{
    AscendC::Power(dst, src, T{scalarValue}, operationShape.axis0);
}

/*!
 * \brief dst[i] = cast(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, CastMode castMode, typename T1, typename T2>
__aicore__ inline void CastAssign(
    AscendC::LocalTensor<T1>& dst, const AscendC::LocalTensor<T2>& src, OperationShape& operationShape)
{
    if constexpr (castMode == CastMode::CAST_NONE) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_RINT) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_FLOOR) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_FLOOR, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_CEIL) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_CEIL, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_ROUND) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ROUND, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_TRUNC) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_TRUNC, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_ODD) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ODD, operationShape.axis0);
    }
}

using OperationShape = Atvoss::Layout::OperationShape;

/*!
 * \brief Add calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Add expression
 */
template <typename T, typename U>
struct Evaluator<OpAdd<T, U>> {
    using Type =
        decltype(Add(std::declval<typename Evaluator<T>::Type>(), std::declval<typename Evaluator<U>::Type>()));

    template <typename Context>
    __aicore__ inline auto operator()(const OpAdd<T, U>& op, Context& context) const
    {
        return Add(Evaluator<T>{}(op.GetLhs(), context), Evaluator<U>{}(op.GetRhs(), context));
    }
};

/*!
 * \brief Add calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Add expression
 */
template <typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpAdd<U, V>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpAdd<U, V>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        static_assert(
            !std::is_scalar_v<typename U::Type> || !std::is_scalar_v<typename V::Type>,
            "OpAdd's inputs not accepts all scalar types");
        if constexpr (std::is_scalar_v<typename U::Type>) {
            return Atvoss::Tile::AddsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context), operationShape);
        } else if constexpr (std::is_scalar_v<typename V::Type>) {
            return Atvoss::Tile::AddsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context), operationShape);
        } else {
            return Atvoss::Tile::AddAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        }
    }
};

/*!
 * \brief Sub calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sub expression
 */
template <typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpSub<U, V>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpSub<U, V>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
#if _ATVOSS_ARCH35_
        static_assert(
            !std::is_scalar_v<typename U::Type> || !std::is_scalar_v<typename V::Type>,
            "OpSub's inputs not accepts all scalar types");
        if constexpr (std::is_scalar_v<typename U::Type>) {
            return Atvoss::Tile::SubsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(), Evaluator<U>{}(op.GetRhs().GetLhs(), context),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        } else if constexpr (std::is_scalar_v<typename V::Type>) {
            return Atvoss::Tile::SubsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context), operationShape);
        } else {
            return Atvoss::Tile::SubAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        }
#else
        return Atvoss::Tile::SubAssign<OperationShape, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
            Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
#endif
    }
};

/*!
 * \brief Mul calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Mul expression
 */
template <typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpMul<U, V>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpMul<U, V>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        static_assert(
            !std::is_scalar_v<typename U::Type> || !std::is_scalar_v<typename V::Type>,
            "MulAssign's inputs not accepts all scalar types");
        if constexpr (std::is_scalar_v<typename U::Type>) {
            return Atvoss::Tile::MulsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context), operationShape);
        } else if constexpr (std::is_scalar_v<typename V::Type>) {
            return Atvoss::Tile::MulsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context), operationShape);
        } else {
            return Atvoss::Tile::MulAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        }
    }
};

/*!
 * \brief Div calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Div expression
 */
template <typename T, typename U, typename V>
struct Evaluator<OpAssign<T, OpDiv<U, V>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpDiv<U, V>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
#if _ATVOSS_ARCH35_
        static_assert(
            !std::is_scalar_v<typename U::Type> || !std::is_scalar_v<typename V::Type>,
            "OpDiv's inputs not accepts all scalar types");
        if constexpr (std::is_scalar_v<typename U::Type>) {
            return Atvoss::Tile::DivsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(), Evaluator<U>{}(op.GetRhs().GetLhs(), context),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        } else if constexpr (std::is_scalar_v<typename V::Type>) {
            return Atvoss::Tile::DivsAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context), operationShape);
        } else {
            return Atvoss::Tile::DivAssign<OperationShape, Dtype>(
                Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
                Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
                Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
        }
#else
        return Atvoss::Tile::DivAssign<OperationShape, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetLhs(), context).GetUbTensor(),
            Evaluator<V>{}(op.GetRhs().GetRhs(), context).GetUbTensor(), operationShape);
#endif
    }
};

/*!
 * \brief Divs calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Divs expression
 */
template <typename T, typename U, int scalarValue>
struct Evaluator<OpAssign<T, OpDivs<scalarValue, U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpDivs<scalarValue, U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::DivsAssign<OperationShape, scalarValue, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief Exp calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Exp expression
 */
template <typename T, typename U>
struct Evaluator<OpAssign<T, OpExp<U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpExp<U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::ExpAssign<OperationShape, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief Abs calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Abs expression
 */
template <typename T, typename U>
struct Evaluator<OpAssign<T, OpAbs<U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpAbs<U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::AbsAssign<OperationShape, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief Sqrt calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sqrt expression
 */
template <typename T, typename U>
struct Evaluator<OpAssign<T, OpSqrt<U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpSqrt<U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::SqrtAssign<OperationShape, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief Sqrt calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Sqrt expression
 */
template <typename T, typename U, int scalarValue>
struct Evaluator<OpAssign<T, OpPower<scalarValue, U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpPower<scalarValue, U>>& op, Context& context) const
    {
        using Dtype = Dtype_t<T>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::PowerAssign<OperationShape, scalarValue, Dtype>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

/*!
 * \brief Cast calculation based on the expression
 * \param[in] args, Input LocalTensor & output LocalTensor
 * \param[in] localVars, Temp LocalTensor
 * \param[in] tail, Length of calculation data
 * \return Cast expression
 */
template <typename T, typename U, typename R, CastMode castMode>
struct Evaluator<OpAssign<T, OpCast<castMode, R, U>>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline auto operator()(const OpAssign<T, OpCast<castMode, R, U>>& op, Context& context) const
    {
        using DstType = Dtype_t<T>;
        using SrcType = Dtype_t<U>;
        OperationShape operationShape = GetShape<Operation::Unary>(context.argsTensors);
        return Atvoss::Tile::CastAssign<OperationShape, castMode, DstType, SrcType>(
            Evaluator<T>{}(op.GetLhs(), context).GetUbTensor(),
            Evaluator<U>{}(op.GetRhs().GetData(), context).GetUbTensor(), operationShape);
    }
};

} // namespace Atvoss::Tile

#endif // ATVOSS_TILE_ASCENDC_MATH_H
