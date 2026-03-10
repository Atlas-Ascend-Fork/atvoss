/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXP_INTF_MATH_H
#define ATVOSS_EXP_INTF_MATH_H

#include "expression/expr_template.h"

namespace Atvoss {
// Disallow dangerous expressions like (Expression{2}, 3)
template <typename T, typename U>
__host_aicore__ constexpr auto operator,(Expression<T> t, U&& u) = delete;

template <typename T, typename U>
struct OpAdd : BinaryOp<T, U> {
    OpAdd() = default;

    constexpr OpAdd(T t, U u) : BinaryOp<T, U>(t, u)
    {}
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator+(Expression<T> lhs, Expression<U> rhs)
{
    return Expression<OpAdd<T, U>>{{lhs.data, rhs.data}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator+(Expression<T> lhs, U&& rhs)
{
    return Expression<OpAdd<T, U>>{{lhs.data, std::forward<U>(rhs)}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator+(T&& lhs, Expression<U> rhs)
{
    return Expression<OpAdd<T, U>>{{std::forward<T>(lhs), rhs.data}};
}

template <typename T, typename U>
struct OpSub : BinaryOp<T, U> {
    OpSub() = default;

    constexpr OpSub(T t, U u) : BinaryOp<T, U>(t, u)
    {}
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator-(Expression<T> lhs, Expression<U> rhs)
{
    return Expression<OpSub<T, U>>{{lhs.data, rhs.data}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator-(Expression<T> lhs, U&& rhs)
{
    return Expression<OpSub<T, U>>{{lhs.data, std::forward<U>(rhs)}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator-(T&& lhs, Expression<U> rhs)
{
    return Expression<OpSub<T, U>>{{std::forward<T>(lhs), rhs.data}};
}

template <typename T, typename U>
struct OpMul : BinaryOp<T, U> {
    OpMul() = default;

    constexpr OpMul(T t, U u) : BinaryOp<T, U>(t, u)
    {}
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator*(Expression<T> lhs, Expression<U> rhs)
{
    return Expression<OpMul<T, U>>{{lhs.data, rhs.data}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator*(Expression<T> lhs, U&& rhs)
{
    return Expression<OpMul<T, U>>{{lhs.data, std::forward<U>(rhs)}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator*(T&& lhs, Expression<U> rhs)
{
    return Expression<OpMul<T, U>>{{std::forward<T>(lhs), rhs.data}};
}

template <typename T, typename U>
struct OpDiv : BinaryOp<T, U> {
    OpDiv() = default;

    constexpr OpDiv(T t, U u) : BinaryOp<T, U>(t, u)
    {}
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator/(Expression<T> lhs, Expression<U> rhs)
{
    return Expression<OpDiv<T, U>>{{lhs.data, rhs.data}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator/(Expression<T> lhs, U&& rhs)
{
    return Expression<OpDiv<T, U>>{{lhs.data, std::forward<U>(rhs)}};
}

template <typename T, typename U>
__host_aicore__ constexpr auto operator/(T&& lhs, Expression<U> rhs)
{
    return Expression<OpDiv<T, U>>{{std::forward<T>(lhs), rhs.data}};
}

template <auto scalarValue, typename T>
struct OpPower : UnaryOp<T> {
    OpPower() = default;

    constexpr OpPower(T t) : UnaryOp<T>(t)
    {}
};

template <auto scalarValue, typename T>
__host_aicore__ constexpr auto Power(Expression<T> lhs)
{
    return Expression<OpPower<scalarValue, T>>{{lhs.data}};
}

template <auto scalarValue, typename T>
__host_aicore__ constexpr auto Power(T&& lhs)
{
    return Expression<OpPower<scalarValue, T>>{{std::forward<T>(lhs)}};
}

template <auto scalarValue, typename T>
struct OpDivs : UnaryOp<T> {
    OpDivs() = default;

    constexpr OpDivs(T t) : UnaryOp<T>(t)
    {}
};

template <auto scalarValue, typename T>
__host_aicore__ constexpr auto Divs(Expression<T> lhs)
{
    return Expression<OpDivs<scalarValue, T>>{{lhs.data}};
}

template <auto scalarValue, typename T>
__host_aicore__ constexpr auto Divs(T&& lhs)
{
    return Expression<OpDivs<scalarValue, T>>{{std::forward<T>(lhs)}};
}

DeclareUnaryOp(Sqrt);

DeclareUnaryOp(Exp);

DeclareUnaryOp(Abs);

template <CastMode castMode, typename R, typename T>
struct OpCast : UnaryOp<T, typename ReplaceTensorType<typename T::TensorType, R>::Type> {
    OpCast() = default;

    constexpr OpCast(T t) : UnaryOp<T, typename ReplaceTensorType<typename T::TensorType, R>::Type>(t)
    {}
};

template <typename T>
struct IsOpCast : std::false_type {};

template <CastMode Mode, typename R, typename T>
struct IsOpCast<OpCast<Mode, R, T>> : std::true_type {};

template <typename T>
inline constexpr bool IsOpCast_v = IsOpCast<T>::value;

template <CastMode castMode = CastMode::CAST_ROUND, typename R, typename T>
__host_aicore__ constexpr auto Cast(Expression<T> lhs)
{
    return Expression<OpCast<castMode, R, T>>{{lhs.data}};
}

template <CastMode castMode = CastMode::CAST_ROUND, typename R, typename T>
__host_aicore__ constexpr auto Cast(T&& lhs)
{
    return Expression<OpCast<castMode, R, T>>{{std::forward<T>(lhs)}};
}

DeclareBinaryOp(Max);

} // namespace Atvoss
#endif // ATVOSS_DEV_MATH_H
