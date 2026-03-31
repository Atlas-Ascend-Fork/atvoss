/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_EXP_H
#define ATVOSS_REDUCE_EXP_H

#include "expression/expr_template.h"

namespace Atvoss {
template <typename T>
struct ReduceOp : UnaryOp<T> {
    ReduceOp() = default;

    constexpr ReduceOp(T t) : UnaryOp<T>(t)
    {}
};

template <typename T>
struct OpReduceSum : ReduceOp<T> {
    OpReduceSum() = default;

    constexpr OpReduceSum(T t) : ReduceOp<T>(t)
    {}
};

template <typename T>
struct OpReduceMax : ReduceOp<T> {
    OpReduceMax() = default;

    constexpr OpReduceMax(T t) : ReduceOp<T>(t)
    {}
};

template <typename T>
struct OpReduceMin : ReduceOp<T> {
    OpReduceMin() = default;

    constexpr OpReduceMin(T t) : ReduceOp<T>(t)
    {}
};

template <typename T>
struct OpReduceProd : ReduceOp<T> {
    OpReduceProd() = default;

    constexpr OpReduceProd(T t) : ReduceOp<T>(t)
    {}
};

template <typename T>
__host_aicore__ constexpr auto ReduceSum(Expression<T> lhs)
{
    return Expression<OpReduceSum<T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceSum(T&& lhs)
{
    return Expression<OpReduceSum<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
__host_aicore__ constexpr auto ReduceMax(Expression<T> lhs)
{
    return Expression<OpReduceMax<T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceMax(T&& lhs)
{
    return Expression<OpReduceMax<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
__host_aicore__ constexpr auto ReduceMin(Expression<T> lhs)
{
    return Expression<OpReduceMin<T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceMin(T&& lhs)
{
    return Expression<OpReduceMin<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
__host_aicore__ constexpr auto ReduceProd(Expression<T> lhs)
{
    return Expression<OpReduceProd<T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceProd(T&& lhs)
{
    return Expression<OpReduceProd<T>>{{std::forward<T>(lhs)}};
}

} // namespace Atvoss
#endif // ATVOSS_REDUCE_EXP_H
