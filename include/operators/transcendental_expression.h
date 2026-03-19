/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXP_INTF_TRANSCENDENTAL_H
#define ATVOSS_EXP_INTF_TRANSCENDENTAL_H

#include "expression/expr_template.h"

namespace Atvoss {
template <Pattern pattern, typename T>
struct OpReduceSum : UnaryOp<T> {
    OpReduceSum() = default;

    constexpr OpReduceSum(T t) : UnaryOp<T>(t)
    {}
};

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceSum(Expression<T> lhs)
{
    return Expression<OpReduceSum<pattern, T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto ReduceSum(T&& lhs)
{
    return Expression<OpReduceSum<pattern, T>>{{std::forward<T>(lhs)}};
}

template <Pattern pattern, typename T>
struct OpBroadcast : UnaryOp<T> {
    OpBroadcast() = default;

    constexpr OpBroadcast(T t) : UnaryOp<T>(t)
    {}
};

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto Broadcast(Expression<T> lhs)
{
    return Expression<OpBroadcast<pattern, T>>{{lhs.data}};
}

template <Pattern pattern, typename T>
__host_aicore__ constexpr auto Broadcast(T&& lhs)
{
    return Expression<OpBroadcast<pattern, T>>{{std::forward<T>(lhs)}};
}
} // namespace Atvoss
#endif // ATVOSS_DEV_TRANSCENDENTAL_H
