/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXP_INTF_TENSOR_H
#define ATVOSS_EXP_INTF_TENSOR_H

#include "expression/expr_template.h"

namespace Atvoss {

template <typename T>
struct OpAlloc : UnaryOp<T> {
    OpAlloc() = default;
    constexpr OpAlloc(T t) : UnaryOp<T>(t)
    {}
};

template <typename T>
constexpr auto Alloc(Expression<T> t)
{
    static_assert(IsParam_v<T>, "Only a Param supports CopyIn");
    static_assert(T::usage == ParamUsage::IN || T::usage == ParamUsage::IN_OUT, "Param usage must be in or in_out");
    return Expression<OpAlloc<T>>{{t.data}};
}

template <typename T>
constexpr auto Alloc(T&& lhs)
{
    return Expression<OpAlloc<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
struct OpFree : UnaryOp<T> {
    OpFree() = default;
    constexpr OpFree(T t) : UnaryOp<T>(t)
    {}
};

template <typename T>
constexpr auto Free(Expression<T> t)
{
    static_assert(IsParam_v<T>, "Only a Param supports CopyIn");
    static_assert(T::usage == ParamUsage::IN || T::usage == ParamUsage::IN_OUT, "Param usage must be in or in_out");
    return Expression<OpFree<T>>{{t.data}};
}

template <typename T>
constexpr auto Free(T&& lhs)
{
    return Expression<OpFree<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
struct OpCopyIn : UnaryOp<T> {
    OpCopyIn() = default;
    constexpr OpCopyIn(T t) : UnaryOp<T>(t)
    {}
};

template <typename T>
constexpr auto CopyIn(Expression<T> t)
{
    static_assert(IsParam_v<T>, "Only a Param supports CopyIn");
    static_assert(T::usage == ParamUsage::IN || T::usage == ParamUsage::IN_OUT, "Param usage must be in or in_out");
    return Expression<OpCopyIn<T>>{{t.data}};
}

template <typename T>
constexpr auto CopyIn(T&& lhs)
{
    return Expression<OpCopyIn<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
struct OpCopyOut : UnaryOp<T> {
    OpCopyOut() = default;
    constexpr OpCopyOut(T t) : UnaryOp<T>(t)
    {}
};

template <typename T>
constexpr auto CopyOut(Expression<T> t)
{
    static_assert(IsParam_v<T>, "Only a Param supports CopyOut");
    static_assert(T::usage == ParamUsage::OUT || T::usage == ParamUsage::IN_OUT, "Param usage must be out or in_out");
    return Expression<OpCopyOut<T>>{{t.data}};
}

template <typename T>
constexpr auto CopyOut(T&& lhs)
{
    return Expression<OpCopyOut<T>>{{std::forward<T>(lhs)}};
}

template <typename T>
struct OpCopy : UnaryOp<T> {
    OpCopy() = default;
    constexpr OpCopy(T t) : UnaryOp<T>(t)
    {}
};

} // namespace Atvoss

#endif