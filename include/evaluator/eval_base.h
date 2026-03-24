/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_EVALUATOR_BASE_H
#define ATVOSS_TILE_EVALUATOR_BASE_H
#include "utils/layout/layout.h"
#include "common/type_def.h"
#include "expression/expr_template.h"

namespace Atvoss::Tile {

template <typename T>
using Dtype_t = typename T::Type::PrimType;

// Primary template
template <typename T>
struct Evaluator {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(const T& value, Context& /*context*/) const
    {
        return value;
    }
};

// Treat Evaluator<Expression<T>> as Evaluator<T>
template <typename T>
struct Evaluator<Expression<T>> : Evaluator<T> {};

// Partial specialization for LocalVar
template <std::size_t N, typename T, typename L>
struct Evaluator<LocalVar<N, T, L>> {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(LocalVar<N, T, L> /*unused*/, Context& context) const
    {
        static_assert(N > 0, "[ERROR]: [Atvoss][Tile] LocalVar number starts from 1");
        return AscendC::Std::get<N - 1>(context.tmpTensors);
    }
};

// Partial specialization for Param
template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct Evaluator<Param<N, T, U, R>> {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(Param<N, T, U, R> /*unused*/, Context& context) const
    {
        static_assert(N > 0, "[ERROR]: [Atvoss][Tile] Param number starts from 1");
        constexpr auto index = N - 1;
        using ArgsTensors = decltype(context.argsTensors);
        using NthType = typename AscendC::Std::tuple_element<index, std::remove_reference_t<ArgsTensors>>::type;
        if constexpr (std::is_same_v<T, NthType> || std::is_same_v<T&, NthType> || std::is_same_v<T&&, NthType>) {
            return AscendC::Std::get<index>(context.argsTensors);
        } else {
            static_assert(
                U == ParamUsage::IN, "[ERROR]: [Atvoss][Tile] Only in-parameters allow implicit type conversions");
            return static_cast<T>(AscendC::Std::get<index>(context.argsTensors));
        }
    }
};

template <typename T, typename U>
struct Evaluator<OpAssign<T, U>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline void operator()(const OpAssign<T, U>& op, Context& context) const
    {
        return Assign(Evaluator<T>{}(op.GetLhs(), context), Evaluator<U>{}(op.GetRhs(), context));
    }
};

// Partial specializtion for E_x, E_y
template <typename T, typename U>
struct Evaluator<OpAndThen<T, U>> {
    using Type = typename Evaluator<U>::Type;

    template <typename Context>
    __aicore__ inline Type operator()(const OpAndThen<T, U>& op, Context& context) const
    {
        // operator, evaluates sequentially
        AscendC::PipeBarrier<PIPE_V>();
        return Atvoss::Tile::Evaluator<T>{}(op.GetLhs(), context), Atvoss::Tile::Evaluator<U>{}(op.GetRhs(), context);
    }
};

template <typename Fun, class... Args>
constexpr __aicore__ inline void Assign(Fun& fun_, Args... args)
{
    fun_(args...);
}

template <typename T, typename U>
__aicore__ inline void Assign(T& dst, const U& src)
{
    dst = src;
}

} // namespace Atvoss::Tile
#endif // ATVOSS_TILE_EVALUATOR_BASE_H
