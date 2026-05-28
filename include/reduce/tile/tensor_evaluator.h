/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_TILE_EVA_DATA_H
#define ATVOSS_REDUCE_TILE_EVA_DATA_H
#include "evaluator/eval_base.h"
#include "utils/layout/layout.h"
#include "operators/tile_shape.h"
#include "reduce/graph/buffer.h"

namespace Atvoss::Reduce::Tile {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;
using Atvoss::HasUsage;
using Atvoss::Tile::Operation;
using Atvoss::Tile::GetShape;
using Atvoss::Tile::CopyAssign;

enum class BufType : uint8_t {
    PARAM = 0b001,
    LOCAL_VAR = 0b010,
    REDUCE_VAR = 0b100,
};

template <typename BufferMap, std::size_t N, BufType bt, std::size_t start = 0>
__aicore__ uint32_t GetBufferId(uint32_t pingPong)
{
    if constexpr (start + 1 < Size_v<BufferMap>) {
        using Node = Get_t<BufferMap, start>;
        using Wrapper = Get_t<BufferMap, start + 1>;

        constexpr bool match =
            (Node::number == N) &&
            ((bt == BufType::PARAM && Wrapper::bufferUsage == Graph::BUF_PARAM) ||
             (bt == BufType::LOCAL_VAR && Wrapper::bufferUsage == Graph::BUF_LOCAL_VAR) ||
             (bt == BufType::REDUCE_VAR && Wrapper::bufferUsage == Graph::BUF_REDUCE_VAR));

        if constexpr (match) {
            if constexpr (Wrapper::bufferUsage == Graph::BUF_LOCAL_VAR) {
                return static_cast<uint32_t>(Wrapper::bufferId);
            } else {
                return static_cast<uint32_t>(pingPong == 0 ? Wrapper::bufferId : Wrapper::bufferId + 1);
            }
        } else {
            return GetBufferId<BufferMap, N, bt, start + 2>(pingPong);
        }
    } else {
        return 0;
    }
}

template <typename T>
using Dtype_t = typename T::Type::PrimType;

template <typename T>
struct Evaluator {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(const T& value, Context& /*context*/) const
    {
        return value;
    }
};

template <typename T>
struct Evaluator<Expression<T>> : Evaluator<T> {};

template <std::size_t N, typename T, typename L>
struct Evaluator<LocalVar<N, T, L>> {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(LocalVar<N, T, L> /*unused*/, Context& context) const
    {
        return AscendC::Std::get<N - 1>(context.tmpTensors);
    }
};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct Evaluator<Param<N, T, U, R>> {
    using Type = T;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(Param<N, T, U, R> /*unused*/, Context& context) const
    {
        constexpr auto index = N - 1;
        return AscendC::Std::get<index>(context.argsTensors);
    }
};

template <typename T, typename U>
struct Evaluator<OpAssign<T, U>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline void operator()(const OpAssign<T, U>& op, Context& context) const
    {
        return Atvoss::Tile::Assign(
            Evaluator<T>{}(op.GetLhs(), context),
            Evaluator<U>{}(op.GetRhs(), context));
    }
};

template <typename T, typename U>
struct Evaluator<OpAndThen<T, U>> {
    using Type = typename Evaluator<U>::Type;

    template <typename Context>
    __aicore__ inline Type operator()(const OpAndThen<T, U>& op, Context& context) const
    {
        AscendC::PipeBarrier<PIPE_V>();
        return Evaluator<T>{}(op.GetLhs(), context), Evaluator<U>{}(op.GetRhs(), context);
    }
};

template <typename T>
struct Evaluator<OpCopyIn<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(const OpCopyIn<T>& op, Context& context) const
    {
        auto& obj = Evaluator<T>{}(op.GetData(), context);
        uint32_t bufferId = GetBufferId<typename Context::BuffMaps, T::number, BufType::PARAM>(context.pingPong);
#if _ATVOSS_ARCH35_
        AscendC::Mutex::Lock<PIPE_MTE2>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        obj.CopyIn(context.gmOffset, context.elementNum,
                   context.segCount, context.segSize,
                   context.srcStride, context.dstStride);
#if _ATVOSS_ARCH35_
        AscendC::Mutex::Unlock<PIPE_MTE2>(bufferId);
        AscendC::Mutex::Lock<PIPE_V>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        return obj;
    }
};

template <typename T>
struct Evaluator<OpCopyOut<T>> {
    using Type = void;

    template <typename Context>
    __aicore__ inline decltype(auto) operator()(const OpCopyOut<T>& op, Context& context) const
    {
        uint32_t bufferId = GetBufferId<typename Context::BuffMaps, T::number, BufType::PARAM>(context.pingPong);
#if _ATVOSS_ARCH35_
        AscendC::Mutex::Unlock<PIPE_V>(bufferId);
        AscendC::Mutex::Lock<PIPE_MTE3>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        auto& obj = Evaluator<T>{}(op.GetData(), context);
        obj.CopyOut(context.gmOffset, context.elementNum);
#if _ATVOSS_ARCH35_
        AscendC::Mutex::Unlock<PIPE_MTE3>(bufferId);
#else
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        return obj;
    }
};
} // namespace Atvoss::Reduce::Tile
#endif // ATVOSS_REDUCE_TILE_EVA_DATA_H
