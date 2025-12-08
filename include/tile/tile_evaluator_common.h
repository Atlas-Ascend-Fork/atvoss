/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_EVAL_COMMON_H
#define TILE_EVAL_COMMON_H
#include "utils/expression/expression.h"
#include "utils/operation.h"
#include "utils/layout/layout.h"
#include "tile_ascendc_math.h"
#include "tile_ascendc_transcendental.h"

namespace Atvoss::Tile::Eval {

using Util::TMP::FindUnique_t;
using Util::TMP::Size_v;


template <typename T>
struct Evaluator {
    using Type = T;

    template <typename ArgTup, typename LocalVarTup>
    __aicore__ decltype(auto) operator()(const T& value, ArgTup& args, LocalVarTup& localVars) const
    {
        if constexpr (std::is_invocable_v<T, ArgTup&, LocalVarTup&>) {
            return value(args, localVars);
        } else {
            return value;
        }
    }
};

template <std::size_t N, typename T>
constexpr auto DefineAutoLocalVar(const Expression<T>& /*EXPRESSION*/)
{
    using ResultType = typename Evaluator<T>::Type;
    return Expression<LocalVar<N, ResultType>>{};
}

namespace Detail {

template <typename Params, std::size_t... Ints>
constexpr auto GetParamTupleImpl(std::index_sequence<Ints...>)
{
    return std::make_tuple(typename FindUnique_t<CheckVarNum<Ints + 1>::template Checker, Params>::Type{}...);
}

template <typename LocalVars, std::size_t... Ints>
constexpr auto GetLocalVarTupleImpl(std::index_sequence<Ints...>)
{
    return std::make_tuple(typename FindUnique_t<CheckVarNum<Ints + 1>::template Checker, LocalVars>::Type{}...);
}

template <typename T>
void MakeAlikeDatum(std::vector<T>& dst, const std::vector<T>& src)
{
    dst.resize(src.size());
}

template <typename L, typename T, typename ArgTup>
void MakeAlikeOneLocalVar(T& localVar, const ArgTup& args)
{
    static_assert(IsLocalVar_v<L>, "[ERROR]: [Atvoss][Tile] A LocalVar is needed");
    if constexpr (!std::is_same_v<typename L::Like, void>) {
        MakeAlikeDatum(localVar, AscendC::Std::get<L::Like::number>(args));
    }
}

template <typename LocalVars, typename LocalVarTup, typename ArgTup, std::size_t... Ints>
void MakeAlikeLocalVarsImpl(LocalVarTup& localVars, const ArgTup& args, std::index_sequence<Ints...>)
{
    (MakeAlikeOneLocalVar<Util::TMP::Get_t<LocalVars, Ints>>(AscendC::Std::get<Ints>(localVars), args), ...);
}

}  // namespace Detail

template <typename Params>
constexpr auto GetParamTuple()
{
    constexpr auto paramCount = Size_v<Params>;
    return Detail::GetParamTupleImpl<Params>(std::make_index_sequence<paramCount>{});
};

template <typename LocalVars>
constexpr auto GetLocalVarTuple()
{
    constexpr auto localVarCount = Size_v<LocalVars>;
    return Detail::GetLocalVarTupleImpl<LocalVars>(std::make_index_sequence<localVarCount>{});
};

template <typename LocalVars, typename LocalVarTup, typename ArgTup>
void MakeAlikeLocalVars(LocalVarTup& localVars, const ArgTup& args)
{
    Detail::MakeAlikeLocalVarsImpl<LocalVars>(localVars, args,
                                              std::make_index_sequence<std::tuple_size_v<LocalVarTup>>{});
}

template <typename T, typename... Args>
auto Evaluate(const Expression<T>& expr, Args&&... args)
{
    // Declare local variables so that we can always use xTup& in
    // Evaluator<>::operator()
    auto argTuple = std::forward_as_tuple(std::forward<Args>(args)...);
    using LocalVars = LocalVars_t<T>;
    [[maybe_unused]] Params_t<T> dummy;  // Force checking the params
    auto localVars = GetLocalVarTuple<LocalVars>();
    MakeAlikeLocalVars<LocalVars>(localVars, argTuple);
    return Evaluator<T>{}(expr.data, argTuple, localVars);
}

// Treat Evaluator<Expression<T>> as Evaluator<T>
template <typename T>
struct Evaluator<Expression<T>> : Evaluator<T> {
};

// Partial specialization for LocalVar
template <std::size_t N, typename T, typename L>
struct Evaluator<LocalVar<N, T, L>> {
    using Type = T&;

    template <typename ArgTup, typename LocalVarTup>
    __aicore__ decltype(auto) operator()(LocalVar<N, T, L> /*unused*/, ArgTup& /*args*/, LocalVarTup& localVars) const
    {
        static_assert(N > 0, "[ERROR]: [Atvoss][Tile] LocalVar number starts from 1");
        return AscendC::Std::get<N - 1>(localVars);
    }
};

// Partial specialization for Param
template <std::size_t N, typename T, ParamUsage U>
struct Evaluator<Param<N, T, U>> {
    using Type = T;

    template <typename ArgTup, typename LocalVarTup>
    __aicore__ decltype(auto) operator()(Param<N, T, U> /*unused*/, ArgTup& args, LocalVarTup& /*localVars*/) const
    {
        static_assert(N > 0, "[ERROR]: [Atvoss][Tile] Param number starts from 1");
        constexpr auto index = N - 1;
        using NthType = typename AscendC::Std::tuple_element<index, std::remove_reference_t<ArgTup>>::type;
        if constexpr (std::is_same_v<T, NthType> || std::is_same_v<T&, NthType> || std::is_same_v<T&&, NthType>) {
            return AscendC::Std::get<index>(args);
        } else {
            static_assert(U == ParamUsage::in,
                          "[ERROR]: [Atvoss][Tile] Only in-parameters allow implicit type conversions");
            return static_cast<T>(AscendC::Std::get<index>(args));
        }
    }
};

// Partial specializtion for E_y = E_x
template <typename T, typename U>
struct Evaluator<OpAssign<T, U>> {
    using Type = void;

    template <typename ArgTup, typename LocalVarTup>
    __aicore__ void operator()(const OpAssign<T, U>& op, ArgTup& args, LocalVarTup& localVars, int tail) const
    {
        return Assign(Evaluator<T>{}(op.GetLhs(), args, localVars), Evaluator<U>{}(op.GetRhs(), args, localVars));
    }
};

// Partial specializtion for E_x, E_y
template <typename T, typename U>
struct Evaluator<OpAndThen<T, U>> {
    using Type = void;

    template <typename ArgTup, typename LocalVarTup, typename... Arguments>
    __aicore__ auto operator()(const OpAndThen<T, U>& op, ArgTup& args, LocalVarTup& localVars,
                               Arguments&... arguments) const
    {
        // operator, evaluates sequentially
        AscendC::PipeBarrier<PIPE_V>();
        return Evaluator<T>{}(op.GetLhs(), args, localVars, arguments...),
               Evaluator<U>{}(op.GetRhs(), args, localVars, arguments...);
    }
};

template <size_t N, typename T1>
static constexpr __aicore__ inline int32_t GetTotal(uint32_t eleCntInTensor = 1, int defaultSize = 1)
{
    constexpr size_t tupleSize = T1::size::value;
    if constexpr (tupleSize == 0) {
        return eleCntInTensor;
    } else if constexpr (N > tupleSize) {
        return eleCntInTensor;
    } else {
        using TValueType = typename T1::template get_type<N>;
        int32_t TValue = TValueType::value;
        defaultSize = defaultSize * TValue;
        if constexpr (N < tupleSize - 1) {
            return GetTotal<N + 1, T1>(defaultSize);
        }
        return defaultSize;
    }
}
template <typename Fun, class... Args>
constexpr __aicore__ inline void Assign(Fun& fun_, Args... args)
{
    fun_(args...);
}

template <typename T, typename U>
__aicore__ void Assign(T& dst, const U& src)
{
    dst = src;
}

template <typename T, typename = void>
struct LayoutImpl : std::false_type {
    using Layout = AscendC::Std::tuple<>;
};

template <typename T>
struct LayoutImpl<T, std::void_t<typename T::layout>> : std::true_type {
    using Layout = typename T::layout;
};

template <typename T>
using Layout_t = typename LayoutImpl<T>::Layout;

template <typename T, typename = void>
struct ShapeImpl {
    using type = typename T::TileShape;
};

template <typename T>
struct ShapeImpl<T, std::void_t<typename std::enable_if<(AscendC::Std::tuple_size<Layout_t<T>>::value > 0)>::type>> {
    using type = typename T::TileShape;
};

template <typename T>
using Shape_t = typename ShapeImpl<T>::type;

template <typename T>
using Stride_t = AscendC::Std::conditional_t<
    (AscendC::Std::tuple_size<Layout_t<T>>::value > 1),
    typename AscendC::Std::tuple_element<AscendC::Std::tuple_size<Layout_t<T>>::value - 1, Layout_t<T>>::type,
    AscendC::Stride<>>;

template <typename ShapeType>
struct ShapeSize {
    static constexpr size_t value = ShapeType::size::value;
};

template <typename... ShapeType>
struct ShapeSize<AscendC::Shape<ShapeType...>> {
    static constexpr size_t value = sizeof...(ShapeType);
};

template <typename T, Operation op = Operation::Unary, typename... Arguments>
static constexpr __aicore__ auto getShape(Arguments&... args)
{
    using OperationShape = Atvoss::Layout::OperationShape;
    if constexpr (ShapeSize<Shape_t<T>>::value == 0) {  // Tile size not specified by the user.
        static_assert(sizeof...(args) > 0, "[ERROR]: [Atvoss][Tile] Arguments pack must not be empty!");
        auto argsTuple = AscendC::Std::forward_as_tuple(args...);
        uint32_t totalCnt = AscendC::Std::get<0>(argsTuple);
        uint32_t shapeN = AscendC::Std::get<1>(argsTuple);
        if constexpr (op == Operation::Unary) {
            return OperationShape{totalCnt};
        } else if constexpr (op == Operation::Binary) {
            return OperationShape{totalCnt / shapeN, shapeN};
        }
    } else {  // User-specified tile size must be validated to ensure it is smaller than the UB space occupied by a single element.
        using DstShape = Shape_t<T>;
        using DstShape0Type = typename AscendC::Std::tuple_element<0, DstShape>::type;
        using DstShape1Type = typename AscendC::Std::tuple_element<1, DstShape>::type;
        static_assert((DstShape0Type::value > 0 && DstShape1Type::value > 0),
                      "[ERROR]: [Atvoss][Tile] Shape dim must not be zero");
        if constexpr (op == Operation::Unary) {
            return OperationShape{GetTotal<0, DstShape>()};
        } else if constexpr (op == Operation::Binary) {
            static_assert(AscendC::Std::tuple_size<DstShape>::value == 2, "[ERROR]: [Atvoss][Tile] DstShape must is 2!");
            return OperationShape{DstShape0Type::value, DstShape1Type::value};
        }
    }
}

template <Operation op = Operation::Unary, typename ArgTup>
static constexpr __aicore__ auto GetShape(ArgTup& args)
{
    auto blockTensor = AscendC::Std::get<0>(args);
    using LayoutType = typename decltype(blockTensor)::LayoutType;
    if constexpr (std::is_same_v<LayoutType, Atvoss::Layout::VariableRankExtents<1>>) {
        if constexpr(op == Operation::Unary){
            return blockTensor.GetLayout().GetUnaryShape();
        }
        if constexpr(op == Operation::Binary){
            return blockTensor.GetLayout().GetBinaryShape();
        }
    } else {
        if constexpr(op == Operation::Unary){
            return LayoutType::ShapeType::UNARY_SHAPE;
        }
        if constexpr(op == Operation::Binary){
            return LayoutType::ShapeType::BINARY_SHAPE;
        }
    }
}

template <typename T>
using Dtype_t = typename T::Type::PrimType;

}  // namespace Atvoss::Tile::Eval

#endif