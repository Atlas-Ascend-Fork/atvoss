/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_GRAPH_HELPER_H
#define ATVOSS_REDUCE_GRAPH_HELPER_H

#include <cstdint>
#include "graph/expr_operations.h"

namespace Atvoss::Reduce::Graph {

using Atvoss::Graph::ContainsNodeInExpr;
using Atvoss::Graph::ExtractInputs;
using Atvoss::Util::Any_v;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Contains_v;
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Map_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::Sort_t;
using Atvoss::Util::Unique_t;

template <typename ExprList, typename TargetNode, std::size_t start, std::size_t current = 0>
struct IsNodeUsedAfter;

template <typename TargetNode, std::size_t start, std::size_t current>
struct IsNodeUsedAfter<TypeList<>, TargetNode, start, current> {
    static constexpr bool value = false;
};

template <typename First, typename... Rest, typename TargetNode, std::size_t start, std::size_t current>
struct IsNodeUsedAfter<TypeList<First, Rest...>, TargetNode, start, current> {
private:
    static constexpr bool needCheck = (current >= start);
    static constexpr bool currentCheck =
        needCheck ? ContainsNodeInExpr<typename First::RhsType, TargetNode>::value : false;
    static constexpr bool rest = IsNodeUsedAfter<TypeList<Rest...>, TargetNode, start, current + 1>::value;

public:
    static constexpr bool value = currentCheck || rest;
};

template <typename Expr>
struct ExtractResultType {
    using Type = typename Expr::LhsType;
};

template <typename T, typename = void>
struct IsReduceOp : std::false_type {};

template <typename T>
struct IsReduceOp<T, std::enable_if_t<std::is_base_of_v<ReduceOp<typename T::DataType>, T>>> : std::true_type {};

template <typename T>
struct ContainsReduceOp : IsReduceOp<T> {};

template <template <typename> class Op, typename Inner>
struct ContainsReduceOp<Op<Inner>> : std::bool_constant<IsReduceOp<Op<Inner>>::value> {};

template <typename T, typename U>
struct ContainsReduceOp<OpAssign<T, U>> : ContainsReduceOp<U> {};

template <typename TargetNode>
struct ContainsNodeInRefList {
    template <typename Op>
    struct Type {
        static constexpr bool value = ContainsNodeInExpr<Op, TargetNode>::value;
    };
};

template <typename ReduceOpList, typename TargetLocalVar>
struct CheckReduceInput {
    static constexpr bool value = Any_v<ContainsNodeInRefList<TargetLocalVar>::template Type, ReduceOpList>;
};

template <typename ReduceOpList, typename CurrentOpAssign>
struct IsDirectConnectToReduce {
    using CurrentOutput = typename CurrentOpAssign::LhsType;
    static constexpr bool value = CheckReduceInput<ReduceOpList, CurrentOutput>::value;
};

template <typename List, typename TargetNode>
struct FilterRefLocalVar {
    using Type = Filter_t<ContainsNodeInRefList<TargetNode>::template Type, List>;
};

template <typename List, typename ExcludeList>
struct FilterNotInExclude {
    template <typename T>
    struct NotInExclude : std::bool_constant<!Contains_v<ExcludeList, T>> {};
    using Type = Filter_t<NotInExclude, List>;
};

// 找到不在CollectedExprs中，定义输入(父节点)的表达式
template <typename Inputs, typename FullList, typename CollectedExprs = TypeList<>, typename Result = TypeList<>>
struct FindUnhandledInputsExpr;

template <typename Inputs, typename CollectedExprs, typename... ResultTypes>
struct FindUnhandledInputsExpr<Inputs, TypeList<>, CollectedExprs, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
};

template <typename Inputs, typename FirstExpr, typename... RestExpr, typename CollectedExprs, typename... ResultTypes>
struct FindUnhandledInputsExpr<Inputs, TypeList<FirstExpr, RestExpr...>, CollectedExprs, TypeList<ResultTypes...>> {
private:
    using CurrentResults = TypeList<ResultTypes...>;
    using LHS = typename FirstExpr::LhsType;
    static constexpr bool IsLocalVarInput = Contains_v<Inputs, LHS>;
    static constexpr bool NotCollected = !Contains_v<CollectedExprs, FirstExpr>;
    using NeedAdd = std::conditional_t<IsLocalVarInput && NotCollected, std::true_type, std::false_type>;
    using NewResult =
        std::conditional_t<NeedAdd::value, Concatenate_t<CurrentResults, TypeList<FirstExpr>>, CurrentResults>;
    using NewCollected =
        std::conditional_t<NeedAdd::value, Concatenate_t<CollectedExprs, TypeList<FirstExpr>>, CollectedExprs>;

public:
    using Type = typename FindUnhandledInputsExpr<Inputs, TypeList<RestExpr...>, NewCollected, NewResult>::Type;
};

template <typename ExprList, typename FullList, typename ExcludeList, typename Result = TypeList<>>
struct FindAllUnhandledInputs;

template <typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindAllUnhandledInputs<TypeList<>, FullList, ExcludeList, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
};

template <typename First, typename... Rest, typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindAllUnhandledInputs<TypeList<First, Rest...>, FullList, ExcludeList, TypeList<ResultTypes...>> {
private:
    using CurrentResults = TypeList<ResultTypes...>;
    using Inputs = typename ExtractInputs<typename First::RhsType>::Type;
    using FilteredInputs = typename FilterNotInExclude<Inputs, ExcludeList>::Type;
    using NewExprs = typename FindUnhandledInputsExpr<FilteredInputs, FullList, CurrentResults>::Type;
    using CombinedResult = Concatenate_t<CurrentResults, NewExprs>;

public:
    using Type = typename FindAllUnhandledInputs<TypeList<Rest...>, FullList, ExcludeList, CombinedResult>::Type;
};

// 找到不在CollectedExprs中，引用输出(子节点)的表达式
template <typename VarList, typename FullList, typename CollectedExprs = TypeList<>, typename Result = TypeList<>>
struct FindOutputRefsExpr;

template <typename FullList, typename CollectedExprs, typename... ResultTypes>
struct FindOutputRefsExpr<TypeList<>, FullList, CollectedExprs, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
};

template <typename FirstVar, typename... RestVars, typename FullList, typename CollectedExprs, typename... ResultTypes>
struct FindOutputRefsExpr<TypeList<FirstVar, RestVars...>, FullList, CollectedExprs, TypeList<ResultTypes...>> {
private:
    using CurrentRefs = typename FilterRefLocalVar<FullList, FirstVar>::Type;
    using FilteredCurrentRefs = Difference_t<CurrentRefs, CollectedExprs>;
    using NewCollected = Concatenate_t<CollectedExprs, FilteredCurrentRefs>;
    using NewResult = Concatenate_t<TypeList<ResultTypes...>, FilteredCurrentRefs>;

public:
    using Type = typename FindOutputRefsExpr<TypeList<RestVars...>, FullList, NewCollected, NewResult>::Type;
};

template <typename Expr>
struct GetExprOutputNumber;

template <typename LHS, typename RHS>
struct GetExprOutputNumber<OpAssign<LHS, RHS>> {
    static constexpr std::size_t value = LHS::number;
};

template <typename Expr>
struct IsExprOutputParam : std::false_type {};

template <typename LHS, typename RHS>
struct IsExprOutputParam<OpAssign<LHS, RHS>> : Atvoss::IsParam<LHS> {};

// 表达式按其输出类型LocalVar、Param排序，LocalVar在前，Param在后，且LocalVar按number排序
template <typename ExprList>
struct SortExprsByOutput;

template <typename... Exprs>
struct SortExprsByOutput<TypeList<Exprs...>> {
    template <typename E1, typename E2>
    struct ExprOrderLess {
        static constexpr bool E1IsParam = IsExprOutputParam<E1>::value;
        static constexpr bool E2IsParam = IsExprOutputParam<E2>::value;
        static constexpr bool value =
            (!E1IsParam && E2IsParam) ||
            (!E1IsParam && !E2IsParam && GetExprOutputNumber<E1>::value < GetExprOutputNumber<E2>::value);
    };
    using Type = Sort_t<ExprOrderLess, TypeList<Exprs...>>;
};

// 收集多输入场景下，分支输入的输入输出(父子节点)表达式
template <typename OtherInputsList, typename FullList, typename ExcludeList, typename Result = TypeList<>>
struct CollectOtherInputRefs;

template <typename FullList, typename ExcludeList, typename... ResultTypes>
struct CollectOtherInputRefs<TypeList<>, FullList, ExcludeList, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
};

template <typename First, typename... Rest, typename FullList, typename ExcludeList, typename... ResultTypes>
struct CollectOtherInputRefs<TypeList<First, Rest...>, FullList, ExcludeList, TypeList<ResultTypes...>> {
private:
    using CurrentResults = TypeList<ResultTypes...>;
    using Inputs = typename ExtractInputs<typename First::RhsType>::Type;
    using FilteredInputs = typename FilterNotInExclude<Inputs, ExcludeList>::Type;
    using InputsDefExprs = typename FindUnhandledInputsExpr<FilteredInputs, FullList, CurrentResults>::Type;

    // 单输出多引用场景
    using Output = typename First::LhsType;
    using OutputRefExprs = typename FindOutputRefsExpr<TypeList<Output>, FullList, CurrentResults>::Type;
    using NodeInOutExprs = Concatenate_t<InputsDefExprs, OutputRefExprs>;
    // 按照输入 + 当前节点 + 输出的顺序排列
    using NextResult = Concatenate_t<InputsDefExprs, TypeList<First>, OutputRefExprs, CurrentResults>;
    using ResultList = Map_t<ExtractResultType, NodeInOutExprs>;
    using NestExclude = Concatenate_t<ExcludeList, ResultList>;
    using NestRest = Unique_t<Concatenate_t<NodeInOutExprs, TypeList<Rest...>>>;

public:
    using Type = typename CollectOtherInputRefs<NestRest, FullList, NestExclude, NextResult>::Type;
};

} // namespace Atvoss::Reduce::Graph

#endif