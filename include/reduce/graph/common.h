/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_GRAPH_COMMON_H
#define ATVOSS_REDUCE_GRAPH_COMMON_H

#include <cstdint>

namespace Atvoss::Reduce::Graph {

using Atvoss::Util::Any_v;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Contains_v;
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Map_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::Unique_t;

template <typename T>
struct ExtractInputs {
    using Type = TypeList<>;
};

template <size_t N, typename T, ParamUsage U, size_t RN>
struct ExtractInputs<Param<N, T, U, RN>> {
    using Type = TypeList<Param<N, T, U, RN>>;
};

template <size_t N, typename T, typename L>
struct ExtractInputs<LocalVar<N, T, L>> {
    using Type = TypeList<LocalVar<N, T, L>>;
};

template <template <typename> class Op, typename Inner>
struct ExtractInputs<Op<Inner>> {
    using Type = typename ExtractInputs<Inner>::Type;
};

template <template <auto, typename> class Op, auto Pattern, typename Inner>
struct ExtractInputs<Op<Pattern, Inner>> {
    using Type = typename ExtractInputs<Inner>::Type;
};

template <template <auto, typename, typename> class Op, auto Pattern, typename R, typename Inner>
struct ExtractInputs<Op<Pattern, R, Inner>> {
    using Type = typename ExtractInputs<Inner>::Type;
};

template <template <typename, typename> class Op, typename T, typename U>
struct ExtractInputs<Op<T, U>> {
    using Type = Concatenate_t<typename ExtractInputs<T>::Type, typename ExtractInputs<U>::Type>;
};

template <template <typename, typename, typename> class Op, typename T, typename U, typename V>
struct ExtractInputs<Op<T, U, V>> {
    using Type = Concatenate_t<
        typename ExtractInputs<T>::Type,
        Concatenate_t<typename ExtractInputs<U>::Type, typename ExtractInputs<V>::Type>>;
};

template <typename Expr, typename TargetNode>
struct ContainsNodeInExpr {
    static constexpr bool value = false;
};

template <typename TargetNode>
struct ContainsNodeInExpr<TargetNode, TargetNode> {
    static constexpr bool value = true;
};

template <typename LHS, typename RHS, typename TargetNode>
struct ContainsNodeInExpr<OpAssign<LHS, RHS>, TargetNode> {
    static constexpr bool value = ContainsNodeInExpr<RHS, TargetNode>::value;
};

template <template <typename> class Op, typename Inner, typename TargetNode>
struct ContainsNodeInExpr<Op<Inner>, TargetNode> {
    static constexpr bool value = ContainsNodeInExpr<Inner, TargetNode>::value;
};

template <template <auto, typename> class Op, auto Pattern, typename Inner, typename TargetNode>
struct ContainsNodeInExpr<Op<Pattern, Inner>, TargetNode> {
    static constexpr bool value = ContainsNodeInExpr<Inner, TargetNode>::value;
};

template <template <typename, typename> class Op, typename T, typename U, typename TargetNode>
struct ContainsNodeInExpr<Op<T, U>, TargetNode> {
    static constexpr bool value = ContainsNodeInExpr<T, TargetNode>::value || ContainsNodeInExpr<U, TargetNode>::value;
};

template <template <typename, typename, typename> class Op, typename T, typename U, typename V, typename TargetNode>
struct ContainsNodeInExpr<Op<T, U, V>, TargetNode> {
    static constexpr bool value = ContainsNodeInExpr<T, TargetNode>::value ||
                                  ContainsNodeInExpr<U, TargetNode>::value || ContainsNodeInExpr<V, TargetNode>::value;
};

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
    using Combined = Concatenate_t<NodeInOutExprs, CurrentResults>;
    using ResultList = Map_t<ExtractResultType, NodeInOutExprs>;
    using NewExclude = Concatenate_t<ExcludeList, ResultList>;
    using NewRest = Unique_t<Concatenate_t<NodeInOutExprs, TypeList<Rest...>>>;

public:
    using Type = typename CollectOtherInputRefs<NewRest, FullList, NewExclude, Combined>::Type;
};

template <typename ExprList, typename FullList, typename ExcludeList, typename ResultList = TypeList<>>
struct FindLocalVarReferences;

template <typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<>, FullList, ExcludeList, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
    using OtherInputs = TypeList<>;
};

template <typename First, typename... Rest, typename FullList, typename ExcludeList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<First, Rest...>, FullList, ExcludeList, TypeList<ResultTypes...>> {
public:
    using CurrentResults = TypeList<ResultTypes...>;
    using LHS = typename First::LhsType;
    using DirectReferences = typename FilterRefLocalVar<FullList, LHS>::Type;
    using NewExclude = Concatenate_t<ExcludeList, TypeList<LHS>>;

    using DirectOtherInputs = typename FindAllUnhandledInputs<DirectReferences, FullList, NewExclude>::Type;
    using NewExcludeForRecursive = Concatenate_t<NewExclude, Map_t<ExtractResultType, DirectOtherInputs>>;

    using RecursiveRefResult = FindLocalVarReferences<DirectReferences, FullList, NewExcludeForRecursive>;
    using RecursiveResult = typename RecursiveRefResult::Type;
    using RecursiveOtherInputs = typename RecursiveRefResult::OtherInputs;

    using DirectAndRecursiveOtherInputs = Concatenate_t<DirectOtherInputs, RecursiveOtherInputs>;
    using NewExcludeWithOtherInputs =
        Concatenate_t<NewExcludeForRecursive, Map_t<ExtractResultType, DirectAndRecursiveOtherInputs>>;

    // 多个Reduce场景
    using NextRefResult =
        FindLocalVarReferences<TypeList<Rest...>, FullList, NewExcludeWithOtherInputs, CurrentResults>;
    using NextResult = typename NextRefResult::Type;
    using NextOtherInputs = typename NextRefResult::OtherInputs;

    using CombinedOtherInputs = Concatenate_t<DirectAndRecursiveOtherInputs, NextOtherInputs>;

    using RefsSoFar = Concatenate_t<DirectReferences, RecursiveResult, NextResult>;

    // Reduce后计算存在多输入表达式的场景
    using OtherInputRefs = typename CollectOtherInputRefs<
        CombinedOtherInputs, FullList, NewExcludeWithOtherInputs, Concatenate_t<CurrentResults, RefsSoFar>>::Type;
    using CombinedRefs = Concatenate_t<RefsSoFar, CombinedOtherInputs, OtherInputRefs>;
    using Type = Unique_t<CombinedRefs>;
    using OtherInputs = Unique_t<CombinedOtherInputs>;
};
} // namespace Atvoss::Reduce::Graph

#endif