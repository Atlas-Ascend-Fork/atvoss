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
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
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
    static constexpr bool shouldCheck = (current >= start);
    static constexpr bool currentCheck =
        shouldCheck ? ContainsNodeInExpr<typename First::RhsType, TargetNode>::value : false;
    static constexpr bool rest = IsNodeUsedAfter<TypeList<Rest...>, TargetNode, start, current + 1>::value;

public:
    static constexpr bool value = currentCheck || rest;
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
    using CurrentLHS = typename CurrentOpAssign::LhsType;
    static constexpr bool value = CheckReduceInput<ReduceOpList, CurrentLHS>::value;
};

template <typename List, typename TargetNode>
struct FilterLocalVar {
    using Type = Filter_t<ContainsNodeInRefList<TargetNode>::template Type, List>;
};

template <typename List, typename FullList, typename Result = TypeList<>>
struct FindLocalVarReferences;

template <typename FullList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<>, FullList, TypeList<ResultTypes...>> {
    using Type = TypeList<ResultTypes...>;
};

template <typename First, typename... Rest, typename FullList, typename... ResultTypes>
struct FindLocalVarReferences<TypeList<First, Rest...>, FullList, TypeList<ResultTypes...>> {
public:
    // 提取当前元素的 LHS 类型
    using LHS = typename First::LhsType;

    // 在 FullList 中过滤出包含该 LHS 的所有语句
    using DirectReferences = typename FilterLocalVar<FullList, LHS>::Type;

    // 递归处理 DirectReferences，找到所有后续依赖
    using RecursiveReferences = typename FindLocalVarReferences<DirectReferences, FullList, TypeList<>>::Type;

    // 递归处理 Rest，继续处理其他 LocalVar
    using NextReferences = typename FindLocalVarReferences<TypeList<Rest...>, FullList, TypeList<ResultTypes...>>::Type;

    // 合并：DirectReferences + RecursiveReferences + NextReferences
    using Combined = Concatenate_t<DirectReferences, Concatenate_t<RecursiveReferences, NextReferences>>;

    // 去重
    using Type = Unique_t<Combined>;
};
} // namespace Atvoss::Reduce::Graph

#endif