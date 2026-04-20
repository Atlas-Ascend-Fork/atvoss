/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_GRAPH_EXPR_OPERATIONS_H
#define ATVOSS_GRAPH_EXPR_OPERATIONS_H

#include "utils/utility.h"
#include "expression/expr_template.h"

namespace Atvoss::Graph {

using Atvoss::Util::Concatenate_t;
using Atvoss::Util::TypeList;

template <typename T>
struct ExtractInputs {
    using Type = TypeList<>;
};

template <size_t N, typename T, Atvoss::ParamUsage U, size_t RN>
struct ExtractInputs<Atvoss::Param<N, T, U, RN>> {
    using Type = TypeList<Atvoss::Param<N, T, U, RN>>;
};

template <size_t N, typename T, typename L>
struct ExtractInputs<Atvoss::LocalVar<N, T, L>> {
    using Type = TypeList<Atvoss::LocalVar<N, T, L>>;
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
struct ContainsNodeInExpr<Atvoss::OpAssign<LHS, RHS>, TargetNode> {
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

} // namespace Atvoss::Graph

#endif