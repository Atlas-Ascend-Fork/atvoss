/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_REDUCE_BLOCK_EXPR_PREPROC_H_
#define ATVOSS_INCLUDE_REDUCE_BLOCK_EXPR_PREPROC_H_

#include <type_traits>
#include "common/type_def.h"
#include "reduce/graph/dag.h"
#include "graph/expr_linearizer.h"
#include "graph/compute_preproc.h"
#include "reduce/utils/reduce_utils.h"

namespace Atvoss::Reduce {

using Atvoss::LocalVars_t;
using Atvoss::Params_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Find_v;
using Atvoss::Util::FindLast_v;
using Atvoss::Util::ForEach;
using Atvoss::Util::Map_t;
using Atvoss::Util::Reverse_t;
using Atvoss::Util::Size_v;

template <typename ExprList, typename Param>
struct FirstAndLastUse {
    static constexpr std::size_t firstUse = Find_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    static constexpr std::size_t lastUse = FindLast_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    using Type =
        TypeList<Param, std::integral_constant<std::size_t, firstUse>, std::integral_constant<std::size_t, lastUse>>;
};

template <typename ExprList>
struct FirstAndLastUseFinder {
    template <typename Param>
    using Type = FirstAndLastUse<ExprList, Param>;
};

template <typename Expression, typename D>
struct ComputeInfo {
    using Expr = typename Expression::Type;
    using Dag = D;
};

template <bool IsBinaryAcc = true, typename T>
__host_aicore__ constexpr auto PreProcessComputeExpr(const Expression<T>& expr)
{
    using ExprT = T;
    using OriExprList = typename Atvoss::FlattenAtOpAndThen<ExprT>::Type;
    static_assert(Size_v<OriExprList> > 0, "Compute expression is empty.");

    using DagX = Graph::ReduceAutoDag<OriExprList, IsBinaryAcc>;

    using PreReduceList = typename DagX::PreReduceList;
    using ReducedList = typename DagX::ReducedList;
    using CombinedExprList = Concatenate_t<PreReduceList, ReducedList>;

    using AllParams = Params_t<CombinedExprList>;
    using AllLocalVars = LocalVars_t<CombinedExprList>;

    using ParamUseList = Map_t<FirstAndLastUseFinder<CombinedExprList>::template Type, AllParams>;
    using LocalVarUseList = Map_t<FirstAndLastUseFinder<CombinedExprList>::template Type, AllLocalVars>;

    auto result1 = ForEach(Reverse_t<ParamUseList>{}, Atvoss::Graph::AllocInserter{}, CombinedExprList{});
    auto result2 = ForEach(ParamUseList{}, Atvoss::Graph::FreeInserter{}, result1);
    auto result3 = ForEach(Reverse_t<LocalVarUseList>{}, Atvoss::Graph::AllocInserter{}, result2);
    auto result4 = ForEach(LocalVarUseList{}, Atvoss::Graph::FreeInserter{}, result3);

    using LastExpression = typename BuildExpression<decltype(result4)>::Type;
    return ComputeInfo<LastExpression, DagX>{};
};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_EXPR_PREPROC_H_
