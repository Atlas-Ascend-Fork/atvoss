/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file expr_preproc.h
 * @brief Reduce Block 表达式预处理: 展平→DAG分析→资源管理插入
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

/**
 * @brief 查找参数在表达式列表中的首次和末次使用位置
 * @tparam ExprList 表达式 TypeList
 * @tparam Param 要查找的参数类型
 */
template <typename ExprList, typename Param>
struct FirstAndLastUse {
    static constexpr std::size_t firstUse = Find_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    static constexpr std::size_t lastUse = FindLast_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    using Type =
        TypeList<Param, std::integral_constant<std::size_t, firstUse>, std::integral_constant<std::size_t, lastUse>>;
};

/**
 * @brief FirstAndLastUse 的 Finder 模板, 用于 Map_t
 * @tparam ExprList 表达式 TypeList
 */
template <typename ExprList>
struct FirstAndLastUseFinder {
    template <typename Param>
    using Type = FirstAndLastUse<ExprList, Param>;
};

/**
 * @brief 编译期表达式预处理结果, 含优化后的表达式和 DAG 分析结果
 * @tparam Expression 优化后的表达式类型 (BuildExpression 输出)
 * @tparam D DAG 类型 (ReduceAutoDag)
 */
template <typename Expression, typename D>
struct ComputeInfo {
    using Expr = typename Expression::Type;   // 优化后的表达式 OpAndThen 链
    using Dag = D;                            // DAG 分析结果
};

/**
 * @brief 表达式预处理管线入口
 *
 * 处理流程:
 *   1. FlattenAtOpAndThen: 表达式树 → 扁平 TypeList
 *   2. ReduceAutoDag: DAG 分析, 划分为 PreReduce/Reduced 两阶段
 *   3. FirstAndLastUseFinder: 分析每个 Param/LocalVar 的首次/末次使用位置
 *   4. AllocInserter/FreeInserter: 在首次使用前插入 OpAlloc, 末次使用后插入 OpFree
 *   5. BuildExpression: TypeList → OpAndThen 表达式链
 *
 * @tparam IsBinaryAcc 是否启用二分累加
 * @tparam T 原始表达式类型
 * @param expr 原始表达式 (Compute 函数生成)
 * @return ComputeInfo{优化后的表达式, DAG 分析结果}
 */
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
    using LastExpression = typename BuildExpression<CombinedExprList>::Type;
    return ComputeInfo<LastExpression, DagX>{};
};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_EXPR_PREPROC_H_
