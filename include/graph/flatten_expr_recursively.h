/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ATVOSS_GRAPH_FLATTEN_EXPR_RECURSIVELY_H__
#define __ATVOSS_GRAPH_FLATTEN_EXPR_RECURSIVELY_H__

#include "utils/utility.h"
#include "expression/expr_template.h"

namespace Atvoss::Graph {
/*--------------------------------------------------------------------------------------------------------------------*/
using Atvoss::IsParam_v;
using Atvoss::Util::Append_t;
using Atvoss::Util::IsTypeList_v;
using Atvoss::Util::Size_v;
/*--------------------------------------------------------------------------------------------------------------------*/
using ParaConst = Param<1, int, ParamUsage::IN>;

/* main template */
template <typename Op, typename... New>
struct ReplaceOp;

/* specially for UnaryOp */
template <template <typename> class Op, typename Old, typename New>
struct ReplaceOp<Op<Old>, New> {
    using Type = Op<New>;
};

template <template <auto, typename> class Op, auto Arg, typename Old, typename New>
struct ReplaceOp<Op<Arg, Old>, New> {
    using Type = Op<Arg, New>;
};

/* specially for BinaryOp */
template <template <typename, typename> class Op, typename OldL, typename OldR, typename NewL, typename NewR>
struct ReplaceOp<Op<OldL, OldR>, NewL, NewR> {
    using Type = Op<NewL, NewR>;
};
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename LocalVarList, typename TargetOp>
constexpr bool FindLocalVarIndex = false;

template <typename TargetOp>
constexpr bool FindLocalVarIndex<TypeList<>, TargetOp> = false;

template <typename Head, typename... Tail, typename TargetOp>
constexpr bool FindLocalVarIndex<TypeList<Head, Tail...>, TargetOp> =
    std::is_same_v<typename Head::Like, TargetOp> ? true : FindLocalVarIndex<TypeList<Tail...>, TargetOp>;

template <typename LocalVarList, typename TargetOp>
constexpr size_t GetLocalVarIndex = 0;

template <typename TargetOp>
constexpr size_t GetLocalVarIndex<TypeList<>, TargetOp> = 0;

template <typename Head, typename... Tail, typename TargetOp>
constexpr size_t GetLocalVarIndex<TypeList<Head, Tail...>, TargetOp> =
    std::is_same_v<typename Head::Like, TargetOp> ? Head::number : GetLocalVarIndex<TypeList<Tail...>, TargetOp>;
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename Operand, size_t Idx, typename M>
constexpr auto CreateLocalVar(M map)
{
    if constexpr (IsParam_v<Operand>) {
        return std::make_pair(map, Operand{});
    } else if constexpr (FindLocalVarIndex<M, Operand>) {
        constexpr size_t idx = GetLocalVarIndex<M, Operand>;
        using LocalVarType = LocalVar<idx, float, ParaConst>;
        return std::make_pair(map, LocalVarType{});
    } else {
        using NewLocalVar = LocalVar<Idx, float, Operand>;
        using MergedMap = Append_t<M, NewLocalVar>;
        using LocalVarType = LocalVar<Idx, float, ParaConst>;
        return std::make_pair(MergedMap{}, LocalVarType{});
    }
}

template <typename Lhs, typename Rhs, typename... Operands>
using MakeAssignOp = OpAssign<Lhs, typename ReplaceOp<Rhs, Operands...>::Type>;
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename Op, typename M, typename Expr>
constexpr auto FlattenOp(Op op, M map, Expr expr);

template <typename Operand, typename M, typename Expr>
constexpr auto FlattenOperand(Operand operand, M map, Expr expr)
{
    auto [mapAfterOperand, localVarForOperand] = CreateLocalVar<Operand, Size_v<M> + 1>(map);
    constexpr bool needFlattenOperand = !IsParam_v<Operand> && !FindLocalVarIndex<M, Operand>;
    if constexpr (needFlattenOperand) {
        auto [mapAfterFlattenOperand, exprAfterFlattenOperand] = FlattenOp(operand, mapAfterOperand, expr);
        return std::tuple(localVarForOperand, mapAfterFlattenOperand, exprAfterFlattenOperand);
    } else {
        return std::tuple(localVarForOperand, mapAfterOperand, expr);
    }
}

template <typename Op, typename M, typename Expr>
constexpr auto FlattenUnaryOp(Op op, M map, Expr expr)
{
    using Operand = typename Op::DataType;
    auto [localVar, mapAfterFlattenOperand, exprAfterFlattenOperand] = FlattenOperand(Operand{}, map, expr);

    using NewAssign = MakeAssignOp<LocalVar<Size_v<M>, float, ParaConst>, Op, decltype(localVar)>;
    using FinalExpr = Append_t<decltype(exprAfterFlattenOperand), NewAssign>;
    return std::make_pair(mapAfterFlattenOperand, FinalExpr{});
}

template <typename Op, typename M, typename Expr>
constexpr auto FlattenBinaryOp(Op op, M map, Expr expr)
{
    using LhsOperand = typename Op::LhsType;
    using RhsOperand = typename Op::RhsType;
    auto [localVarForLhs, mapAfterFlattenLhsOperand, exprAfterFlattenLhsOperand] =
        FlattenOperand(LhsOperand{}, map, expr);
    auto [localVarForRhs, mapAfterFlattenRhsOperand, exprAfterFlattenRhsOperand] =
        FlattenOperand(RhsOperand{}, mapAfterFlattenLhsOperand, exprAfterFlattenLhsOperand);

    using NewAssign =
        MakeAssignOp<LocalVar<Size_v<M>, float, ParaConst>, Op, decltype(localVarForLhs), decltype(localVarForRhs)>;
    using FinalExpr = Append_t<decltype(exprAfterFlattenRhsOperand), NewAssign>;
    return std::make_pair(mapAfterFlattenRhsOperand, FinalExpr{});
}

template <typename Op, typename M, typename Expr>
constexpr auto FlattenOp(Op op, M map, Expr expr)
{
    static_assert(IsUnaryOp_v<Op> || IsBinaryOp_v<Op>, "Op must be either UnaryOp or BinaryOp.");

    if constexpr (IsUnaryOp_v<Op>) {
        return FlattenUnaryOp(op, map, expr);
    } else {
        return FlattenBinaryOp(op, map, expr);
    }
}
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename Lhs, typename Rhs, typename M, typename Expr>
constexpr auto FlattenOpAssign(OpAssign<Lhs, Rhs>, M map, Expr expr)
{
    if constexpr (IsParam_v<Rhs>) {
        using FinalExpr = Append_t<Expr, OpAssign<Lhs, Rhs>>;
        return std::make_pair(map, FinalExpr{});
    } else if constexpr (IsUnaryOp_v<Rhs>) {
        using Operand = typename Rhs::DataType;
        auto [localVar, mapAfterFlattenOperand, exprAfterFlattenOperand] = FlattenOperand(Operand{}, map, expr);

        using NewAssign = MakeAssignOp<Lhs, Rhs, decltype(localVar)>;
        using FinalExpr = Append_t<decltype(exprAfterFlattenOperand), NewAssign>;
        return std::make_pair(mapAfterFlattenOperand, FinalExpr{});
    } else if constexpr (IsBinaryOp_v<Rhs>) {
        using LhsOperand = typename Rhs::LhsType;
        using RhsOperand = typename Rhs::RhsType;
        auto [localVarForLhs, mapAfterFlattenLhsOperand, exprAfterFlattenLhsOperand] =
            FlattenOperand(LhsOperand{}, map, expr);
        auto [localVarForRhs, mapAfterFlattenRhsOperand, exprAfterFlattenRhsOperand] =
            FlattenOperand(RhsOperand{}, mapAfterFlattenLhsOperand, exprAfterFlattenLhsOperand);

        using NewAssign = MakeAssignOp<Lhs, Rhs, decltype(localVarForLhs), decltype(localVarForRhs)>;
        using FinalExpr = Append_t<decltype(exprAfterFlattenRhsOperand), NewAssign>;
        return std::make_pair(mapAfterFlattenRhsOperand, FinalExpr{});
    } else {
        static_assert(true, "OpAssign's Right only support Param/UnaryOp/BinaryOp");
    }
}

template <typename M, typename Expr>
constexpr auto FlattenTypeList(TypeList<>, M map, Expr expr)
{
    return std::make_pair(map, expr);
}

template <typename Head, typename... Tail, typename M, typename Expr>
constexpr auto FlattenTypeList(TypeList<Head, Tail...>, M map, Expr expr)
{
    static_assert(IsOpAssign_v<Head>, "Each one of TypeList must be a OpAssign");

    auto [curMap, curExpr] = FlattenOpAssign(Head{}, map, expr);
    return FlattenTypeList(TypeList<Tail...>{}, curMap, curExpr);
}
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename Expr>
constexpr auto FlattenExprRecursively(Expr expr)
{
    if constexpr (IsOpAssign_v<Expr>) {
        auto [map, finalExpr] = FlattenOpAssign(expr, TypeList<>{}, TypeList<>{});
        return finalExpr;
    } else if constexpr (IsTypeList_v<Expr>) {
        auto [map, finalExpr] = FlattenTypeList(expr, TypeList<>{}, TypeList<>{});
        return finalExpr;
    } else {
        static_assert(true, "Expr must be a OpAssign or TypeList");
    }
}
/*--------------------------------------------------------------------------------------------------------------------*/
} // namespace Atvoss::Graph

#endif // __ATVOSS_GRAPH_FLATTEN_EXPR_RECURSIVELY_H__