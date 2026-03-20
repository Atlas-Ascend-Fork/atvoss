/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXPR_LINEARIZER_H
#define ATVOSS_EXPR_LINEARIZER_H
#include <cstddef>

#include "expression/expr_template.h"
#include "utils/utility.h"
#include "expr_remove_redundant_cast.h"

namespace Atvoss {
namespace Detail {
/**
 *
 * @tparam Expr TypeList
 * @tparam From 需要替换的类型
 * @tparam To 目标类型
 */
template <typename Expr, typename From, typename To>
struct ReplaceExpr;

/**
 * 递归替换引擎
 * @tparam Expr TypeList
 * @tparam From 需要替换的类型
 * @tparam To 目标类型
 */
template <typename Expr, typename From, typename To>
struct ReplaceRecursive {
    using Type = Expr; // 默认不处理
};

// 基本情况：完全匹配则替换
template <typename Expr, typename From, typename To>
struct ReplaceExpr {
    using Type = std::conditional_t<
        std::is_same_v<Expr, From>, To,
        // 否则递归下降
        typename ReplaceRecursive<Expr, From, To>::Type>;
};

// 对 Param 不再递归
template <size_t N, typename T, auto Usage, size_t R, typename From, typename To>
struct ReplaceExpr<Atvoss::Param<N, T, Usage, R>, From, To> {
    using Type = Atvoss::Param<N, T, Usage, R>;
};

// 二元操作符递归
template <template <typename, typename> class Op, typename LHS, typename RHS, typename From, typename To>
struct ReplaceRecursive<Op<LHS, RHS>, From, To> {
    using newLhs = typename ReplaceExpr<LHS, From, To>::Type;
    using newRhs = typename ReplaceExpr<RHS, From, To>::Type;
    using Type = Op<newLhs, newRhs>;
};

// 一元操作符递归
template <template <typename> class Op, typename Inner, typename From, typename To>
struct ReplaceRecursive<Op<Inner>, From, To> {
    using newInner = typename ReplaceExpr<Inner, From, To>::Type;
    using Type = Op<newInner>;
};

// 带非类型模板参数的操作符
template <template <auto, typename> class Op, auto N, typename Inner, typename From, typename To>
struct ReplaceRecursive<Op<N, Inner>, From, To> {
    using newInner = typename ReplaceExpr<Inner, From, To>::Type;
    using Type = Op<N, newInner>;
};

// 匹配OpCast
template <template <auto, typename, typename> class Op, auto N, typename R, typename Inner, typename From, typename To>
struct ReplaceRecursive<Op<N, R, Inner>, From, To> {
    using newInner = typename ReplaceExpr<Inner, From, To>::Type;
    using Type = Op<N, R, newInner>;
};

// ========================================================
// ExtractTypeListPostOrder: 后序遍历提取表达式列表
// ========================================================
template <typename Expr, typename = void>
struct ExtractTypeListPostOrder;

// 终止：Param 不展开
template <size_t N, typename T, auto Usage, size_t R>
struct ExtractTypeListPostOrder<Atvoss::Param<N, T, Usage, R>> {
    using Type = Atvoss::Util::TypeList<>;
};

// 终止：LocalVar 不展开
template <size_t N, typename T, typename U>
struct ExtractTypeListPostOrder<Atvoss::LocalVar<N, T, U>> {
    using Type = Atvoss::Util::TypeList<>;
};

template <template <typename, typename> class Op, typename LHS, typename RHS>
struct ExtractTypeListPostOrder<Op<LHS, RHS>, std::enable_if_t<IsBinaryOp_v<Op<LHS, RHS>>>> {
    using lhsList = typename ExtractTypeListPostOrder<LHS>::Type;
    using rhsList = typename ExtractTypeListPostOrder<RHS>::Type;
    using concatList = typename Atvoss::Util::Concatenate<
        typename std::conditional_t<
            Atvoss::Util::IsSpecializationOf_v<Atvoss::OpAssign, Op<LHS, RHS>>,
            typename Atvoss::Util::Concatenate<rhsList, lhsList>, // 赋值操作符：先右后左
            typename Atvoss::Util::Concatenate<lhsList, rhsList>  // 先左后右
            >::Type,
        std::conditional_t<
            Atvoss::Util::IsSpecializationOf_v<Atvoss::OpAndThen, Op<LHS, RHS>>, Atvoss::Util::TypeList<>,
            Atvoss::Util::TypeList<Op<LHS, RHS>>>>::Type;
    using Type = concatList;
};

// 一元操作符
template <template <typename> class Op, typename Expr>
struct ExtractTypeListPostOrder<Op<Expr>> {
    using innerList = typename ExtractTypeListPostOrder<Expr>::Type;
    using Type = typename Atvoss::Util::Concatenate<innerList, Atvoss::Util::TypeList<Op<Expr>>>::Type;
};

template <template <auto, typename> class Op, auto Pattern, typename Expr>
struct ExtractTypeListPostOrder<Op<Pattern, Expr>> {
    using innerList = typename ExtractTypeListPostOrder<Expr>::Type;
    using Type = typename Atvoss::Util::Concatenate<innerList, Atvoss::Util::TypeList<Op<Pattern, Expr>>>::Type;
};

template <template <auto, typename, typename> class Op, auto Pattern, typename R, typename Expr>
struct ExtractTypeListPostOrder<Op<Pattern, R, Expr>> {
    using innerList = typename ExtractTypeListPostOrder<Expr>::Type;
    using Type = typename Atvoss::Util::Concatenate<innerList, Atvoss::Util::TypeList<Op<Pattern, R, Expr>>>::Type;
};

using Atvoss::Util::TypeList;

// LocalVar的number与其右边的表达式的映射关系
template <size_t ID, typename Expr>
struct VarDef {};

// 查找变量定义
template <size_t ID, typename DefList>
struct FindVarDef;

template <size_t ID>
struct FindVarDef<ID, TypeList<>> {
    using Type = void;
};

template <size_t ID, typename Expr, typename... Rest>
struct FindVarDef<ID, TypeList<VarDef<ID, Expr>, Rest...>> {
    using Type = Expr;
};

template <size_t ID, size_t OtherID, typename Expr, typename... Rest>
struct FindVarDef<ID, TypeList<VarDef<OtherID, Expr>, Rest...>> {
    using Type = typename FindVarDef<ID, TypeList<Rest...>>::Type;
};

// 检查 LocalVar 是否会被后面的 Param 引用
template <size_t ID, typename StmtList>
struct IsLocalVarReferencedByParam;

template <size_t ID>
struct IsLocalVarReferencedByParam<ID, TypeList<>> {
    static constexpr bool value = false;
};

template <size_t ID, typename NextStmt, typename... OtherStmts>
struct IsLocalVarReferencedByParam<ID, TypeList<NextStmt, OtherStmts...>> {
    // 检查 NextStmt 是否是 OpAssign<Param<...>, LocalVar<ID, ...>>
    template <typename Stmt>
    struct CheckStmt;

    template <size_t PID, typename PT, ParamUsage PU, size_t LID, typename LT, typename LSrc>
    struct CheckStmt<OpAssign<Param<PID, PT, PU>, LocalVar<LID, LT, LSrc>>> {
        static constexpr bool value = (LID == ID);
    };

    template <typename Other>
    struct CheckStmt {
        static constexpr bool value = false;
    };

    static constexpr bool current = CheckStmt<NextStmt>::value;
    static constexpr bool rest = IsLocalVarReferencedByParam<ID, TypeList<OtherStmts...>>::value;
    static constexpr bool value = current || rest;
};

/**
 * 把无效的localVar简化掉，比如localVar1 = in1*in1, out = localVar1简化成 out = in1*in1.
 * @tparam TypeList 需要简化的表达式的初始列表
 * @tparam DefList VarDef的列表，存放的是LocalVar的number与其右边的表达式的映射关系
 * @tparam Result 化简后的结果
 */
template <typename TypeList, typename DefList = Atvoss::Util::TypeList<>, typename Result = Atvoss::Util::TypeList<>>
struct SimplifyImpl;

// 递归终止条件：需要简化的列表为空
template <typename... Defs, typename... ResultTypes>
struct SimplifyImpl<TypeList<>, TypeList<Defs...>, TypeList<ResultTypes...>> {
    using result = TypeList<ResultTypes...>;
};

// 递归处理
template <typename First, typename... Rest, typename... Defs, typename... ResultTypes>
struct SimplifyImpl<TypeList<First, Rest...>, TypeList<Defs...>, TypeList<ResultTypes...>> {
private:
    // 辅助：处理 LocalVar 定义语句
    template <size_t ID, typename T, typename Likes, typename Expr>
    struct ProcessLocalVarDef {
        // 检查这个 LocalVar 是否会被后面的 Param 引用
        static constexpr bool willBeReplaced = IsLocalVarReferencedByParam<ID, TypeList<Rest...>>::value;

        using newDefs = TypeList<Defs..., VarDef<ID, Expr>>;

        // 如果会被替换，就不保留这个语句
        using Type = std::conditional_t<
            willBeReplaced, std::integral_constant<int, 0>, // 标记为删除
            OpAssign<LocalVar<ID, T, Likes>, Expr>          // 保留
            >;
    };

    // 辅助：处理 Param 赋值给 LocalVar
    template <size_t PID, typename PT, ParamUsage PU, size_t LID, typename LT, typename Likes, typename RestList>
    struct ProcessParamAssign {
        // 查找 LocalVar 的定义
        using found = FindVarDef<LID, TypeList<Defs...>>;
        using newDefs = TypeList<Defs...>;

        using Type = std::conditional_t<
            !std::is_same_v<typename found::Type, void>,
            OpAssign<Param<PID, PT, PU>, typename found::Type>,    // 替换为表达式
            OpAssign<Param<PID, PT, PU>, LocalVar<LID, LT, Likes>> // 保持原样
            >;

        template <typename T>
        using ReplaceLocalVar2Param = ReplaceExpr<T, LocalVar<LID, LT, Likes>, Param<PID, PT, PU>>;
        using restList = Util::Map_t<ReplaceLocalVar2Param, RestList>;
    };

    // 辅助：处理其他语句
    template <typename Stmt>
    struct ProcessOther {
        using newDefs = TypeList<Defs...>;
        using Type = Stmt;
    };

    // 分发处理逻辑
    template <typename Stmt, typename RestList>
    struct ProcessStmt;

    // 匹配 OpAssign<LocalVar<...>, Expr>
    template <size_t ID, typename T, typename Likes, typename Expr, typename RestList>
    struct ProcessStmt<OpAssign<LocalVar<ID, T, Likes>, Expr>, RestList> {
        using processor = ProcessLocalVarDef<ID, T, Likes, Expr>;
        using newDefs = typename processor::newDefs;
        using resultType = typename processor::Type;
        using restList = RestList;
    };

    // 匹配 OpAssign<Param<...>, LocalVar<...>>
    template <size_t PID, typename PT, ParamUsage PU, size_t LID, typename LT, typename Likes, typename RestList>
    struct ProcessStmt<OpAssign<Param<PID, PT, PU>, LocalVar<LID, LT, Likes>>, RestList> {
        using processor = ProcessParamAssign<PID, PT, PU, LID, LT, Likes, RestList>;
        using newDefs = typename processor::newDefs;
        using resultType = typename processor::Type;
        using restList = typename processor::restList;
    };

    // 匹配其他类型
    template <typename Stmt, typename RestList>
    struct ProcessStmt {
        using processor = ProcessOther<Stmt>;
        using newDefs = typename processor::newDefs;
        using resultType = typename processor::Type;
        using restList = RestList;
    };

    // 处理当前语句
    using processor = ProcessStmt<First, TypeList<Rest...>>;
    using newDefs = typename processor::newDefs;
    using currentResult = typename processor::resultType;

    // 判断是否需要将当前结果添加到最终列表中
    using nextResult = std::conditional_t<
        std::is_same_v<currentResult, std::integral_constant<int, 0>>,
        // 如果是删除标记，不添加
        TypeList<ResultTypes...>,
        // 否则添加
        TypeList<ResultTypes..., currentResult>>;

    // 继续处理剩余部分
    using next = SimplifyImpl<typename processor::restList, newDefs, nextResult>;

public:
    using result = typename next::result;
};

// 将 [LocalVar1 = in1 * ini1, out = LocalVar1] 化简成 out = in1 * in1
template <typename List>
struct Simplify {
    using Type = typename SimplifyImpl<List>::result;
};

// ========================================================
// OptimizeWithLocalVars: 将中间结果缓存为 LocalVar
// ========================================================
template <typename ExprList, size_t LocalVarNumber, typename Processed>
struct OptimizeWithLocalVarsImpl;

// 终止条件
template <size_t LocalVarNumber, typename Processed>
struct OptimizeWithLocalVarsImpl<Atvoss::Util::TypeList<>, LocalVarNumber, Processed> {
    using Type = Processed;
};

// 主递归：处理第一个表达式
template <typename First, typename... Rest, size_t LocalVarNumber, typename Processed>
struct OptimizeWithLocalVarsImpl<Atvoss::Util::TypeList<First, Rest...>, LocalVarNumber, Processed> {
private:
    // 判断是否应该缓存
    static constexpr bool shouldCache = !Atvoss::IsParam_v<First>;
    static constexpr bool firstIsOpAssign = Atvoss::Util::IsSpecializationOf_v<Atvoss::OpAssign, First>;
    static constexpr auto NextLocalVarNumber = firstIsOpAssign ? LocalVarNumber : LocalVarNumber + 1;

    // LocalVar 类型
    using localVarType = Atvoss::LocalVar<
        LocalVarNumber, typename First::RetType, Atvoss::Param<1ul, typename First::RetType, (Atvoss::ParamUsage)0>>;

    // 构造赋值语句
    using assignmentType = std::conditional_t<firstIsOpAssign, First, Atvoss::OpAssign<localVarType, First>>;

    template <typename Expr>
    using ReplaceOne = typename ReplaceExpr<Expr, First, localVarType>::Type;
    using updatedRest = Atvoss::Util::TypeList<ReplaceOne<Rest>...>;

    // 下一步状态
    using nextProcessed = typename std::conditional_t<
        shouldCache, Atvoss::Util::Append<Processed, assignmentType>, Util::TypeWrapper<Processed>>::Type;

    using nextState = std::conditional_t<
        shouldCache, OptimizeWithLocalVarsImpl<updatedRest, NextLocalVarNumber, nextProcessed>,
        OptimizeWithLocalVarsImpl<Atvoss::Util::TypeList<Rest...>, LocalVarNumber, nextProcessed>>;

public:
    using Type = typename nextState::Type;
};

// 主入口
template <typename ExprList>
struct OptimizeWithLocalVars : OptimizeWithLocalVarsImpl<ExprList, 1, Atvoss::Util::TypeList<>> {};

template <typename ExprList, typename Processed>
struct OptimizeBindBuffExprImpl;

// 终止条件
template <typename Processed>
struct OptimizeBindBuffExprImpl<Atvoss::Util::TypeList<>, Processed> {
    using Type = Processed;
};

template <typename T>
struct SimplifyAssignOfOpParam {
    using Type = T;
};

template <typename LHS, typename RHS>
struct SimplifyAssignOfOpParam<Atvoss::OpAssign<LHS, RHS>> {
    using Type = LHS;
};

template <typename T>
struct SimplifyAssign;

// 对 Param 不再递归
template <size_t N, typename T, auto Usage, size_t R>
struct SimplifyAssign<Atvoss::Param<N, T, Usage, R>> {
    using Type = Atvoss::Param<N, T, Usage, R>;
};

template <typename LHS, typename RHS>
struct SimplifyAssign<Atvoss::OpAssign<LHS, RHS>> {
    using Type = Atvoss::OpAssign<LHS, typename SimplifyAssign<RHS>::Type>;
};

// 二元操作符通用匹配
template <template <typename, typename> class Op, typename LHS, typename RHS>
struct SimplifyAssign<Op<LHS, RHS>> {
    using Type = Op<typename SimplifyAssignOfOpParam<LHS>::Type, typename SimplifyAssignOfOpParam<RHS>::Type>;
};

// 一元操作符通用匹配
template <template <typename> class Op, typename Expr>
struct SimplifyAssign<Op<Expr>> {
    using Type = Op<typename SimplifyAssignOfOpParam<Expr>::Type>;
};

// 匹配带非类型模板参数的一元操作符：Op<Pat, E>
template <template <auto, typename> class Op, auto N, typename Expr>
struct SimplifyAssign<Op<N, Expr>> {
    using Type = Op<N, typename SimplifyAssignOfOpParam<Expr>::Type>;
};

// 匹配带非类型模板参数的一元操作符：Op<Pat, E>
template <auto N, typename R, typename Expr>
struct SimplifyAssign<Atvoss::OpCast<N, R, Expr>> {
    using Type = Atvoss::OpCast<N, R, typename SimplifyAssignOfOpParam<Expr>::Type>;
};

// First为OpAssign直接跳过，对OpAssign的右边进行化简，
// 比如把OpAdd(LocalVar2, OpAssign(LocalVar1, OpMul(in1*in1)))
// 化简成OpAdd(LocalVar2, LocalVar1)
template <typename First, typename... Rest, typename Processed>
struct OptimizeBindBuffExprImpl<Atvoss::Util::TypeList<First, Rest...>, Processed> {
private:
    // 判断是否应该缓存
    static constexpr bool shouldCache = !Atvoss::IsParam_v<First>;

    // LocalVar 类型
    using localVarType = First;

    // 化简OpAssign：把OpAssign(LocalVar3, OpAdd(LocalVar2, OpAssign(LocalVar1, OpMul(in1*in1))))
    // 化简成：OpAssign(LocalVar3, OpAdd(LocalVar2, LocalVar1))
    using assignSimplified = std::conditional_t<
        Atvoss::Util::IsSpecializationOf_v<Atvoss::OpAssign, First>, TypeList<typename SimplifyAssign<First>::Type>,
        TypeList<>>;

    // 下一步状态
    using nextProcessed = Atvoss::Util::Concatenate_t<Processed, assignSimplified>;

    using nextState = OptimizeBindBuffExprImpl<Atvoss::Util::TypeList<Rest...>, nextProcessed>;

public:
    using Type = typename nextState::Type;
};

// 主入口
template <typename ExprList>
struct OptimizeBindBuffExpr : OptimizeBindBuffExprImpl<ExprList, Atvoss::Util::TypeList<>> {};

template <typename Expr>
struct ExprLinearizer {
    using postOrderList = Atvoss::Util::Unique_t<typename ExtractTypeListPostOrder<Expr>::Type>;
    using removeRedundantCastExprList =
        decltype(Atvoss::Graph::RemoveRedundantCast<typename std::conditional_t<
                     Util::Size<typename LocalVars<Expr>::Type>::value == 0, OptimizeWithLocalVars<postOrderList>,
                     OptimizeBindBuffExpr<postOrderList>>::Type>());
    using optimizedList = typename Simplify<removeRedundantCastExprList>::Type;
};
} // namespace Detail

template <typename Expr>
__host_aicore__ constexpr auto ToLinearizerExpr(Expr expr)
{
    return typename ToOpAndThenExpr<typename Detail::ExprLinearizer<typename Expr::Type>::optimizedList>::Type();
}

} // namespace Atvoss

#endif // ATVOSS_EXPR_LINEARIZER_H