/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXPR_REMOVE_CAST_H
#define ATVOSS_EXPR_REMOVE_CAST_H

#include "utils/layout/layout.h"
#include "utils/patterns.h"
#include "operators/math_expression.h"

namespace Atvoss::Graph {
using Atvoss::Util::Append_t;
using Atvoss::Util::Drop_t;
using Atvoss::Util::Find_v;
using Atvoss::Util::Get_t;
using Atvoss::Util::IsEmpty_v;
using Atvoss::Util::Map_t;
using Atvoss::Util::Reverse_t;
using Atvoss::Util::Set_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::TypeList;

template <typename Expr, typename LocalVarType>
struct IsAssignedTo : std::false_type {};

template <typename RhsType, typename LocalVarType>
struct IsAssignedTo<OpAssign<LocalVarType, RhsType>, LocalVarType> : std::true_type {};

template <typename LocalVarType>
struct CheckForAssign {
    template <typename Expr>
    using Type = IsAssignedTo<Expr, LocalVarType>;
};

template <typename ExprList>
struct SafeReplaceRange {
    template <typename ReplaceItem>
    // TypeList<integral_constant<size_t, Pos>,
    //          LocalVar_to_replace, New_LocalVar_or_Param>
    struct Mapper {
        using LocalVarType = Get_t<ReplaceItem, 1>;
        static constexpr std::size_t startPos = Get_t<ReplaceItem, 0>{} + 1;
        static constexpr std::size_t writePos =
            startPos + Find_v<CheckForAssign<LocalVarType>::template Type, Drop_t<ExprList, startPos>>;
        using Type = TypeList<
            std::integral_constant<std::size_t, startPos>, std::integral_constant<std::size_t, writePos>,
            Get_t<ReplaceItem, 1>, Get_t<ReplaceItem, 2>>;
    };
};

template <typename Index, typename Data>
struct IndexedData {
    using IndexType = Index;
    using DataType = Data;

    IndexedData() = default;
    constexpr explicit IndexedData(Index)
    {}
    template <typename T>
    constexpr IndexedData(Index, T&& value) : data(std::forward<T>(value))
    {}

    Data data;
};

template <typename RemovePositions>
struct NotContains {
    template <typename T>
    using Type = std::bool_constant<!Atvoss::Util::Contains_v<RemovePositions, T>>;
};

template <typename Index, typename Data>
IndexedData(Index, Data) -> IndexedData<Index, Data>;

struct RedundantCastFinder {
    template <typename T, typename Result>
    constexpr auto operator()(T, Result) const
    {
        using Expr = typename T::Type;
        using NextIndex = std::integral_constant<std::size_t, typename Result::IndexType{} + 1>;
        using Data = typename Result::DataType;
        // 条件1：当前操作是赋值（OpAssign）
        if constexpr (Atvoss::Util::IsSpecializationOf_v<OpAssign, Expr>) {
            using TargetType = typename Expr::LhsType;
            using SourceType = typename Expr::RhsType;
            // 条件2：右值是类型转换（OpCast）
            if constexpr (IsOpCast_v<SourceType>) {
                using CastSourceType = typename SourceType::DataType;
                // 条件3：被转换的源数据是 局部变量或输入参数
                // 条件4：转换前后类型相同
                if constexpr (!IsLocalVar_v<CastSourceType> && !IsParam_v<CastSourceType>) {
                    return IndexedData<NextIndex, Data>{};
                } else if constexpr (std::is_same_v<typename CastSourceType::Type, typename TargetType::Type>) {
                    // TypeList<当前索引（冗余节点位置）, 被赋值的局部变量（TargetType）,
                    // 直接使用的源变量（CastSourceType）>
                    return IndexedData{
                        NextIndex{}, Atvoss::Util::Append_t<
                                         Data, TypeList<typename Result::IndexType, TargetType, CastSourceType>>{}};
                } else {
                    return IndexedData<NextIndex, Data>{};
                }
            } else {
                return IndexedData<NextIndex, Data>{};
            }
        } else {
            return IndexedData<NextIndex, Data>{};
        }
    }
};

template <std::size_t I, typename DeleteIndices>
struct IsDeleted : std::false_type {};

template <std::size_t I, std::size_t J, typename... Rest>
struct IsDeleted<I, TypeList<std::integral_constant<std::size_t, J>, Rest...>> {
    static constexpr bool value = (I == J) || IsDeleted<I, TypeList<Rest...>>::value;
};

template <typename ReplaceItemList>
struct ExtractDeleteIndices {
    using Type = TypeList<>;
};

template <>
struct ExtractDeleteIndices<TypeList<>> {
    using Type = TypeList<>;
};

template <typename Index, typename A, typename B, typename... Tail>
struct ExtractDeleteIndices<TypeList<TypeList<Index, A, B>, Tail...>> {
    static_assert(
        std::is_same_v<Index, std::integral_constant<std::size_t, Index::value>>,
        "First element must be std::integral_constant<std::size_t, ...>");

    using Rest = typename ExtractDeleteIndices<TypeList<Tail...>>::Type;
    using Type = Append_t<Rest, Index>;
};

template <typename Node, typename OldSym, typename NewSym>
struct ReplaceSymbol {
    using Type = std::conditional_t<std::is_same_v<Node, OldSym>, NewSym, Node>;
};

template <typename L, typename R, typename OldSym, typename NewSym>
struct ReplaceSymbol<OpAssign<L, R>, OldSym, NewSym> {
    using Type =
        OpAssign<typename ReplaceSymbol<L, OldSym, NewSym>::Type, typename ReplaceSymbol<R, OldSym, NewSym>::Type>;
};

template <CastMode Mode, typename R, typename T, typename OldSym, typename NewSym>
struct ReplaceSymbol<OpCast<Mode, R, T>, OldSym, NewSym> {
    using Type = OpCast<Mode, R, typename ReplaceSymbol<T, OldSym, NewSym>::Type>;
};

template <std::size_t I, typename Node, typename RangeList>
struct ApplyReplacements;

template <std::size_t I, typename Node>
struct ApplyReplacements<I, Node, TypeList<>> {
    using Type = Node;
};

template <
    std::size_t I, typename Node, std::size_t Start, std::size_t End, typename OldSym, typename NewSym,
    typename... Rest>
struct ApplyReplacements<
    I, Node,
    TypeList<
        TypeList<std::integral_constant<std::size_t, Start>, std::integral_constant<std::size_t, End>, OldSym, NewSym>,
        Rest...>> {
    static constexpr bool in_range = (Start <= I && I < End);

    using NodeAfterCurrent = std::conditional_t<in_range, typename ReplaceSymbol<Node, OldSym, NewSym>::Type, Node>;

    using Type = typename ApplyReplacements<I, NodeAfterCurrent, TypeList<Rest...>>::Type;
};

template <typename Lhs, typename Rhs, bool IsRedundant>
struct UnwrapOrNot {
    using Type = OpAssign<Lhs, Rhs>;
};

// 特化：当是冗余 cast 时，unwrap
template <typename Lhs, auto Mode, typename Reg, typename Src>
struct UnwrapOrNot<Lhs, OpCast<Mode, Reg, Src>, true> {
    using Type = OpAssign<Lhs, OpCopy<Src>>;
};

template <typename Node>
struct TryUnwrapRedundantCastInAssign {
    using Type = Node; // 默认不变
};

template <typename Lhs, typename Rhs>
struct TryUnwrapRedundantCastInAssign<OpAssign<Lhs, Rhs>> {
private:
    template <typename T>
    struct IsRedundantCast : std::false_type {};

    // 对 OpCast 的部分特化
    template <auto Mode, typename Reg, typename Src>
    struct IsRedundantCast<OpCast<Mode, Reg, Src>> {
        static constexpr bool value =
            (IsLocalVar_v<Src> || IsParam_v<Src>) && std::is_same_v<typename Src::Type, typename Lhs::Type>;
    };

    template <typename T>
    static constexpr bool is_redundant_cast_v = IsRedundantCast<T>::value;

    static constexpr bool is_redundant = is_redundant_cast_v<Rhs>;

public:
    using Type = typename UnwrapOrNot<Lhs, Rhs, is_redundant>::Type;
};

template <typename ExprList, typename ReplaceRangeList, std::size_t I = 0, typename Acc = TypeList<>, typename = void>
struct BuildOptimizedList;

template <typename ExprList, typename ReplaceRangeList, std::size_t I, typename Acc>
struct BuildOptimizedList<ExprList, ReplaceRangeList, I, Acc, std::enable_if_t<(I == Size_v<ExprList>)>> {
    using Type = Acc;
};

template <typename ExprList, typename ReplaceRangeList, std::size_t I, typename Acc>
struct BuildOptimizedList<ExprList, ReplaceRangeList, I, Acc, std::enable_if_t<(I < Size_v<ExprList>)>> {
    using CurrentNode = Get_t<ExprList, I>;

    // Step A: 先 unwrap 冗余 cast（如果适用）
    using UnwrappedNode = typename TryUnwrapRedundantCastInAssign<CurrentNode>::Type;

    // Step B: 再应用符号替换（如 item1 → item0，在后续表达式中）
    using ProcessedNode = typename ApplyReplacements<I, UnwrappedNode, ReplaceRangeList>::Type;

    using NewAcc = Append_t<Acc, ProcessedNode>;
    using Type = typename BuildOptimizedList<ExprList, ReplaceRangeList, I + 1, NewAcc>::Type;
};

template <typename ExprList>
__host_aicore__ constexpr decltype(auto) RemoveRedundantCast()
{
    using ReplaceItemList = typename decltype(ForEach(
        ExprList{}, RedundantCastFinder{},
        IndexedData{std::integral_constant<std::size_t, 0>{}, TypeList<>{}}))::DataType;

    if constexpr (IsEmpty_v<ReplaceItemList>) {
        return ExprList{};
    } else {
        using ReplaceRangeList = Map_t<SafeReplaceRange<ExprList>::template Mapper, ReplaceItemList>;

        using OptimizedNodeList = typename BuildOptimizedList<ExprList, ReplaceRangeList>::Type;

        return OptimizedNodeList{};
    }
}

} // namespace Atvoss::Graph

#endif // ATVOSS_EXPR_REMOVE_CAST_H