/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_EXPRESSION_EXPR_TEMPLATE_H
#define ATVOSS_EXPRESSION_EXPR_TEMPLATE_H

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "utils/utility.h"
#include "utils/patterns.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "kernel_basic_intf.h"
#endif

namespace Atvoss {

template <typename T>
struct OpCopy;

using Atvoss::Util::TypeList;

enum class ParamUsage
{
    IN,
    OUT,
    IN_OUT,
};
/*--------------------------------------------------------------------------------------------------------------------*/
/*!
 * Expression: Type for a basic "expression".  If T is trivial (as it should normally
 * be), Expression<T> is trivially copy-constructible and trivially move-
 * constructible, but is not trivially assignable.  Objects of the same
 * Expression type cannot be assigned, because the data member is const.
 * Objects of different Expression types can be assigned, but it will
 * result in a new Expression object (and is not a "normal" assignment).
 */
template <typename T>
struct Expression {
    static_assert(!std::is_rvalue_reference_v<T>, "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    using Type = T;
    using RetType = typename std::conditional_t<std::is_scalar_v<T>, Util::RetTypeWrapper<T>, T>::RetType;
    using TensorType = typename std::conditional_t<std::is_scalar_v<T>, Util::TensorTypeWrapper<T>, T>::TensorType;

    T const data{};

    template <typename U>
    [[nodiscard]] constexpr auto operator=(Expression<U> u);
};

// Let T deduce to value type or lvalue reference type
template <typename T>
Expression(T&& value) -> Expression<T>;

template <typename T>
using IsExpression = Util::IsSpecializationOf<Expression, T>;

template <typename T>
inline constexpr bool IsExpression_v = IsExpression<T>::value;
/*--------------------------------------------------------------------------------------------------------------------*/
template <std::size_t N, typename T, typename L = void>
struct LocalVar {
    static_assert(!std::is_reference_v<T>, "[ERROR]: [Atvoss][Expression] A LocalVar must not be a reference");
    using Type = T;
    using RetType = std::decay_t<T>;
    using TensorType = RetType;
    using Like = L;
    static constexpr std::size_t number = N;

    template <typename V>
    constexpr auto operator=(Expression<V>)
    {
        static_assert(
            Util::AlwaysFalse_v<V>, "[ERROR]: [Atvoss][Expression] Please use Expression<LocalVar> for assignment");
    }
};

template <typename T>
struct IsLocalVar : std::false_type {};

template <std::size_t N, typename T, typename L>
struct IsLocalVar<LocalVar<N, T, L>> : std::true_type {};

template <typename T>
inline constexpr bool IsLocalVar_v = IsLocalVar<T>::value;
/*--------------------------------------------------------------------------------------------------------------------*/
template <std::size_t N, typename T, ParamUsage U = ParamUsage::IN, std::size_t RN = N>
struct Param {
    using Type = T;
    using RetType = std::decay_t<T>;
    using TensorType = RetType;
    static constexpr std::size_t number = N;
    // for IN_OUT scenario. Two `Param`s point to the same GM
    static constexpr std::size_t inplaceNumber = RN;
    static constexpr ParamUsage usage = U;

    template <typename W>
    constexpr auto operator=(Expression<W>)
    {
        static_assert(
            Util::AlwaysFalse_v<W>, "[ERROR]: [Atvoss][Expression] Please use Expression<Param> for assignment");
    }
};

template <typename T>
struct IsParam : std::false_type {};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct IsParam<Param<N, T, U, R>> : std::true_type {};

template <typename T>
inline constexpr bool IsParam_v = IsParam<T>::value;
/*--------------------------------------------------------------------------------------------------------------------*/
template <std::size_t N>
struct CheckVarNum {
    template <typename T>
    struct Checker {
        static constexpr bool value = (T::number == N);
    };
};

template <typename T, typename = void>
struct IsUnaryOp : std::false_type {};

template <typename T>
struct IsUnaryOp<T, std::void_t<typename T::IsUnaryOp>> : std::true_type {};

template <typename T>
inline constexpr bool IsUnaryOp_v = IsUnaryOp<T>::value;

template <typename T, typename = void>
struct IsBinaryOp : std::false_type {};

template <typename T>
struct IsBinaryOp<T, std::void_t<typename T::IsBinaryOp>> : std::true_type {};

template <typename T>
inline constexpr bool IsBinaryOp_v = IsBinaryOp<T>::value;

namespace Detail {

template <typename T, typename = void>
struct LocalVarCollector {
    using Type = TypeList<>;
};

template <typename T>
struct LocalVarCollector<T, std::enable_if_t<IsLocalVar_v<T>>> {
    using Type = TypeList<T>;
};

template <typename T>
struct LocalVarCollector<T, std::enable_if_t<IsUnaryOp_v<T>>> {
    using Type = typename LocalVarCollector<typename T::DataType>::Type;
};

template <typename T>
struct LocalVarCollector<T, std::enable_if_t<IsBinaryOp_v<T>>> {
    using Type = Atvoss::Util::Concatenate_t<
        typename LocalVarCollector<typename T::LhsType>::Type, typename LocalVarCollector<typename T::RhsType>::Type>;
};

template <typename Head, typename... Tail>
struct LocalVarCollector<Util::TypeList<Head, Tail...>> {
    using Type = Atvoss::Util::Concatenate_t<
        typename LocalVarCollector<Head>::Type, typename LocalVarCollector<Util::TypeList<Tail...>>::Type>;
};

template <typename T>
struct UniqueLocalVars {
    using Type = Atvoss::Util::Unique_t<typename Detail::LocalVarCollector<T>::Type>;
};

template <typename T, typename = void>
struct ParamCollector {
    using Type = TypeList<>;
};

template <typename T>
struct ParamCollector<T, std::enable_if_t<IsParam_v<T>>> {
    using Type = TypeList<T>;
};

template <typename T>
struct ParamCollector<T, std::enable_if_t<IsUnaryOp_v<T>>> {
    using Type = typename ParamCollector<typename T::DataType>::Type;
};

template <typename T>
struct ParamCollector<T, std::enable_if_t<IsBinaryOp_v<T>>> {
    using Type = Atvoss::Util::Concatenate_t<
        typename ParamCollector<typename T::LhsType>::Type, typename ParamCollector<typename T::RhsType>::Type>;
};

template <typename Head, typename... Tail>
struct ParamCollector<Util::TypeList<Head, Tail...>> {
    using Type = Atvoss::Util::Concatenate_t<
        typename ParamCollector<Head>::Type, typename ParamCollector<Util::TypeList<Tail...>>::Type>;
};

template <typename T>
struct UniqueParams {
    using Type = Atvoss::Util::Unique_t<typename Detail::ParamCollector<T>::Type>;
};

template <typename UnsortedLst>
struct SortedParams {
    static constexpr std::size_t size = Atvoss::Util::Size_v<UnsortedLst>;

    template <std::size_t Number>
    struct GetByNumber {
        static constexpr std::size_t pos =
            Atvoss::Util::Find_v<Atvoss::CheckVarNum<Number>::template Checker, UnsortedLst>;
        using Type = Atvoss::Util::Get_t<UnsortedLst, pos>;
    };

    // Sort ...
    template <std::size_t... Numbers>
    static constexpr auto BuildSortedList(std::index_sequence<Numbers...>)
    {
        return Util::TypeList<typename GetByNumber<Numbers + 1>::Type...>{};
    }

public:
    using Type = decltype(BuildSortedList(std::make_index_sequence<size>{}));
};
} // namespace Detail

template <typename T, std::size_t N>
struct IsParamN : std::false_type {};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct IsParamN<Param<N, T, U, R>, N> : std::true_type {};

template <std::size_t N>
struct IsParamNChecker {
    template <typename T>
    using Type = IsParamN<T, N>;
};

template <typename T, std::size_t N>
struct IsLocalVarN : std::false_type {};

template <std::size_t N, typename T, typename L>
struct IsLocalVarN<LocalVar<N, T, L>, N> : std::true_type {};

template <std::size_t N>
struct IsLocalVarNChecker {
    template <typename T>
    using Type = IsLocalVarN<T, N>;
};

template <typename T, typename V, typename = void>
struct HasParamN {
    static constexpr bool value = false;
};

template <typename T, typename V>
struct HasParamN<T, V, std::enable_if_t<IsParam_v<V>>> {
private:
    using ParamsInT = typename Detail::UniqueParams<T>::Type;

public:
    static constexpr bool value =
        Atvoss::Util::Find_v<IsParamNChecker<V::number>::template Type, ParamsInT> != Atvoss::Util::Size_v<ParamsInT>;
};

template <typename T, typename V>
struct HasParamN<T, V, std::enable_if_t<IsLocalVar_v<V>>> {
private:
    using LocalVarsT = typename Detail::UniqueLocalVars<T>::Type;

public:
    static constexpr bool value = Atvoss::Util::Find_v<IsLocalVarNChecker<V::number>::template Type, LocalVarsT> !=
                                  Atvoss::Util::Size_v<LocalVarsT>;
};

template <typename V /*Param or LocalVar*/>
struct HasParamNChecker {
    template <typename T>
    using Type = HasParamN<T, V>;
};

template <typename T, typename = void>
struct HasUsage : std::false_type {};

template <typename T>
struct HasUsage<T, std::void_t<decltype(T::usage)>> : std::true_type {};

template <typename T>
struct LocalVars {
private:
    using UnsortedType = typename Detail::UniqueLocalVars<T>::Type;

    static constexpr std::size_t size = Atvoss::Util::Size_v<UnsortedType>;
    template <typename U>
    struct InRange : std::bool_constant<(U::number > 0 && U::number <= size)> {};
    static_assert(
        Atvoss::Util::All_v<InRange, UnsortedType>,
        "[ERROR]: [Atvoss][Expression] LocalVars must be numbered sequentially from 1");

public:
    using Type = typename Detail::SortedParams<UnsortedType>::Type;
};

template <typename T>
using LocalVars_t = typename LocalVars<T>::Type;

template <typename T>
struct Params {
private:
    using UnsortedType = typename Detail::UniqueParams<T>::Type;

    static constexpr std::size_t size = Atvoss::Util::Size_v<UnsortedType>;
    template <typename U>
    struct InRange : std::bool_constant<(U::number > 0 && U::number <= size)> {};
    static_assert(
        Atvoss::Util::All_v<InRange, UnsortedType>,
        "[ERROR]: [Atvoss][Expression] Params must be numbered sequentially from 1");

public:
    using Type = typename Detail::SortedParams<UnsortedType>::Type;
};

template <typename T>
using Params_t = typename Params<T>::Type;

template <typename U>
struct IsInVar : std::bool_constant<U::usage == ParamUsage::IN || U::usage == ParamUsage::IN_OUT> {};

template <typename U>
struct IsOutVar : std::bool_constant<U::usage == ParamUsage::OUT || U::usage == ParamUsage::IN_OUT> {};

template <typename U>
struct IsInplaceVar : std::bool_constant<U::usage == ParamUsage::IN_OUT> {};

template <typename T>
struct InParams {
    using Type = Atvoss::Util::Filter_t<IsInVar, Params_t<T>>;
};

template <typename T>
using InParams_t = typename InParams<T>::Type;

template <typename T>
struct OutParams {
    using Type = Atvoss::Util::Filter_t<IsOutVar, Params_t<T>>;
};

template <typename T>
using OutParams_t = typename OutParams<T>::Type;

template <typename T>
struct IsInParam : std::false_type {};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct IsInParam<Param<N, T, U, R>> : std::bool_constant<IsInVar<Param<N, T, U, R>>::value> {};

template <typename T>
struct IsOutParam : std::false_type {};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct IsOutParam<Param<N, T, U, R>> : std::bool_constant<IsOutVar<Param<N, T, U, R>>::value> {};

template <typename T>
struct IsInplaceParam : std::false_type {};

template <std::size_t N, typename T, ParamUsage U, std::size_t R>
struct IsInplaceParam<Param<N, T, U, R>> : std::bool_constant<IsInplaceVar<Param<N, T, U, R>>::value> {};

/*--------------------------------------------------------------------------------------------------------------------*/
template <typename T, typename R = typename std::decay_t<T>::RetType>
struct UnaryOp : private Util::CompressedData<T> {
private:
    using Storage = Util::CompressedData<T>;

public:
    static_assert(!std::is_rvalue_reference_v<T>, "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");

    using IsUnaryOp = void;
    using DataType = T;
    using TensorType = typename T::TensorType;
    using RetType = R;

    UnaryOp() = default;
    constexpr UnaryOp(T t) : Storage(t)
    {}

    constexpr const T& GetData() const
    {
        return Storage::Data();
    }
};
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename T, typename U, typename R = typename std::decay_t<T>::RetType>
struct BinaryOp : private Util::CompressedPair<T, U> {
private:
    using Storage = Util::CompressedPair<T, U>;

public:
    static_assert(
        !(std::is_rvalue_reference_v<T> || std::is_rvalue_reference_v<U>),
        "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");

    using IsBinaryOp = void;
    using LhsType = T;
    using RhsType = U;
    using TensorType = typename T::TensorType;
    using RetType = R;

    BinaryOp() = default;
    constexpr BinaryOp(T t, U u) : Storage(t, u)
    {}

    constexpr const T& GetLhs() const
    {
        return Storage::First();
    }
    constexpr const U& GetRhs() const
    {
        return Storage::Second();
    }
};
/*--------------------------------------------------------------------------------------------------------------------*/
template <typename T, typename U, typename V, typename R>
struct TernaryOp {
    static_assert(
        !(std::is_rvalue_reference_v<T> || std::is_rvalue_reference_v<U>),
        "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");

    using IsBinaryOp = void;
    using LhsType = T;
    using RhsType = U;
    using VhsType = V;
    using TensorType = typename T::TensorType;
    using RetType = std::decay_t<R>;
    T lhs;
    U rhs;
    V ths;
};

template <typename T, typename U>
struct OpAssign : BinaryOp<T, U> {
    OpAssign() = default;
    constexpr OpAssign(T t, U u) : BinaryOp<T, U>(t, u)
    {}
};

template <typename T>
struct IsOpAssign : std::false_type {};

template <typename T, typename U>
struct IsOpAssign<OpAssign<T, U>> : std::true_type {};

template <typename T>
inline constexpr bool IsOpAssign_v = IsOpAssign<T>::value;

template <typename T>
template <typename U>
__host_aicore__ constexpr auto Expression<T>::operator=(Expression<U> u)
{
    static_assert(
        (IsParam_v<T> || IsLocalVar_v<T> || std::is_lvalue_reference_v<T>),
        "[ERROR]: [Atvoss][Expression] Only a Param, LocalVar, or reference can appear on the left side "
        "of assignment");
    if constexpr (IsLocalVar_v<U> || IsParam_v<U>) {
        constexpr auto result = Atvoss::OpCopy<U>(u.data);
        return Expression<OpAssign<T, std::decay_t<decltype(result)>>>{{data, u.data}};
    } else {
        return Expression<OpAssign<T, U>>{{data, u.data}};
    }
}

template <typename T, typename U>
constexpr auto BindBuff(T buff, U expr)
{
    return buff = expr;
}

template <typename T, typename U>
struct OpAndThen : BinaryOp<T, U, typename U::RetType> {
    OpAndThen() = default;
    constexpr OpAndThen(T t, U u) : BinaryOp<T, U, typename U::RetType>(t, u)
    {}
};

namespace Detail {

template <typename Expr, typename List>
struct ToOpAndThenExprHelper;

template <typename T>
struct ToOpAndThenExprHelper<Expression<T>, TypeList<>> {
    using Type = Expression<T>;
};

template <typename T, typename U, typename... Rest>
struct ToOpAndThenExprHelper<Expression<T>, TypeList<U, Rest...>> {
    using Type = typename ToOpAndThenExprHelper<Expression<OpAndThen<T, U>>, TypeList<Rest...>>::Type;
};

}; // namespace Detail

template <typename List>
struct ToOpAndThenExpr;

template <typename T>
struct ToOpAndThenExpr<TypeList<T>> {
    using Type = Expression<T>;
};

template <typename T, typename U>
struct ToOpAndThenExpr<OpAndThen<T, U>> {
    using Type = Expression<OpAndThen<T, U>>;
};

template <typename T, typename U, typename... Rest>
struct ToOpAndThenExpr<TypeList<T, U, Rest...>> {
    using Type = typename Detail::ToOpAndThenExprHelper<Expression<T>, TypeList<U, Rest...>>::Type;
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator,(Expression<T> t, Expression<U> u)
{
    return Expression<OpAndThen<T, U>>{{t.data, u.data}};
}

template <typename T>
struct FlattenAtOpAndThen {
    using Type = TypeList<T>;
};

template <typename T, typename U>
struct FlattenAtOpAndThen<OpAndThen<T, U>> {
    using Type =
        Atvoss::Util::Concatenate_t<typename FlattenAtOpAndThen<T>::Type, typename FlattenAtOpAndThen<U>::Type>;
};

// declare unary op
#define DeclareUnaryOp(Name)                                    \
    template <typename T>                                       \
    struct Op##Name : UnaryOp<T> {                              \
        Op##Name() = default;                                   \
        constexpr Op##Name(T t) : UnaryOp<T>(t)                 \
        {}                                                      \
    };                                                          \
    template <typename T>                                       \
    __host_aicore__ constexpr auto Name(Expression<T> lhs)      \
    {                                                           \
        return Expression<Op##Name<T>>{{lhs.data}};             \
    }                                                           \
    template <typename T>                                       \
    __host_aicore__ constexpr auto Name(T&& lhs)                \
    {                                                           \
        return Expression<Op##Name<T>>{{std::forward<T>(lhs)}}; \
    }

// declare binary op
#define DeclareBinaryOp(Name)                                                 \
    template <typename T, typename U>                                         \
    struct Op##Name : BinaryOp<T, U> {                                        \
        Op##Name() = default;                                                 \
        constexpr Op##Name(T t, U u) : BinaryOp<T, U>(t, u)                   \
        {}                                                                    \
    };                                                                        \
    template <typename T, typename U>                                         \
    __host_aicore__ constexpr auto Name(Expression<T> lhs, Expression<U> rhs) \
    {                                                                         \
        return Expression<Op##Name<T, U>>{{lhs.data, rhs.data}};              \
    }                                                                         \
    template <typename T, typename U>                                         \
    __host_aicore__ constexpr auto Name(Expression<T> lhs, U&& rhs)           \
    {                                                                         \
        return Expression<Op##Name<T, U>>{{lhs.data, std::forward<U>(rhs)}};  \
    }                                                                         \
    template <typename T, typename U>                                         \
    __host_aicore__ constexpr auto Name(T&& lhs, Expression<U> rhs)           \
    {                                                                         \
        return Expression<Op##Name<T, U>>{{std::forward<T>(lhs), rhs.data}};  \
    }

template <std::size_t N, typename T = void, typename L>
__host_aicore__ constexpr auto PlaceHolderTmpLike(Expression<L> /*unused*/)
{
    static_assert(IsParam_v<L>, "[ERROR]: [Atvoss][Expression] A LocalVar can only be like a Param");
    if constexpr (std::is_void_v<T>) {
        return Expression<LocalVar<N, typename L::Type, L>>{};
    } else {
        return Expression<LocalVar<N, T, L>>{};
    }
}

template <std::size_t N, typename T, ParamUsage U = ParamUsage::IN>
__host_aicore__ constexpr auto PlaceHolder()
{
    return Expression<Param<N, T, U>>{};
}

template <typename Expr, typename List>
struct BuildExpressionHelper;

template <typename T>
struct BuildExpressionHelper<Expression<T>, TypeList<>> {
    using Type = Expression<T>;
};

template <typename T, typename U, typename... Rest>
struct BuildExpressionHelper<Expression<T>, TypeList<U, Rest...>> {
    using Type = typename BuildExpressionHelper<Expression<OpAndThen<T, U>>, TypeList<Rest...>>::Type;
};

template <typename List>
struct BuildExpression;

template <typename T>
struct BuildExpression<TypeList<T>> {
    using Type = Expression<T>;
};

template <typename T, typename U, typename... Rest>
struct BuildExpression<TypeList<T, U, Rest...>> {
    using Type = typename BuildExpressionHelper<Expression<T>, TypeList<U, Rest...>>::Type;
};

template <typename T>
using BuildExpression_t = typename BuildExpression<T>::Tye;

/*!
 * Maker: Base class to express that a class is a maker of calculation expression template
 */
class Maker {};

/*--------------------------------------------------------------------------------------------------------------------*/
template <typename T, typename To>
struct ReplaceTensorType;

template <template <typename, typename> class BlockTensor, typename From, typename Layout, typename To>
struct ReplaceTensorType<BlockTensor<From, Layout>, To> {
    using Type = BlockTensor<To, Layout>;
};

template <template <typename> class GlobalTensor, typename From, typename To>
struct ReplaceTensorType<GlobalTensor<From>, To> {
    using Type = GlobalTensor<To>;
};

} // namespace Atvoss

#endif // ATVOSS_EXPRESSION_EXPR_TEMPLATE_H