/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_COMMON_H
#define Atvoss_DEV_COMMON_H
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "utility.h"
#include "../patterns.h"

#if __has_include(<type_traits>) || __has_include(<tuple>) 
#else 
namespace std = AscendC::std;
#endif 
namespace Atvoss {
    enum class ParamUsage {
        in,
        out,
        in_out,
    };
}

namespace Atvoss::ExprTmpl {

template <typename T, typename = void>
struct HasDataTrait {
    static constexpr bool value = true;
};
template <typename T>
struct HasDataTrait<T, std::void_t<decltype(T::hasData)>> {
    static constexpr bool value = T::hasData;
};
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
    static constexpr bool hasData = HasDataTrait<T>::value;

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

template <std::size_t N, typename T, typename L = void>
struct LocalVar {
    static_assert(!std::is_reference_v<T>,  "[ERROR]: [Atvoss][Expression] A LocalVar must not be a reference");
    using Type = T;
    using Like = L;
    using layout = typename L::layout;
    static constexpr std::size_t number = N;
    static constexpr bool hasData = false;

    template <typename V>
    constexpr auto operator=(Expression<V>) {
        static_assert(Util::AlwaysFalse_v<V>, "[ERROR]: [Atvoss][Expression] Please use Expression<LocalVar> for assignment");
    }
};

template <typename T>
struct IsLocalVar : std::false_type {};

template <std::size_t N, typename T, typename L>
struct IsLocalVar<LocalVar<N, T, L>> : std::true_type {};

template <typename T>
inline constexpr bool IsLocalVar_v = IsLocalVar<T>::value;

template <std::size_t N, typename T, typename U, ParamUsage V = ParamUsage::in>
struct Param {
    using Type = T;
    static constexpr std::size_t number = N;
    using layout = U;
    static constexpr ParamUsage usage = V;
    static constexpr bool hasData = false;

    template <typename W>
    constexpr auto operator=(Expression<W>) {
        static_assert(Util::AlwaysFalse_v<W>, "[ERROR]: [Atvoss][Expression] Please use Expression<Param> for assignment");
    }
};

template <typename T>
struct IsParam : std::false_type {};

template <std::size_t N, typename T, typename U, ParamUsage V>
struct IsParam<Param<N, T, U, V>> : std::true_type {};

template <typename T>
inline constexpr bool IsParam_v = IsParam<T>::value;

template <std::size_t N, typename T>
__host_aicore__ constexpr auto DefineLocalVar(){
    return Expression<LocalVar<N, T>>{};
}

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

template <typename T, bool Compress>
struct CompressedDataStorage;

template <typename T>
struct UnaryOp : private Util::CompressedData<T> {
private:
    using Storage = Util::CompressedData<T>;

public:
    static_assert(!std::is_rvalue_reference_v<T>,
                  "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    static constexpr bool hasData = HasDataTrait<T>::value;
    using IsUnaryOp = void;
    using DataType = T;

    UnaryOp() = default;
    constexpr UnaryOp(T t) : Storage(t) {}

    constexpr const T& GetData() const { return Storage::Data(); }
};

template <typename T, typename U>
struct BinaryOp : private Util::CompressedPair<T, U> {
private:
    using Storage = Util::CompressedPair<T, U>;

public:
    static_assert(!(std::is_rvalue_reference_v<T> ||
                    std::is_rvalue_reference_v<U>),
                  "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    static constexpr bool hasData =
            HasDataTrait<T>::value || HasDataTrait<U>::value;
    using IsBinaryOp = void;
    using LhsType = T;
    using RhsType = U;

    BinaryOp() = default;
    constexpr BinaryOp(T t, U u) : Storage(t, u) {}

    constexpr const T& GetLhs() const { return Storage::First(); }
    constexpr const U& GetRhs() const { return Storage::Second(); }
};

template <typename T, typename U, typename V>
struct TernaryOp {
    static_assert(!(std::is_rvalue_reference_v<T> ||
                    std::is_rvalue_reference_v<U>),
                  "[ERROR]: [Atvoss][Expression] Rvalue references cannot be stored");
    static constexpr bool hasData = HasDataTrait<T>::value || HasDataTrait<U>::value  || HasDataTrait<V>::value;
    using IsBinaryOp = void;
    using LhsType = T;
    using RhsType = U;
    using VhsType = V;
    T lhs;
    U rhs;
    V ths;
};

using Util::TMP::TypeList;

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
        using Type = Util::TMP::Concatenate_t<
                typename LocalVarCollector<typename T::LhsType>::Type,
                typename LocalVarCollector<typename T::RhsType>::Type>;
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
        using Type = Util::TMP::Concatenate_t<
                typename ParamCollector<typename T::LhsType>::Type,
                typename ParamCollector<typename T::RhsType>::Type>;
    };

    template <typename T>
    struct UniqueParams {
        using Type =
                Util::TMP::Unique_t<typename Detail::ParamCollector<T>::Type>;
    };

} // namespace Detail

template <typename T>
struct LocalVars {
    using Type =  Util::TMP::Unique_t<typename Detail::LocalVarCollector<T>::Type>;

private:
    static constexpr std::size_t size = Util::TMP::Size_v<Type>;
    template <typename U>
    struct InRange  : std::bool_constant<(U::number > 0 && U::number <= size)> {};
    static_assert(Util::TMP::Check_v<InRange, Type>,
                  "[ERROR]: [Atvoss][Expression] LocalVars must be numbered sequentially from 1");
};

template <typename T>
using LocalVars_t = typename LocalVars<T>::Type;

template <typename T>
struct Params {
    using Type = typename Detail::UniqueParams<T>::Type;

private:
    static constexpr std::size_t size = Util::TMP::Size_v<Type>;

    template <typename U>
    struct InRange : std::bool_constant<(U::number > 0 && U::number <= size)> {};
    static_assert(Util::TMP::Check_v<InRange, Type>,
                  "[ERROR]: [Atvoss][Expression] Params must be numbered sequentially from 1");
};

template <typename T>
using Params_t = typename Params<T>::Type;

template <typename T>
struct InParams {
    template <typename U>
    struct IsInVar : std::bool_constant<U::usage == ParamUsage::in ||
                                        U::usage == ParamUsage::in_out> {};
    using Type = Util::TMP::Filter_t<IsInVar, Params_t<T>>;
};

template <typename T>
using InParams_t = typename InParams<T>::Type;

template <typename T>
struct OutParams {
    template <typename U>
    struct IsOutVar : std::bool_constant<U::usage == ParamUsage::out ||
                                         U::usage == ParamUsage::in_out> {};
    using Type = Util::TMP::Filter_t<IsOutVar, Params_t<T>>;
};

template <typename T>
using OutParams_t = typename OutParams<T>::Type;

template <typename T, typename U>
struct OpAssign : BinaryOp<T, U> {
    OpAssign() = default;
    constexpr OpAssign(T t, U u) : BinaryOp<T, U>(t, u) {}
};

template <typename T>
template <typename U>
__host_aicore__ constexpr auto Expression<T>::operator=(Expression<U> u)
{
    static_assert(
            (IsParam_v<T> || IsLocalVar_v<T> || std::is_lvalue_reference_v<T>),
            "[ERROR]: [Atvoss][Expression] Only a Param, LocalVar, or reference can appear on the left side "
            "of assignment");
    return Expression<OpAssign<T, U>>{{data, u.data}};
}

template <typename T, typename U>
struct OpAndThen : BinaryOp<T, U> {
    OpAndThen() = default;
    constexpr OpAndThen(T t, U u) : BinaryOp<T, U>(t, u) {}
};

template <typename T, typename U>
__host_aicore__ constexpr auto operator,(Expression<T> t, Expression<U> u)
{
    return Expression<OpAndThen<T, U>>{{t.data, u.data}};
}
/*! 
 * Maker: Base class to express that a class is a maker of calculation expression template 
 */
class Maker {};

} // namespace Atvoss::ExprTmpl

namespace Atvoss{
    template <std::size_t N, typename L>
    __host_aicore__ constexpr auto PlaceHolderTmpLike(ExprTmpl::Expression<L> /*unused*/) {
        static_assert(ExprTmpl::IsParam_v<L>, "[ERROR]: [Atvoss][Expression] A LocalVar can only be like a Param");
        return ExprTmpl::Expression<ExprTmpl::LocalVar<N, typename L::Type, L>>{};
    }

    template <std::size_t N, typename T,  ParamUsage V = ParamUsage::in, typename U= AscendC::Std::tuple<>>
    __host_aicore__ constexpr auto PlaceHolder()
    {
        return ExprTmpl::Expression<ExprTmpl::Param<N, T, U, V>>{};
    }
};

#endif //Atvoss_DEV_COMMON_H
