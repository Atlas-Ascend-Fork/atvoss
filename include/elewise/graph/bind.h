/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_GRAPH_BIND_H
#define ATVOSS_GRAPH_BIND_H

#include "utils/utility.h"
#include "expression/expr_template.h"
#include "operators/tensor_expression.h"

namespace Atvoss::Tile {

using namespace Atvoss;

using Atvoss::Util::Any;
using Atvoss::Util::Any_v;
using Atvoss::Util::Append_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Contains;
using Atvoss::Util::Contains_v;
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Find_v;
using Atvoss::Util::FindLast_v;
using Atvoss::Util::First_t;
using Atvoss::Util::Get_t;
using Atvoss::Util::IsSpecializationOf_v;
using Atvoss::Util::Map_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::TypeList;
using Atvoss::Util::Unique_t;

template <typename V /*Param or LocalVar*/, typename OpX /*OpCopyX or OpAdd or OpReduceSum ...*/>
struct Bind;

template <typename... Args>
struct OpPatternBase {
public:
    using OpArgs = TypeList<Args...>;
};

template <typename OpX>
struct OpPattern;

template <template <auto, typename...> class OpX, auto NoneType, typename... Args>
struct OpPattern<OpX<NoneType, Args...>> : OpPatternBase<Args...> {
public:
    using OpTpl = OpX<NoneType, Args...>;

    template <typename T>
    struct RebuildOp;

    template <typename... NewArgs>
    struct RebuildOp<TypeList<NewArgs...>> {
        using Type = OpX<NoneType, NewArgs...>;
    };
};

template <template <typename...> class OpX, typename... Args>
struct OpPattern<OpX<Args...>> : OpPatternBase<Args...> {
public:
    using OpTpl = OpX<Args...>;

    template <typename T>
    struct RebuildOp;

    template <typename... NewArgs>
    struct RebuildOp<TypeList<NewArgs...>> {
        using Type = OpX<NewArgs...>;
    };
};

struct ExtractDependOps {
    template <typename T, typename ResultList>
    __host_aicore__ constexpr auto operator()(T, ResultList) const
    {
        using B = typename T::Type;
        return Concatenate_t<ResultList, typename B::DependOps>{};
    }
};

template <typename T>
struct ExtractBindParams {
    using Type = TypeList<>;
};

template <typename Head, typename... Tail>
struct ExtractBindParams<TypeList<Head, Tail...>> {
    using Type = Concatenate_t<typename Head::AllParams, typename ExtractBindParams<TypeList<Tail...>>::Type>;
};

template <typename T>
using ExtractBindParams_t = typename ExtractBindParams<T>::Type;

template <typename B>
struct ExtractBindAssignTo {
    using Type = typename B::AssignedTo;
};

template <template <typename...> class Op, typename B>
struct IsBindOfOp : std::bool_constant<IsSpecializationOf_v<Op, typename B::Operation>> {};

template <template <typename...> class Op, typename B>
inline constexpr bool IsBindOfOp_v = IsBindOfOp<Op, B>::value;

template <template <typename...> class Op>
struct BindOpChecker {
    template <typename B>
    using Type = IsBindOfOp<Op, B>;
};

template <typename Source>
struct IsInputOf {
    template <typename Target>
    using Type = Contains<typename Target::InNonScalarOps, Source>;
};

template <typename ConnectTargetList>
struct ConnectToAny {
    template <typename B>
    using Type = Any<IsInputOf<B>::template Type, ConnectTargetList>;
};

template <typename ConnectTargetList, typename B>
inline constexpr bool ConnectToAny_v = Any_v<IsInputOf<B>::template Type, ConnectTargetList>;

template <typename B>
struct IsScalarBind : std::bool_constant<B::isScalarOp> {};

template <typename B>
inline constexpr bool IsScalarBind_v = IsScalarBind<B>::value;

// TODO
template <bool cache, typename B>
struct IsCacheBind
    : std::bool_constant<cache && \
        false/*IsBindOfOp_v<OpCopyInBrc, Head> ||
        IsBindOfOp_v<OpVecBrc, Head>*/> {};

template <bool cache, typename B>
inline constexpr bool IsCacheBind_v = IsCacheBind<cache, B>::value;

template <typename BindList, typename Param, typename = void>
struct LastAssignRhs {
    using Type = Param;
};

template <typename BindList, typename Param>
struct LastAssignRhs<
    BindList, Param,
    std::enable_if_t<IsLocalVar_v<Param> || (IsParam_v<Param> && !std::is_scalar_v<typename Param::Type>)>> {
private:
    /*Find CopyIn or OpAssign*/
    template <typename ToCheck>
    struct ParamEquals
        : std::bool_constant<!IsBindOfOp_v<OpCopyOut, ToCheck> && std::is_same_v<Param, typename ToCheck::AssignedTo>> {
    };
    static constexpr std::size_t lastPos = FindLast_v<ParamEquals, BindList>;

public:
    using Type = Get_t<BindList, lastPos>;
};

template <typename BindList, typename Param>
struct LastAssignRhs<BindList, Param, std::enable_if_t<(IsParam_v<Param> && std::is_scalar_v<typename Param::Type>)>> {
    using Type = Param;
};

template <typename BindList>
struct LastAssignRhsFinder {
    template <typename Param>
    using Type = LastAssignRhs<BindList, Param>;
};

template <typename Bind2ParamMap, typename B, typename = void>
struct FindAssignedTo {
private:
    template <typename ToCheck>
    struct BindEquals : std::bool_constant<std::is_same_v<B, First_t<ToCheck>>> {};
    static constexpr std::size_t pos = Find_v<BindEquals, Bind2ParamMap>;
    static_assert(pos < Size_v<Bind2ParamMap>, "BUG");
    using B2P = Get_t<Bind2ParamMap, pos>;

public:
    using Type = Get_t<B2P, 1>;
};

template <typename Bind2ParamMap, typename B>
struct FindAssignedTo<Bind2ParamMap, B, std::enable_if_t<!IsSpecializationOf_v<Bind, B>>> {
    using Type = B;
};

template <typename Bind2ParamMap>
struct BindAssignedToFinder {
    template <typename B>
    using Type = FindAssignedTo<Bind2ParamMap, B>;
};

template <typename OpX, template <typename> class Proc>
struct ReplaceOpArgs {
private:
    using Pattern = OpPattern<OpX>;
    using OpArgs = typename Pattern::OpArgs;
    using NewArgs = Map_t<Proc, OpArgs>;

public:
    using Type = typename Pattern::template RebuildOp<NewArgs>::Type;
};

template <typename OpX, template <typename> class Proc>
using ReplaceOpArgs_t = typename ReplaceOpArgs<OpX, Proc>::Type;

// Check whether @ToCheckOp can be free
// (not used since @start in @OpList any more)
template <typename OpLst, typename ToCheckOp, bool cacheBrc, std::size_t start>
__host_aicore__ constexpr bool IsAbleToFree()
{
    if constexpr (IsCacheBind_v<cacheBrc, ToCheckOp>) {
        return false;
    } else if constexpr (start < Size_v<OpLst>) {
        using Op = Get_t<OpLst, start>;
        using InputOps = typename Op::InNonScalarOps;
        if constexpr (Contains_v<InputOps, ToCheckOp>) {
            return false;
        }
        return IsAbleToFree<OpLst, ToCheckOp, cacheBrc, start + 1>();
    } else {
        return true;
    }
}

template <typename V /*Param or LocalVar*/, typename OpX /*OpCopyX or OpAdd or OpReduceSum ...*/>
struct Bind {
private:
    using Pattern = OpPattern<OpX>;

public:
    // Param<> or LocalVar<>
    using AssignedTo = V;
    // OpX<...>
    using Operation = typename Pattern::OpTpl;
    using BindType = Bind<V, Operation>;
    using TensorType = typename Operation::TensorType;
    using RetType = typename Operation::RetType;
    // Arguments except values.
    // Exp: in OpReduceSum<AR, ...>, AR will be removed.
    using OpArgs = typename Pattern::OpArgs;
    // Input & Outpu GM Params in OpArgs
    using AllParams = Unique_t<Filter_t<IsParam, OpArgs>>;
    // Input GM Params in OpArgs
    using InParams = Filter_t<IsInParam, AllParams>;
    // Output GM Params in OpArgs
    using OutParams = Filter_t<IsOutParam, AllParams>;

private:
    // InOps in OpArgs by filter In/Out GM & scalar type like float/bool/int/enum...
    using InOpsTmp1 = Unique_t<Difference_t<OpArgs, AllParams>>;
    using InOpsTmp2 = Filter_t<std::is_scalar, InOpsTmp1>;

public:
    using InOps = Difference_t<InOpsTmp1, InOpsTmp2>;
    // TODO: InScalarParams
    // TODO: InScalarOps & InNonScalarOps
    using InScalarOps = Filter_t<IsScalarBind, InOps>;
    using InNonScalarOps = Difference_t<InOps, InScalarOps>;
    // dependencies & self
    using DependOps = Append_t<Unique_t<decltype(ForEach(InOps{}, ExtractDependOps{}, TypeList<>{}))>, BindType>;
    // TODO
    constexpr static bool isScalarOp = false;
    constexpr static bool isCopyXOp =
        (IsSpecializationOf_v<OpCopyIn, Operation> || IsSpecializationOf_v<OpCopyOut, Operation>);
    constexpr static bool isCopyInOp = IsSpecializationOf_v<OpCopyIn, Operation>;
};

} // namespace Atvoss::Tile

#endif // ATVOSS_GRAPH_BIND_H