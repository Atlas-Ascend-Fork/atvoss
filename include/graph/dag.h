/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_GRAPH_DAG_H
#define ATVOSS_GRAPH_DAG_H

#include "graph/bind.h"
#include "graph/node.h"
#include "graph/buffer.h"

namespace Atvoss::Tile {

using Atvoss::Util::Append_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Contains_v;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Find_v;
using Atvoss::Util::FindLast_v;
using Atvoss::Util::First_t;
using Atvoss::Util::GetLastN_t;
using Atvoss::Util::IsSpecializationOf_v;
using Atvoss::Util::Prepend_t;
using Atvoss::Util::Reverse_t;
using Atvoss::Util::Set_t;
using Atvoss::Util::TypeList;
using Atvoss::Util::TypeWrapper;
using Atvoss::Util::UpdateOrPrepend_t;

constexpr static std::size_t MAX_BUFFER_NUMBER = 10;

template <typename ExprList, typename Param>
struct FirstAndLastUse {
    static constexpr std::size_t firstUse = Find_v<HasParamNChecker<Param>::template Type, ExprList>;
    static constexpr std::size_t lastUse = FindLast_v<HasParamNChecker<Param>::template Type, ExprList>;
    using Type =
        TypeList<Param, std::integral_constant<std::size_t, firstUse>, std::integral_constant<std::size_t, lastUse>>;
};

template <typename ExprList>
struct FirstAndLastUseFinder {
    template <typename Param>
    using Type = FirstAndLastUse<ExprList, Param>;
};

struct CopyInInserter {
    template <typename T, typename ExprList>
    __host_aicore__ constexpr auto operator()(T, ExprList) const
    {
        using ParamUse = typename T::Type;
        using Param = Get_t<ParamUse, 0>;
        constexpr auto firstUse = Get_t<ParamUse, 1>::value;
        if constexpr (
            !std::is_scalar_v<typename Param::Type> &&
            (Param::usage == ParamUsage::IN || Param::usage == ParamUsage::IN_OUT)) {
            using OldItem = Get_t<ExprList, firstUse>;
            using NewItem = OpAndThen<OpCopyIn<Param>, OldItem>;
            return Set_t<ExprList, firstUse, NewItem>{};
        } else {
            return ExprList{};
        }
    }
};

struct CopyOutInserter {
    template <typename T, typename ExprList>
    __host_aicore__ constexpr auto operator()(T, ExprList) const
    {
        using ParamUse = typename T::Type;
        using Param = Get_t<ParamUse, 0>;
        constexpr auto lastUse = Get_t<ParamUse, 2>::value;
        if constexpr (Param::usage == ParamUsage::OUT || Param::usage == ParamUsage::IN_OUT) {
            static_assert(!std::is_scalar_v<typename Param::Type>, "Scalar Tensor is only supported in ParamUsage::IN");
            using OldItem = Get_t<ExprList, lastUse>;
            using NewItem = OpAndThen<OldItem, OpCopyOut<Param>>;
            return Set_t<ExprList, lastUse, NewItem>{};
        } else {
            return ExprList{};
        }
    }
};

struct AddCopyX {
    template <typename Param, typename ExprList>
    __host_aicore__ constexpr auto operator()(TypeWrapper<Param>, ExprList) const
    {
        if constexpr (std::is_scalar_v<typename Param::Type>) {
            static_assert(Param::usage == ParamUsage::IN, "Scalar Tensor is only supported in ParamUsage::IN");
            return ExprList{};
        } else if constexpr (Param::usage == ParamUsage::IN) {
            return Concatenate_t<TypeList<OpCopyIn<Param>>, ExprList>{};
        } else if constexpr (Param::usage == ParamUsage::OUT) {
            return Concatenate_t<ExprList, TypeList<OpCopyOut<Param>>>{};
        } else { // in_out
            return Concatenate_t<TypeList<OpCopyIn<Param>>, ExprList, TypeList<OpCopyOut<Param>>>{};
        }
    }
};

struct OpAssign2Bind {
public:
    template <typename T, typename BindList>
    __host_aicore__ constexpr auto operator()(TypeWrapper<T>, BindList) const
    {
        if constexpr (IsSpecializationOf_v<OpCopyIn, T>) {
            using CopyInParam = typename T::DataType;
            using Bi = Bind<CopyInParam, T>;
            return Append_t<BindList, Bi>{};
        } else if constexpr (IsSpecializationOf_v<OpCopyOut, T>) {
            using CopyOutParam = typename T::DataType;
            using RealOpX = ReplaceOpArgs_t<T, LastAssignRhsFinder<BindList>::template Type>;
            using Bo = Bind<CopyOutParam, RealOpX>;
            return Append_t<BindList, Bo>{};
        } else { // OpAssign
            using AssignedTo = typename T::LhsType;
            using OpX = typename T::RhsType; // OpAdd / OpReduceSum ...
            using RealOpX = ReplaceOpArgs_t<OpX, LastAssignRhsFinder<BindList>::template Type>;
            using Bs = Bind<AssignedTo, RealOpX>;
            return Append_t<BindList, Bs>{};
        }
    }
};

// Extract assigned Params from `OpAssign` Expression with `Usage=U`
template <ParamUsage U>
struct ExtractAssignedParams {
public:
    template <typename T, typename ResultList>
    __host_aicore__ constexpr auto operator()(TypeWrapper<T>, ResultList) const
    {
        using AssignedTo = typename T::LhsType;
        if constexpr (IsParam_v<AssignedTo>) {
            if constexpr (AssignedTo::usage == U) {
                return Append_t<ResultList, AssignedTo>{};
            } else {
                return ResultList{};
            }
        } else {
            return ResultList{};
        }
    }
};

// Create new params with Usage=Out for inplace params.
template <typename InplaceParams, std::size_t nextParamNum, typename ParamsMap = TypeList<>, std::size_t start = 0>
__host_aicore__ constexpr decltype(auto) CreateReplacerForInplaceParams()
{
    if constexpr (start < Size_v<InplaceParams>) {
        using OldParams = Get_t<InplaceParams, start>;
        using NewParams = Param<nextParamNum, typename OldParams::Type, ParamUsage::OUT, OldParams::number>;
        using ParamsMapNext = Append_t<ParamsMap, TypeList<OldParams, NewParams>>;
        return CreateReplacerForInplaceParams<InplaceParams, nextParamNum + 1, ParamsMapNext, start + 1>();
    } else {
        return ParamsMap{};
    }
}

template <typename ToFind>
struct InplaceParamEqual {
    template <typename M>
    using Type = std::is_same<First_t<M>, ToFind>;
};

template <typename InplaceParamsReplaceMap, typename InplaceParams, typename Arg, typename = void>
struct FindInplaceParamReplacer {
    using Type = Arg;
};

template <typename InplaceParamsReplaceMap, typename InplaceParams, typename Arg>
struct FindInplaceParamReplacer<
    InplaceParamsReplaceMap, InplaceParams, Arg, std::enable_if_t<IsParam_v<Arg> && Contains_v<InplaceParams, Arg>>> {
private:
    using R = Filter_t<InplaceParamEqual<Arg>::template Type, InplaceParamsReplaceMap>;
    static_assert(Size_v<R> == 1, "BUG");

public:
    using Type = Get_t<First_t<R>, 1>;
};

template <typename InplaceParamsReplaceMap, typename InplaceParams>
struct InplaceParamReplaceFinder {
    template <typename Arg>
    using Type = FindInplaceParamReplacer<InplaceParamsReplaceMap, InplaceParams, Arg>;
};

// Replace Params with Usage=InOut to new Params with Usage=Out.
template <
    typename InplaceParams, typename ReplaceMap, typename ExprList, std::size_t nextParamNum,
    std::size_t pos = Size_v<ExprList> - 1>
__host_aicore__ constexpr decltype(auto) ReplaceInplaceParam()
{
    if constexpr (Size_v<InplaceParams> == 0) {
        return ExprList{};
    } else {
        using AssignExpr = Get_t<ExprList, pos>;
        using AssignLhs = typename AssignExpr::LhsType;
        using AssignRhs = typename AssignExpr::RhsType;
        if constexpr (Contains_v<InplaceParams, AssignLhs>) {
            using Pm = Filter_t<InplaceParamEqual<AssignLhs>::template Type, ReplaceMap>;
            static_assert(Size_v<Pm> == 1, "BUG");
            using NewLhs = Get_t<First_t<Pm>, 1>;
            using InplaceParamsNext = Difference_t<InplaceParams, TypeList<AssignLhs>>;
            using ReplaceMapNext = Difference_t<ReplaceMap, Pm>;
            using NewRhs =
                ReplaceOpArgs_t<AssignRhs, InplaceParamReplaceFinder<ReplaceMapNext, InplaceParamsNext>::template Type>;
            using NewAssign = OpAssign<NewLhs, NewRhs>;
            using ExprListNext = Set_t<ExprList, pos, NewAssign>;
            return ReplaceInplaceParam<InplaceParamsNext, ReplaceMapNext, ExprListNext, nextParamNum + 1, pos - 1>();
        } else {
            using NewRhs =
                ReplaceOpArgs_t<AssignRhs, InplaceParamReplaceFinder<ReplaceMap, InplaceParams>::template Type>;
            using NewAssign = OpAssign<AssignLhs, NewRhs>;
            using ExprListNext = Set_t<ExprList, pos, NewAssign>;
            return ReplaceInplaceParam<InplaceParams, ReplaceMap, ExprListNext, nextParamNum + 1, pos - 1>();
        }
    }
}

template <typename Arg, typename = void>
struct ModifyInplaceParamUsage {
    using Type = Arg;
};

template <typename Arg>
struct ModifyInplaceParamUsage<Arg, std::enable_if_t<IsInplaceParam<Arg>::value>> {
    using Type = Param<Arg::number, typename Arg::Type, ParamUsage::IN>;
};

struct InplaceParamUsageEditer {
    template <typename Arg>
    using Type = ModifyInplaceParamUsage<Arg>;
};

// Change Params with Usage=InOut to Usage=In
template <typename InplaceParams, typename ExprList, std::size_t pos = 0>
__host_aicore__ constexpr decltype(auto) ChangeInplaceParamUsageToIn()
{
    if constexpr (Size_v<InplaceParams> == 0) {
        return ExprList{};
    } else if constexpr (pos < Size_v<ExprList>) {
        using AssignExpr = Get_t<ExprList, pos>;
        using Lhs = typename AssignExpr::LhsType;
        using Rhs = typename AssignExpr::RhsType;
        using NewLhs =
            std::conditional_t<IsInplaceParam<Lhs>::value, Param<Lhs::number, typename Lhs::Type, ParamUsage::IN>, Lhs>;
        using NewRhs = ReplaceOpArgs_t<Rhs, InplaceParamUsageEditer::Type>;
        using NewAssign = OpAssign<NewLhs, NewRhs>;
        using ExprListNext = Set_t<ExprList, pos, NewAssign>;
        return ChangeInplaceParamUsageToIn<InplaceParams, ExprListNext, pos + 1>();
    } else {
        return ExprList{};
    }
}

template <typename ExprList>
struct InplaceParamsProcessor {
private:
    using OriAllParams = Params_t<ExprList>;
    using InplaceParams = Filter_t<IsInplaceVar, OriAllParams>;
    using AssignedInplaceParams =
        Unique_t<decltype(ForEach(ExprList{}, ExtractAssignedParams<ParamUsage::IN_OUT>{}, TypeList<>{}))>;
    using ReplaceParamsMap =
        decltype(CreateReplacerForInplaceParams<AssignedInplaceParams, Size_v<OriAllParams> + 1>());
    using NewExprListTmp =
        decltype(ReplaceInplaceParam<AssignedInplaceParams, ReplaceParamsMap, ExprList, Size_v<OriAllParams> + 1>());

public:
    using Type = decltype(ChangeInplaceParamUsageToIn<InplaceParams, NewExprListTmp>());
};

template <int32_t targetBufId>
struct BufId2ParamFinder {
    template <typename M>
    struct Apply : std::bool_constant<First_t<M>::value == targetBufId> {};
};

template <typename BufId2ParamMap, int32_t combinedBufId, typename Param>
struct UpdateBufId2Param {
private:
    constexpr static auto bufIds = DecodeBufferId<combinedBufId>::Value;
    constexpr static auto outBufId = bufIds[0];
    constexpr static auto pos = Find_v<BufId2ParamFinder<outBufId>::template Apply, BufId2ParamMap>;
    using NewItem = TypeList<std::integral_constant<int32_t, outBufId>, Param>;

public:
    using Type = UpdateOrPrepend_t<BufId2ParamMap, pos, NewItem>;
};

template <typename T, typename = void>
struct GetTypeOrLike {
    using Type = T;
};

template <typename T>
struct GetTypeOrLike<T, std::enable_if_t<IsLocalVar_v<T>>> {
    using Type = typename T::Like;
};

template <typename T>
using GetTypeOrLike_t = typename GetTypeOrLike<T>::Type;

template <
    int32_t combinedBufId, typename CurrentBind, typename OutBindLst, typename BufId2ParamMap, std::size_t localVarIdx>
__host_aicore__ constexpr decltype(auto) ReuseOrCreateLocalVar()
{
    // 1. if connect to output, use ParamOut.
    // 2. if bufId is reused, use corresponding Param or LocalVar
    // 3. Create a new LocalVar
    constexpr auto outPos = Find_v<IsInputOf<CurrentBind>::template Type, OutBindLst>;
    if constexpr (outPos < Size_v<OutBindLst>) {
        using CopyOutBind = Get_t<OutBindLst, outPos>;
        using ParamX = typename CopyOutBind::AssignedTo;
        using BufId2ParamMapNext = typename UpdateBufId2Param<BufId2ParamMap, combinedBufId, ParamX>::Type;
        return TypeList<
            ParamX, BufId2ParamMapNext, std::bool_constant<true>, std::integral_constant<std::size_t, localVarIdx>>{};
    } else {
        constexpr auto bufIds = DecodeBufferId<combinedBufId>::Value;
        constexpr auto outBufId = bufIds[0];
        constexpr auto pos = Find_v<BufId2ParamFinder<outBufId>::template Apply, BufId2ParamMap>;
        if constexpr (pos < Size_v<BufId2ParamMap>) {
            using ParamX = Get_t<Get_t<BufId2ParamMap, pos>, 1>;
            return TypeList<
                ParamX, BufId2ParamMap, std::bool_constant<false>, std::integral_constant<std::size_t, localVarIdx>>{};
        } else {
            using OriParamX = typename CurrentBind::AssignedTo;
            using Type = typename OriParamX::Type;
            using Like = GetTypeOrLike_t<OriParamX>;
            using NewLocalVar = LocalVar<localVarIdx, Type, Like>;
            using BufId2ParamMapNext = typename UpdateBufId2Param<BufId2ParamMap, outBufId, NewLocalVar>::Type;
            return TypeList<
                NewLocalVar, BufId2ParamMapNext, std::bool_constant<true>,
                std::integral_constant<std::size_t, localVarIdx + 1>>{};
        }
    }
};

template <
    const int32_t* const* bufIds, typename BindLst, typename OutBindLst,
    /* Needs to update during each loop below */
    typename AssignLst = TypeList<>,                                         /* return */
    typename Bind2ParamMap = TypeList<>, typename Param2BufMap = TypeList<>, /* return */
    typename BufId2ParamMap = TypeList<>, std::size_t pos = 0, std::size_t localVarIdx = 1>
__host_aicore__ constexpr decltype(auto) Bind2OpAssign()
{
    if constexpr (pos < Size_v<BindLst>) {
        using B = Get_t<BindLst, pos>;
        using ParamX = typename B::AssignedTo;
        using Operation = typename B::Operation;
        constexpr auto pingId = bufIds[0][pos];
        constexpr auto pongId = bufIds[1][pos];
        if constexpr (B::isCopyXOp) {
            if constexpr (B::isCopyInOp) {
                using AssignLstNext = Append_t<AssignLst, Operation>;
                using Bind2ParamMapNext = Prepend_t<TypeList<B, ParamX>, Bind2ParamMap>;
                constexpr auto paramIdx = ParamX::number;
                using Param2BufMapNext =
                    Append_t<Param2BufMap, ParamBufIdMap<paramIdx, BufType::PARAM, pingId, pongId>>;
                using BufId2ParamMapNext = typename UpdateBufId2Param<BufId2ParamMap, pingId, ParamX>::Type;
                return Bind2OpAssign<
                    bufIds, BindLst, OutBindLst, AssignLstNext, Bind2ParamMapNext, Param2BufMapNext, BufId2ParamMapNext,
                    pos + 1, localVarIdx>();
            } else { // CopyOut
                using NewCopyOut = ReplaceOpArgs_t<Operation, BindAssignedToFinder<Bind2ParamMap>::template Type>;
                using AssignLstNext = Append_t<AssignLst, NewCopyOut>;
                return Bind2OpAssign<
                    bufIds, BindLst, OutBindLst, AssignLstNext, Bind2ParamMap, Param2BufMap, BufId2ParamMap, pos + 1,
                    localVarIdx>();
            }
        } else { // OpAssign
            using Result = decltype(ReuseOrCreateLocalVar<pingId, B, OutBindLst, BufId2ParamMap, localVarIdx>());
            using AssignLhs = Get_t<Result, 0>;
            using BufId2ParamMapNext = Get_t<Result, 1>;
            constexpr auto updateParam2BufMap = Get_t<Result, 2>::value;
            constexpr auto localVarIdxNext = Get_t<Result, 3>::value;
            using AssignRhs = ReplaceOpArgs_t<Operation, BindAssignedToFinder<Bind2ParamMap>::template Type>;
            using AssignLstNext = Append_t<AssignLst, OpAssign<AssignLhs, AssignRhs>>;
            using Bind2ParamMapNext = Prepend_t<TypeList<B, AssignLhs>, Bind2ParamMap>;
            if constexpr (updateParam2BufMap) {
                using Param2BufMapNext = Append_t<
                    Param2BufMap,
                    ParamBufIdMap<
                        AssignLhs::number, IsParam_v<AssignLhs> ? BufType::PARAM : BufType::LOCAL_VAR, pingId, pongId>>;
                return Bind2OpAssign<
                    bufIds, BindLst, OutBindLst, AssignLstNext, Bind2ParamMapNext, Param2BufMapNext, BufId2ParamMapNext,
                    pos + 1, localVarIdxNext>();
            } else {
                return Bind2OpAssign<
                    bufIds, BindLst, OutBindLst, AssignLstNext, Bind2ParamMapNext, Param2BufMap, BufId2ParamMapNext,
                    pos + 1, localVarIdxNext>();
            }
        }
    } else {
        return TypeList<AssignLst, Param2BufMap>{};
    }
}

template <typename ExprList /*TypeList*/>
struct DagBase {
public:
    // Collect all parameters used (Usage = In/InOut/Out)
    // REMEMBER:
    //  Size_v<AllParams> <= Size_v<InParams> + Size_v<OutParams>
    using AllParams = Params_t<ExprList>;
    // Coolect CopyIn parameters (Usage = In/InOut)
    using InParams = Filter_t<IsInVar, AllParams>;
    // Collect CopyOut parameters (Usage = InOut/Out)
    using OutParams = Filter_t<IsOutVar, AllParams>;
};

template <typename ExprList /*TypeList*/>
struct ManualDag : DagBase<ExprList> {
private:
    using Base = DagBase<ExprList>;

public:
    using AllLocalVars = LocalVars_t<ExprList>;
    // TypeList of ParamBufIdMap
    using BufMap = typename GenerateBufferId<Size_v<typename Base::AllParams>, Size_v<AllLocalVars>>::Type;
    // Get a type list of type lists of three items:
    //      (Param, FirstUseConst, LastUseConst)
    using ParamUseList = Map_t<FirstAndLastUseFinder<ExprList>::template Type, typename Base::AllParams>;
    using LocalVarUseList = Map_t<FirstAndLastUseFinder<ExprList>::template Type, AllLocalVars>;

private:
    // For each input Param in ParamUseList, add an OpCopyIn before its first use.
    // Reverse order makes the result "look" better.
    constexpr static auto result1 = ForEach(Reverse_t<ParamUseList>{}, CopyInInserter{}, ExprList{});

    // For each output Param in ParamUseList, add an OpCopyOut after its last use.
    constexpr static auto result2 = ForEach(ParamUseList{}, CopyOutInserter{}, result1);

public:
    using ExprListWithCopyX = decltype(result2);
};

template <typename ExprList /*TypeList*/, MemLevel memOpt = MemLevel::LEVEL_0>
struct FullAutoDag {
public:
    // Handle Params with Usage=InOut and has been `OpAssign`ed Params.
    using NewExprList = typename InplaceParamsProcessor<ExprList>::Type;
    // Extract common members
    using Base = DagBase<NewExprList>;
    // Insert CopyIn & Append CopyOut
    using OriExprListWithCopyX = decltype(ForEach(typename Base::AllParams{}, AddCopyX{}, NewExprList{}));
    // For loop each OpCopyIn & OpAssign & OpCopyOut
    using BindList = decltype(ForEach(OriExprListWithCopyX{}, OpAssign2Bind{}, TypeList<>{}));
    // TypeList of CopyOut Bind.
    using OutList = GetLastN_t<BindList, Size_v<typename Base::OutParams>>;
    // Compute Order List
    using OrderdOps = Unique_t<decltype(ForEach(OutList{}, ExtractDependOps{}, TypeList<>{}))>;
    // Node information of current DAG.
    using FullNodeInfo = DagNodeInfo<OrderdOps, OutList>;

    /***  Functions  ***/
private:
    template <typename NodeInfo>
    __host_aicore__ constexpr static MemLevel ChooseBufferLevel()
    {
        if constexpr (memOpt == MemLevel::LEVEL_0) {
            if constexpr ((NodeInfo::GetBufferNumLevel2()) <= MAX_BUFFER_NUMBER) {
                return MemLevel::LEVEL_2;
            } else if constexpr ((NodeInfo::GetBufferNumLevel1()) <= MAX_BUFFER_NUMBER) {
                return MemLevel::LEVEL_1;
            } else {
                return MemLevel::LEVEL_0;
            }
        } else {
            return memOpt;
        }
    }

    // Get All Mte2 Num including Persist ones in @NodeInfo.
    template <typename NodeInfo>
    __host_aicore__ constexpr static std::size_t GetMte2Num()
    {
        if constexpr (ChooseBufferLevel<NodeInfo>() == MemLevel::LEVEL_0) {
            return (NodeInfo::GetGMCountBeforeFirstCalcNode()) + (NodeInfo::GetPersistMte2Num());
        } else {
            return NodeInfo::inSizeWoScalar;
        }
    }

    // Get All Mte3 Num including Persist ones in @NodeInfo.
    template <typename NodeInfo>
    __host_aicore__ constexpr static std::size_t GetMte3Num()
    {
        constexpr auto persistMte3Num = NodeInfo::GetPersistMte3Num();
        if constexpr (ChooseBufferLevel<NodeInfo>() == MemLevel::LEVEL_0) {
            return (NodeInfo::GetFirstCopyOutNodeGMCount()) + persistMte3Num;
        } else {
            return (NodeInfo::GetLvl12Mte3Count()) + persistMte3Num;
        }
    }

    // Get All Temp Buff Num including Persist ones in @NodeInfo.
    template <typename NodeInfo>
    __host_aicore__ constexpr static std::size_t GetTempBufNum()
    {
        constexpr auto bufferLvl = ChooseBufferLevel<NodeInfo>();
        constexpr auto persistTempCalcBufNum = NodeInfo::GetPersistTempCalcBufNum();
        if constexpr (bufferLvl == MemLevel::LEVEL_0) {
            return (NodeInfo::GetLvl0TmpSize()) + persistTempCalcBufNum;
        } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
            return (NodeInfo::GetLvl1TmpSize()) + persistTempCalcBufNum;
        } else {
            return (NodeInfo::GetTempCalcNodeSize()) + persistTempCalcBufNum;
        }
    }

    // Get All Buff Num including Persist ones in @NodeInfo.
    template <typename NodeInfo>
    __host_aicore__ constexpr static uint32_t GetBufferNum()
    {
        constexpr auto bufferLvl = ChooseBufferLevel<NodeInfo>();
        if constexpr (bufferLvl == MemLevel::LEVEL_0) {
            return NodeInfo::GetBufferNumLevel0();
        } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
            return NodeInfo::GetBufferNumLevel1();
        } else { // bufferLvl == 2
            return NodeInfo::GetBufferNumLevel2();
        }
    }

    template <typename NodeInfo>
    __host_aicore__ constexpr static const int32_t* const* GetBufferIds()
    {
        // |mte2|mte3|tmp|mte2|mte3|
        constexpr auto bufferLvl = ChooseBufferLevel<NodeInfo>();
        // Total Mte2 Count including Persist Mte2 in CopyInBrc
        constexpr auto mte2Count = GetMte2Num<NodeInfo>();
        // Total Mte3 Count including Persist Mte3 if CopyInBrc/VecBrc connects to CopyOut
        constexpr auto mte3Count = GetMte3Num<NodeInfo>();
        // Total Temp Buf Count including Cache Temp buf in CopyInBrc & VecBrc
        constexpr auto tempBufCount = GetTempBufNum<NodeInfo>();
        constexpr auto totalCount = (mte2Count + mte3Count) * BUF_PING_PONG + tempBufCount;
        static_assert(
            totalCount <= BUF_MAX_COUNT,
            "Buffer count exceeded 32. Please try to switch MemLevel to LEVEL_1 or LEVEL_0.");
        constexpr auto pongOffset = mte2Count + mte3Count + tempBufCount;
        // Persist Mte2 Count
        constexpr auto persistMte2Count = NodeInfo::GetPersistMte2Num();
        // Persist Mte3 Count
        constexpr auto persistMte3Count = NodeInfo::GetPersistMte3Num();
        // Persist Temp Buf Count
        constexpr auto persistTempBufCount = NodeInfo::GetPersistTempCalcBufNum();
        // Generate Buffer ID List.
        using PersistMte2Lst = typename GenerateBufferWrappers<
            persistMte2Count, BUF_MTE2,
            /*Offset=*/0>::Type;
        using Mte2Lst = typename GenerateBufferWrappers<
            mte2Count - persistMte2Count, BUF_MTE2,
            /*Offset=*/persistMte2Count>::Type;
        using PersistMte3Lst = typename GenerateBufferWrappers<
            persistMte3Count, BUF_MTE3,
            /*Offset=*/mte2Count>::Type;
        using Mte3Lst = typename GenerateBufferWrappers<
            mte3Count - persistMte3Count, BUF_MTE3,
            /*Offset=*/mte2Count + persistMte3Count>::Type;
        using PersistTmpLst = typename GenerateBufferWrappers<
            persistTempBufCount, BUF_TEMP,
            /*Offset=*/mte2Count + mte3Count>::Type;
        using TmpLst = typename GenerateBufferWrappers<
            tempBufCount - persistTempBufCount, BUF_TEMP,
            /*Offset=*/mte2Count + mte3Count + persistTempBufCount>::Type;
        using PongMte3Lst = std::conditional_t<
            bufferLvl == MemLevel::LEVEL_0,
            typename GenerateBufferWrappers<
                mte3Count - persistMte3Count, BUF_MTE3,
                /*Offset=*/totalCount - (mte3Count - persistMte3Count),
                /*Where=*/BUF_PONG>::Type,
            TypeList<>>;
        return GenerateBufferIdOrder<
            typename NodeInfo::SavedOpList, typename NodeInfo::SavedOutList,
            TypeList<PersistMte2Lst, Mte2Lst, PersistMte3Lst, Mte3Lst, PersistTmpLst, TmpLst, PongMte3Lst>, pongOffset,
            bufferLvl>();
    }

private:
    using TmpRet = decltype(Bind2OpAssign<GetBufferIds<FullNodeInfo>(), OrderdOps, OutList>());

public:
    // common members.
    using AllParams = typename Base::AllParams;
    using InParams = typename Base::InParams;
    using OutParams = typename Base::OutParams;
    // TypeList of rebuilt CopyIn / CopyOut / OpAssign
    using ExprListWithCopyX = Get_t<TmpRet, 0>;
    // TypeList of ParamBufIdMap
    using BufMap = Get_t<TmpRet, 1>;
    // TypeList of LocalVar
    using AllLocalVars = LocalVars_t<ExprListWithCopyX>;
    // Get a type list of type lists of three items:
    //      (Param, FirstUseConst, LastUseConst)
    using ParamUseList = Map_t<FirstAndLastUseFinder<ExprListWithCopyX>::template Type, AllParams>;
    using LocalVarUseList = Map_t<FirstAndLastUseFinder<ExprListWithCopyX>::template Type, AllLocalVars>;
};

} // namespace Atvoss::Tile
#endif // ATVOSS_GRAPH_DAG_H