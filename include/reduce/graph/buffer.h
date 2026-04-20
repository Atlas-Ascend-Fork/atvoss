/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_GRAPH_BUFFER_H
#define ATVOSS_REDUCE_GRAPH_BUFFER_H

#include <cstdint>
#include <type_traits>
#include "common.h"
#include "expression/expr_template.h"

namespace Atvoss::Reduce::Graph {

using Atvoss::Params_t;
using Atvoss::Util::Append_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::TypeList;

constexpr static uint8_t BUF_PING_PONG = 2;

constexpr static uint32_t BUF_PARAM = 0b001;
constexpr static uint32_t BUF_LOCAL_VAR = 0b010;
constexpr static uint32_t BUF_REDUCE_VAR = 0b100;

template <typename ExprList, std::size_t pos>
struct CanRelease {
    template <typename Node>
    struct Type {
        static constexpr bool value = IsNodeUsedAfter<ExprList, Node, pos + 1, 0>::value == false;
    };
};

template <int32_t bufId, uint32_t bufUsage>
struct BufferWrapper {
    constexpr static int32_t bufferId = bufId;
    constexpr static uint32_t bufferUsage = bufUsage;
};

template <typename Node, typename BufferMap>
struct GetBufferId {
    static constexpr int32_t value = -1;
    static constexpr uint32_t usage = 0;
};

template <size_t N, typename T, ParamUsage U, size_t RN, typename BufWrapper>
struct GetBufferId<Param<N, T, U, RN>, TypeList<Param<N, T, U, RN>, BufWrapper>> {
    static constexpr int32_t value = BufWrapper::bufferId;
    static constexpr uint32_t usage = BufWrapper::bufferUsage;
};

template <size_t N, typename T, typename L, typename BufWrapper>
struct GetBufferId<LocalVar<N, T, L>, TypeList<LocalVar<N, T, L>, BufWrapper>> {
    static constexpr int32_t value = BufWrapper::bufferId;
    static constexpr uint32_t usage = BufWrapper::bufferUsage;
};

template <typename Node, typename First, typename BufWrapper, typename... Rest>
struct GetBufferId<Node, TypeList<First, BufWrapper, Rest...>> {
    static constexpr int32_t value = GetBufferId<Node, TypeList<Rest...>>::value;
    static constexpr uint32_t usage = GetBufferId<Node, TypeList<Rest...>>::usage;
};

template <typename LHS>
struct GetBufferUsage {
    static constexpr uint32_t value = BUF_LOCAL_VAR;
};

template <size_t N, typename T, ParamUsage U, size_t RN>
struct GetBufferUsage<Param<N, T, U, RN>> {
    static constexpr uint32_t value = BUF_PARAM;
};

template <typename BufferPool, int32_t BufId>
struct RemoveFromPool {
    using Type = BufferPool;
};

template <int32_t BufId, int32_t FirstVal, typename... Rest>
struct RemoveFromPool<TypeList<std::integral_constant<int32_t, FirstVal>, Rest...>, BufId> {
    using Type = std::conditional_t<
        FirstVal == BufId, TypeList<Rest...>,
        Append_t<typename RemoveFromPool<TypeList<Rest...>, BufId>::Type, std::integral_constant<int32_t, FirstVal>>>;
};

template <int32_t BufId>
struct RemoveFromPool<TypeList<>, BufId> {
    using Type = TypeList<>;
};

template <typename FreeBufferPool, typename BufferMap, typename InputNodeList, typename ExprList, std::size_t opPos>
struct ReleaseCurrentInputs {
    using Type = FreeBufferPool;
};

template <
    typename First, typename... Rest, typename FreeBufferPool, typename BufferMap, typename ExprList, std::size_t opPos>
struct ReleaseCurrentInputs<FreeBufferPool, BufferMap, TypeList<First, Rest...>, ExprList, opPos> {
private:
    static constexpr bool shouldRelease = CanRelease<ExprList, opPos>::template Type<First>::value;
    static constexpr int32_t bufferId = GetBufferId<First, BufferMap>::value;
    static constexpr uint32_t bufferUsage = GetBufferId<First, BufferMap>::usage;
    static constexpr bool isLocalVar = bufferUsage == BUF_LOCAL_VAR;
    static constexpr bool shouldAddToPool = shouldRelease && isLocalVar && bufferId >= 0;
    using RestType = typename ReleaseCurrentInputs<FreeBufferPool, BufferMap, TypeList<Rest...>, ExprList, opPos>::Type;

public:
    using Type =
        std::conditional_t<shouldAddToPool, Append_t<RestType, std::integral_constant<int32_t, bufferId>>, RestType>;
};

template <typename FreeBufferPool, typename BufferMap, typename ExprList, std::size_t opPos>
struct ReleaseCurrentInputs<FreeBufferPool, BufferMap, TypeList<>, ExprList, opPos> {
    using Type = FreeBufferPool;
};

template <
    typename ExprList, typename ReduceOpList, bool IsBinaryAcc, typename BufferMap, typename FreeBufferPool,
    std::size_t StartBufIdForLocalVar, std::size_t opPos = 0, std::size_t nextParamBufId = 0,
    std::size_t nextLocalBufId = StartBufIdForLocalVar, typename AllocList = TypeList<>,
    std::size_t Size = Size_v<ExprList>>
struct GenerateBufferIdOrderAux {
private:
    using CurrentOpAssign = Get_t<ExprList, opPos>;
    using LHS = typename CurrentOpAssign::LhsType;
    using RHS = typename CurrentOpAssign::RhsType;
    using CurrentInputs = typename ExtractInputs<RHS>::Type;

    static_assert(GetBufferId<LHS, BufferMap>::value < 0, "Duplicate output in different expressions");

    constexpr static bool IsLocalVarBuffer()
    {
        return GetBufferUsage<LHS>::value == BUF_LOCAL_VAR;
    }

    constexpr static bool IsReduceVarBuffer()
    {
        return IsBinaryAcc && IsLocalVarBuffer() && IsDirectConnectToReduce<ReduceOpList, CurrentOpAssign>::value;
    }

    constexpr static bool NeedsDoubleBuffer()
    {
        return !IsLocalVarBuffer() || IsReduceVarBuffer();
    }

    constexpr static bool CanReuseBuffer()
    {
        return IsLocalVarBuffer() && !IsReduceVarBuffer();
    }

    constexpr static bool HasFreeBuffer()
    {
        return Size_v<FreeBufferPool> > 0;
    }

    constexpr static int32_t AllocBufferId()
    {
        if constexpr (NeedsDoubleBuffer()) {
            return static_cast<int32_t>(nextParamBufId);
        } else {
            if constexpr (CanReuseBuffer() && HasFreeBuffer()) {
                return Get_t<FreeBufferPool, 0>::value;
            } else {
                return static_cast<int32_t>(nextLocalBufId);
            }
        }
    }

    constexpr static uint32_t GetAllocBufferUsage()
    {
        if constexpr (IsReduceVarBuffer()) {
            return BUF_REDUCE_VAR;
        } else {
            return GetBufferUsage<LHS>::value;
        }
    }

    constexpr static std::size_t GetNextParamBufId()
    {
        if constexpr (NeedsDoubleBuffer()) {
            return nextParamBufId + 1;
        } else {
            return nextParamBufId;
        }
    }

    constexpr static std::size_t GetNextLocalBufId()
    {
        if constexpr (!NeedsDoubleBuffer()) {
            if constexpr (CanReuseBuffer() && HasFreeBuffer()) {
                return nextLocalBufId;
            } else {
                return nextLocalBufId + 1;
            }
        } else {
            return nextLocalBufId;
        }
    }

    constexpr static auto UpdateFreePoolAfterAlloc()
    {
        if constexpr (!NeedsDoubleBuffer() && CanReuseBuffer() && HasFreeBuffer()) {
            return typename RemoveFromPool<FreeBufferPool, AllocBufferId()>::Type{};
        } else {
            return FreeBufferPool{};
        }
    }

    using NextFreeBufferPool = decltype(UpdateFreePoolAfterAlloc());
    using Buf = BufferWrapper<AllocBufferId(), GetAllocBufferUsage()>;
    using NextAllocList = Append_t<AllocList, Buf>;
    using NestBufferMap = Concatenate_t<BufferMap, TypeList<LHS, Buf>>;
    using UpdatedFreePool =
        typename ReleaseCurrentInputs<NextFreeBufferPool, NestBufferMap, CurrentInputs, ExprList, opPos>::Type;

public:
    using Type = typename GenerateBufferIdOrderAux<
        ExprList, ReduceOpList, IsBinaryAcc, NestBufferMap, UpdatedFreePool, StartBufIdForLocalVar, opPos + 1,
        GetNextParamBufId(), GetNextLocalBufId(), NextAllocList, Size>::Type;
};

template <
    typename ExprList, typename ReduceOpList, bool IsBinaryAcc, typename BufferMap, typename FreeBufferPool,
    std::size_t StartBufIdForLocalVar, std::size_t nextParamBufId, std::size_t nextLocalBufId, typename AllocList,
    std::size_t Size>
struct GenerateBufferIdOrderAux<
    ExprList, ReduceOpList, IsBinaryAcc, BufferMap, FreeBufferPool, StartBufIdForLocalVar, Size, nextParamBufId,
    nextLocalBufId, AllocList, Size> {
    using Type = AllocList;
};

template <typename AllocList>
struct MaxBufferId;

template <>
struct MaxBufferId<TypeList<>> {
    static constexpr int32_t value = -1;
};

template <typename First, typename... Rest>
struct MaxBufferId<TypeList<First, Rest...>> {
    static constexpr int32_t current = First::bufferId;
    static constexpr int32_t rest = MaxBufferId<TypeList<Rest...>>::value;
    static constexpr int32_t value = (current > rest) ? current : rest;
};

template <typename AllocList, typename Ids = TypeList<>>
struct TempCalcCount;

template <typename Ids>
struct TempCalcCount<TypeList<>, Ids> {
    static constexpr std::size_t value = 0;
};

template <typename First, typename... Rest, typename Ids>
struct TempCalcCount<TypeList<First, Rest...>, Ids> {
    static constexpr bool isLocalVar = First::bufferUsage != BUF_PARAM;
    static constexpr int32_t bufId = First::bufferId;
    static constexpr bool exist = Contains_v<Ids, std::integral_constant<int32_t, bufId>>;
    static constexpr bool needCount = isLocalVar && !exist;
    using NewIds = std::conditional_t<needCount, Append_t<Ids, std::integral_constant<int32_t, bufId>>, Ids>;
    static constexpr std::size_t current = needCount ? 1 : 0;
    static constexpr std::size_t rest = TempCalcCount<TypeList<Rest...>, NewIds>::value;
    static constexpr std::size_t value = current + rest;
};

template <int32_t bufferId, uint32_t bufferUsage, uint32_t pongOffset>
static constexpr int32_t CalcPongBufferId()
{
    if constexpr (bufferUsage == BUF_LOCAL_VAR) {
        return bufferId;
    } else {
        return bufferId + pongOffset;
    }
}

template <typename AllocList, uint32_t pongOffset>
struct ExtractBufferId {};

template <typename... Ts, uint32_t pongOffset>
struct ExtractBufferId<TypeList<Ts...>, pongOffset> {
    static constexpr size_t size = sizeof...(Ts);
    constexpr static int32_t arr[2][size] = {
        {Ts::bufferId...}, {CalcPongBufferId<Ts::bufferId, Ts::bufferUsage, pongOffset>()...}};
    constexpr static const int32_t* Value[2] = {arr[0], arr[1]};
};

template <typename ExprList, typename ReduceOpList = TypeList<>, bool IsBinaryAcc = true>
struct BufferIdGenerator {
private:
    static constexpr bool IsSingleExprNoBinaryAcc = (Size_v<ExprList> == 1) && !IsBinaryAcc;
    static constexpr std::size_t ResultParam = Size_v<Params_t<Map_t<ExtractResultType, ExprList>>>;
    static constexpr std::size_t StartBufIdForLocalVar =
        IsSingleExprNoBinaryAcc ? 1 : (IsBinaryAcc ? ResultParam + Size_v<ReduceOpList> : ResultParam);

    using AllocList = std::conditional_t<
        IsSingleExprNoBinaryAcc, TypeList<BufferWrapper<0, BUF_PARAM>, BufferWrapper<1, BUF_PARAM>>,
        std::conditional_t<
            Size_v<ExprList> == 1,
            TypeList<BufferWrapper<0, BUF_PARAM>, BufferWrapper<StartBufIdForLocalVar, BUF_LOCAL_VAR>>,
            typename GenerateBufferIdOrderAux<
                ExprList, ReduceOpList, IsBinaryAcc, TypeList<>, TypeList<>, StartBufIdForLocalVar>::Type>>;

    static constexpr int32_t maxBufferId = MaxBufferId<AllocList>::value;
    static constexpr std::size_t tempCalcCount = TempCalcCount<AllocList>::value;

    static constexpr int32_t singleExprNoBinaryAccArr[2][2] = {{0, 1}, {1, 0}};
    static constexpr const int32_t* singleExprNoBinaryAccBufferIds[2] = {
        singleExprNoBinaryAccArr[0], singleExprNoBinaryAccArr[1]};

public:
    static constexpr BufferIdGenerator GetInstance()
    {
        return BufferIdGenerator{};
    }

    static constexpr const int32_t* const* GetBufferIds()
    {
        if constexpr (IsSingleExprNoBinaryAcc) {
            return singleExprNoBinaryAccBufferIds;
        } else {
            return ExtractBufferId<AllocList, maxBufferId + 1>::Value;
        }
    }

    static constexpr int32_t GetTempCalcCount()
    {
        return tempCalcCount;
    }
};

} // namespace Atvoss::Reduce::Graph
#endif
