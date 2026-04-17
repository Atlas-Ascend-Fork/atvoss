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

#include "node.h"
#include <cstdint>
#include <type_traits>

namespace Atvoss::Reduce::Graph {

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
    std::size_t opPos = 0, std::size_t nextBufId = 0, typename AllocList = TypeList<>,
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
        if constexpr (CanReuseBuffer() && HasFreeBuffer()) {
            return Get_t<FreeBufferPool, 0>::value;
        } else {
            return static_cast<int32_t>(nextBufId);
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

    constexpr static std::size_t GetNextBufId()
    {
        if constexpr (CanReuseBuffer() && HasFreeBuffer()) {
            return nextBufId;
        } else {
            return nextBufId + 1;
        }
    }

    constexpr static auto UpdateFreePoolAfterAlloc()
    {
        if constexpr (CanReuseBuffer() && HasFreeBuffer()) {
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
        ExprList, ReduceOpList, IsBinaryAcc, NestBufferMap, UpdatedFreePool, opPos + 1, GetNextBufId(), NextAllocList,
        Size>::Type;
};

template <
    typename ExprList, typename ReduceOpList, bool IsBinaryAcc, typename BufferMap, typename FreeBufferPool,
    std::size_t nextBufId, typename AllocList, std::size_t Size>
struct GenerateBufferIdOrderAux<
    ExprList, ReduceOpList, IsBinaryAcc, BufferMap, FreeBufferPool, Size, nextBufId, AllocList, Size> {
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

template <typename AllocList, typename SeenIds = TypeList<>>
struct TempCalcCount;

template <typename SeenIds>
struct TempCalcCount<TypeList<>, SeenIds> {
    static constexpr std::size_t value = 0;
};

template <typename First, typename... Rest, typename SeenIds>
struct TempCalcCount<TypeList<First, Rest...>, SeenIds> {
    static constexpr bool isLocalVar = First::bufferUsage == BUF_LOCAL_VAR;
    static constexpr int32_t bufId = First::bufferId;
    static constexpr bool alreadySeen = Contains_v<SeenIds, std::integral_constant<int32_t, bufId>>;
    static constexpr bool shouldCount = isLocalVar && !alreadySeen;
    using NewSeenIds =
        std::conditional_t<shouldCount, Append_t<SeenIds, std::integral_constant<int32_t, bufId>>, SeenIds>;
    static constexpr std::size_t current = shouldCount ? 1 : 0;
    static constexpr std::size_t rest = TempCalcCount<TypeList<Rest...>, NewSeenIds>::value;
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
    using AllocList = std::conditional_t<
        Size_v<ExprList> == 1,
        TypeList<BufferWrapper<0, 1>, BufferWrapper<1, IsBinaryAcc ? BUF_REDUCE_VAR : BUF_LOCAL_VAR>>,
        typename GenerateBufferIdOrderAux<
            ExprList, ReduceOpList, IsBinaryAcc, TypeList<>, TypeList<>, 0, 0, TypeList<>>::Type>;

    static constexpr int32_t maxBufferId = MaxBufferId<AllocList>::value;
    static constexpr uint32_t pongOffset = static_cast<uint32_t>(maxBufferId + 1);
    static constexpr std::size_t tempCalcCount = TempCalcCount<AllocList>::value;

public:
    static constexpr size_t size = Size_v<AllocList>;
    static constexpr int32_t maxId = maxBufferId;
    static constexpr uint32_t offset = pongOffset;
    static constexpr std::size_t tempCalcNum = tempCalcCount;

    static constexpr BufferIdGenerator GetInstance()
    {
        return BufferIdGenerator{};
    }

    static constexpr const int32_t* const* GetBufferIds()
    {
        return ExtractBufferId<AllocList, pongOffset>::Value;
    }
};

} // namespace Atvoss::Reduce::Graph
#endif
