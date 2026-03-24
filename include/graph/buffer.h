/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_GRAPH_BUFFER_H
#define ATVOSS_GRAPH_BUFFER_H

#include "utils/utility.h"
#include "graph/bind.h"

namespace Atvoss::Tile {

using Atvoss::Util::Any_v;
using Atvoss::Util::Append_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Difference_t;
using Atvoss::Util::First_t;
using Atvoss::Util::Get_t;
using Atvoss::Util::IsNotEmpty;
using Atvoss::Util::IsNotEmpty_v;
using Atvoss::Util::RemoveFirstN_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::TypeList;

enum class BufType : uint8_t
{
    PARAM = 0,
    LOCAL_VAR
};

enum class BufPosInList : uint8_t
{
    PERSIST_MTE2 = 0,
    MTE2,
    PERSIST_MTE3,
    MTE3,
    PERSIST_TEMP,
    TEMP,
    PONG_MTE3,
    MAX_POS,
};

constexpr static std::size_t BUF_ALLOCATED_IDX = static_cast<std::size_t>(BufPosInList::MAX_POS);
constexpr static std::size_t BUF_TO_RELEASE_IDX = static_cast<std::size_t>(BufPosInList::MAX_POS);

constexpr static uint32_t BUF_MTE2 = 0b00'001;
constexpr static uint32_t BUF_MTE3 = 0b00'010;
constexpr static uint32_t BUF_TEMP = 0b00'100;
constexpr static uint32_t BUF_PLACEHOLDER = 0b01'000;
constexpr static uint32_t BUF_SCALAR = 0b10'000;

constexpr static uint32_t BUF_PING = 0b000'01;
constexpr static uint32_t BUF_PONG = 0b000'10;

constexpr static uint8_t BUF_COMBINE_SHIFT = 5;
constexpr static uint32_t BUF_COMBINE_MASK = 0x1F;
constexpr static uint8_t BUF_COMBINED_MAX = 5;

constexpr static uint8_t BUF_PING_PONG = 2;
constexpr static uint8_t BUF_MAX_COUNT = 32;

template <std::size_t N, BufType bt, uint32_t pingId, uint32_t pongId>
struct ParamBufIdMap {
    static const std::size_t paramNum = N;
    static const BufType bufType = bt;
    static const uint32_t pingBufId = pingId;
    static const uint32_t pongBufId = pongId;
};

template <std::size_t N, BufType bt, std::size_t pongOffset, std::size_t bufIdOffset = 0, typename TL = TypeList<>>
struct ParamBufIdMapGenerator {
private:
    static constexpr uint32_t pingId = static_cast<uint32_t>(N - 1 + bufIdOffset);
    static constexpr uint32_t pongId = static_cast<uint32_t>(bt == BufType::PARAM ? pingId + pongOffset : pingId);

public:
    using Type = Append_t<
        typename ParamBufIdMapGenerator<N - 1, bt, pongOffset, bufIdOffset, TL>::Type,
        ParamBufIdMap<N, bt, pingId, pongId>>;
};

template <BufType bt, std::size_t pongOffset, std::size_t bufIdOffset, typename TL /*TypeList*/>
struct ParamBufIdMapGenerator<0, bt, pongOffset, bufIdOffset, TL> {
    using Type = TL;
};

template <std::size_t paramCount, std::size_t localVarCount>
struct GenerateBufferId {
private:
    using ParamBuf = typename ParamBufIdMapGenerator<
        paramCount, BufType::PARAM,
        /*PongOffset=*/paramCount + localVarCount>::Type;
    using LocalBuf = typename ParamBufIdMapGenerator<
        localVarCount, BufType::LOCAL_VAR,
        /*PongOffset=*/paramCount + localVarCount,
        /*BufIdOffset=*/paramCount>::Type;

public:
    using Type = Concatenate_t<ParamBuf, LocalBuf>;
};

// Buffer wrapper Tempalte
template <int32_t bufId, uint32_t bufUsage, uint32_t where = BUF_PING>
struct BufferWrapper {
    // < 0 means no buffer will be allocated
    // only a placeholder to make schedule happy.
    constexpr static int32_t bufferId = bufId;
    constexpr static uint32_t bufferUsage = bufUsage;
    constexpr static uint32_t pingPong = where;
};

// Template used to generate buffer wrappers.
template <
    std::size_t count, uint32_t bufUsage, uint32_t bufIdOffset = 0, uint32_t where = BUF_PING,
    typename ResultList = TypeList<>, typename = void>
struct GenerateBufferWrappers {
    using Type = Append_t<
        typename GenerateBufferWrappers<count - 1, bufUsage, bufIdOffset, where, ResultList>::Type,
        BufferWrapper<count - 1 + bufIdOffset, bufUsage, where>>;
};

template <std::size_t count, uint32_t bufUsage, uint32_t bufIdOffset, uint32_t where, typename ResultList>
struct GenerateBufferWrappers<count, bufUsage, bufIdOffset, where, ResultList, std::enable_if_t<count <= 0>> {
    using Type = TypeList<>;
};

// Map to store relationship from bind to buffers.
template <typename B, typename... Ts>
struct Mapping {
    using Bind = B;
    using Buffers = TypeList<Ts...>;
};

template <typename B>
struct MappingBindEqual {
    template <typename M>
    using Type = std::is_same<typename M::Bind, B>;
};

template <typename Ms, typename B>
using MappingFind_t = Filter_t<MappingBindEqual<B>::template Type, Ms>;

// To Combine several buffers used in one Node.
// At most 5 (5*5 = 25 < 32. uint32_t has 32 bits).
template <typename... Ts>
struct CombineBufferWrapper {};

template <typename T>
struct CombineBufferWrapper<T> {
    constexpr static uint32_t bufferId = static_cast<uint32_t>(T::bufferId);
    constexpr static uint32_t bufferUsage = T::bufferUsage;
    constexpr static uint32_t pingPong = T::pingPong;
};

template <typename T, typename... Ts>
struct CombineBufferWrapper<T, Ts...> {
    constexpr static uint32_t bufferId = static_cast<uint32_t>(T::bufferId) << (BUF_COMBINE_SHIFT * sizeof...(Ts)) |
                                         CombineBufferWrapper<Ts...>::bufferId;
    constexpr static uint32_t bufferUsage =
        T::bufferUsage << (BUF_COMBINE_SHIFT * sizeof...(Ts)) | CombineBufferWrapper<Ts...>::bufferUsage;
    constexpr static uint32_t pingPong =
        T::pingPong << (BUF_COMBINE_SHIFT * sizeof...(Ts)) | CombineBufferWrapper<Ts...>::pingPong;
    ;
};

// Combined buffer wrapper template
template <typename... Ts>
struct CombinedBufferWrappers : BufferWrapper<
                                    static_cast<int>(
                                        static_cast<uint32_t>(sizeof...(Ts)) << (BUF_COMBINE_SHIFT * BUF_COMBINED_MAX) |
                                        CombineBufferWrapper<Ts...>::bufferId),
                                    CombineBufferWrapper<Ts...>::bufferUsage, CombineBufferWrapper<Ts...>::pingPong> {};

// Calculate Pong Buffer ID as per @bufferId and @pongOffset.
static constexpr int32_t CalcPongBufferId(
    int32_t bufferId, uint32_t bufferUsage, uint32_t pingPong, uint32_t pongOffset)
{
    if (pingPong > BUF_PONG) {
        // combined buffer wrapper.
        // bufferUsage/id/pingPong will always non-negative.
        if (pingPong == 0) {
            return 0;
        } else {
            const uint32_t id = static_cast<uint32_t>(bufferId) & BUF_COMBINE_MASK;
            const uint32_t usage = bufferUsage & BUF_COMBINE_MASK;
            const uint32_t pp = pingPong & BUF_COMBINE_MASK;
            const uint32_t currentId =
                pp == BUF_PING ? (bufferUsage == BUF_TEMP ? id : (id + pongOffset)) : (id - pongOffset);
            const int32_t idNext = static_cast<int32_t>(static_cast<uint32_t>(bufferId) >> BUF_COMBINE_SHIFT);
            const uint32_t usageNext = bufferUsage >> BUF_COMBINE_SHIFT;
            const uint32_t ppNext = pingPong >> BUF_COMBINE_SHIFT;
            return static_cast<int32_t>(
                static_cast<uint32_t>(CalcPongBufferId(idNext, usageNext, ppNext, pongOffset)) << BUF_COMBINE_SHIFT |
                currentId);
        }
    } else {
        // pure buffer wrapper
        return pingPong == BUF_PING ? ((bufferUsage == BUF_TEMP || bufferUsage == BUF_SCALAR) ?
                                           bufferId :
                                           (bufferUsage == BUF_PLACEHOLDER ? -1 : (bufferId + pongOffset))) :
                                      (bufferId - pongOffset);
    }
}

// Extract PingPong BufferID from `AllocList`
template <typename Ts, uint32_t pongOffset>
struct ExtractBufferId {};

template <typename... Ts, uint32_t pongOffset>
struct ExtractBufferId<TypeList<Ts...>, pongOffset> {
    static constexpr size_t size = sizeof...(Ts);
    constexpr static int32_t arr[2][size] = {
        {Ts::bufferId...}, {CalcPongBufferId(Ts::bufferId, Ts::bufferUsage, Ts::pingPong, pongOffset)...}};
    constexpr static const int32_t* Value[2] = {arr[0], arr[1]};
};

struct PreReduceOnlyCopyInBufferId {
    constexpr static int32_t arr[2][2] = {{0, 1}, {2, 3}};
    constexpr static const int32_t* Value[2] = {arr[0], arr[1]};
};

// Combined BufferID Decode
template <uint32_t... Ints>
struct IntegerSequence {};

template <uint32_t N, uint32_t... Ints>
struct MakeIntegerSequenceAux : MakeIntegerSequenceAux<N - 1, N - 1, Ints...> {};

template <uint32_t... Ints>
struct MakeIntegerSequenceAux<0, Ints...> {
    using Type = IntegerSequence<Ints...>;
};

template <uint32_t N>
using MakeIntegerSequence = typename MakeIntegerSequenceAux<N>::Type;

template <int32_t bufferId, uint32_t N, uint32_t pos>
struct DecodeBufferIdWithPos {
    constexpr static int Value =
        static_cast<int>(static_cast<uint32_t>(bufferId) >> (BUF_COMBINE_SHIFT * (N - 1 - pos)) & BUF_COMBINE_MASK);
};

template <int32_t bufferId>
struct CombinedBufferCount {
    const static uint32_t tmp =
        static_cast<uint32_t>(bufferId) >> (BUF_COMBINE_SHIFT * BUF_COMBINED_MAX) & BUF_COMBINE_MASK;
    constexpr static uint32_t Value = (tmp == 0 || tmp > BUF_COMBINED_MAX) ? 1 : tmp;
};

template <int32_t bufferId, typename IntSeq>
struct DecodeBufferIdAux {};

template <int32_t bufferId, uint32_t... Ints>
struct DecodeBufferIdAux<bufferId, IntegerSequence<Ints...>> {
    constexpr static int Value[sizeof...(Ints)] = {DecodeBufferIdWithPos<bufferId, sizeof...(Ints), Ints>::Value...};
};

template <int32_t bufferId, uint32_t N = CombinedBufferCount<bufferId>::Value>
struct DecodeBufferId {
    using IntSequence = MakeIntegerSequence<N>;
    constexpr static const int* const Value = {DecodeBufferIdAux<bufferId, IntSequence>::Value};
};

// Release Buffer
template <
    typename InputOps, typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, uint32_t bufferUsage,
    bool cacheBrc = false, uint32_t where = BUF_PING>
struct ReleaseBufferByUsageAux {};

template <
    typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, uint32_t bufferUsage, bool cacheBrc,
    uint32_t where>
struct ReleaseBufferByUsageAux<TypeList<>, OpLst, ToReleaseLst, currentOpPos, bufferUsage, cacheBrc, where> {
    using Type = TypeList<>;
};

// Check & Release Buffer in TypeList<Head, Tail...>
template <
    typename Head, typename... Tail, typename OpLst, typename ToReleaseLst, std::size_t currentOpPos,
    uint32_t bufferUsage, bool cacheBrc, uint32_t where>
struct ReleaseBufferByUsageAux<
    TypeList<Head, Tail...>, OpLst, ToReleaseLst, currentOpPos, bufferUsage, cacheBrc, where> {
private:
    template <typename T>
    struct BufferUsageEqual : std::bool_constant<bufferUsage == T::bufferUsage && T::pingPong == where> {};

    constexpr static bool ableToRelease = IsAbleToFree<OpLst, Head, cacheBrc, currentOpPos + 1>();
    // Find the Map of @Head in @ToReleaseLst
    using Mappings = MappingFind_t<ToReleaseLst, Head>;
    static_assert(Size_v<Mappings> == 1, "Mappings::Size == 1");
    // Find the buffer from the mapped buffers as per @bufferUsage
    using BufMapping = First_t<Mappings>;
    using Buffers = Filter_t<BufferUsageEqual, typename BufMapping::Buffers>;
    using Left = typename ReleaseBufferByUsageAux<
        TypeList<Tail...>, OpLst, ToReleaseLst, currentOpPos, bufferUsage, cacheBrc, where>::Type;

public:
    using Type = Concatenate_t<std::conditional_t<ableToRelease, Buffers, TypeList<>>, Left>;
};

// Release buffer in @ToReleaseLst as per usage @bufferUsage
template <
    typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, uint32_t bufferUsage, bool cacheBrc = false,
    uint32_t where = BUF_PING>
struct ReleaseBufferByUsage {
    using Op = Get_t<OpLst, currentOpPos>;
    using InOps = typename Op::InNonScalarOps;
    using Type =
        typename ReleaseBufferByUsageAux<InOps, OpLst, ToReleaseLst, currentOpPos, bufferUsage, cacheBrc, where>::Type;
};

// Collect unused input ops buffer in @InputOps from @ToReleaseLst
template <typename InputOps, typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, bool cacheBrc>
struct ReleaseUnusedInputAux {};

template <typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, bool cacheBrc>
struct ReleaseUnusedInputAux<TypeList<>, OpLst, ToReleaseLst, currentOpPos, cacheBrc> {
    using Type = TypeList<>;
};

template <
    typename Head, typename... Tail, typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, bool cacheBrc>
struct ReleaseUnusedInputAux<TypeList<Head, Tail...>, OpLst, ToReleaseLst, currentOpPos, cacheBrc> {
private:
    constexpr static bool ableToRelease = IsAbleToFree<OpLst, Head, cacheBrc, currentOpPos + 1>();
    // Find @Head in @ToReleaseLst
    using Mappings = MappingFind_t<ToReleaseLst, Head>;
    static_assert(Size_v<Mappings> == 1, "Mappings::Size == 1");
    using Left = typename ReleaseUnusedInputAux<TypeList<Tail...>, OpLst, ToReleaseLst, currentOpPos, cacheBrc>::Type;

public:
    using Type = Concatenate_t<std::conditional_t<ableToRelease, Mappings, TypeList<>>, Left>;
};

// Remove unused input ops of OpX @currentOpPos from @ToReleaseLst
template <typename OpLst, typename ToReleaseLst, std::size_t currentOpPos, bool cacheBrc>
struct ReleaseUnusedInput {
    using Op = Get_t<OpLst, currentOpPos>;
    using InputOps = typename Op::InNonScalarOps;
    using NeedReleaseLst = typename ReleaseUnusedInputAux<InputOps, OpLst, ToReleaseLst, currentOpPos, cacheBrc>::Type;
    using Type = Difference_t<ToReleaseLst, NeedReleaseLst>;
};

// Get first element from the first non-empty List.
template <typename T>
struct PriorityGetFirstAux {};

template <typename T>
struct PriorityGetFirstAux<TypeList<T>> {
    using Type = First_t<T>;
};

template <typename Head, typename... Tail>
struct PriorityGetFirstAux<TypeList<Head, Tail...>> {
    using Type =
        std::conditional_t<IsNotEmpty_v<Head>, First_t<Head>, typename PriorityGetFirstAux<TypeList<Tail...>>::Type>;
};

template <typename... Ts>
struct PriorityGetFirst {
    static_assert(Any_v<IsNotEmpty, TypeList<Ts...>>, "At least one memory List should not be empty.");
    using Type = typename PriorityGetFirstAux<TypeList<Ts...>>::Type;
};

template <typename... Ts>
using PriorityGetFirst_t = typename PriorityGetFirst<Ts...>::Type;

// Pop the first element of @Ts if equals to @ToPop
template <typename Ts, typename ToPop, typename = void>
struct PopFrontIfEqual {
    using Type = Ts;
};

template <typename Ts, typename ToPop>
struct PopFrontIfEqual<Ts, ToPop, std::enable_if_t<IsNotEmpty_v<Ts> && (std::is_same_v<First_t<Ts>, ToPop>)>> {
    using Type = RemoveFirstN_t<Ts, 1>;
};

template <typename Ts, typename ToPop>
using PopFrontIfEqual_t = typename PopFrontIfEqual<Ts, ToPop>::Type;

template <typename BufLstLst>
struct BufLstLstDecoder {
public:
    using PersistMte2Lst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::PERSIST_MTE2)>;
    using Mte2Lst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::MTE2)>;
    using PersistMte3Lst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::PERSIST_MTE3)>;
    using Mte3Lst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::MTE3)>;
    using PersistTmpLst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::PERSIST_TEMP)>;
    using TmpLst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::TEMP)>;
    using PongMte3Lst = Get_t<BufLstLst, static_cast<std::size_t>(BufPosInList::PONG_MTE3)>;
};

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
struct AllocMte2 {
private:
    using BufLsts = BufLstLstDecoder<BufLstLst>;
    // pop front
    using UsedTmpLst = std::conditional_t<memLvl == MemLevel::LEVEL_2 || cache, TypeList<>, typename BufLsts::TmpLst>;
    using UsedPongMte3Lst =
        std::conditional_t<memLvl != MemLevel::LEVEL_0 || cache, TypeList<>, typename BufLsts::PongMte3Lst>;
    using UsedPersistMte2Lst = std::conditional_t<!cache, TypeList<>, typename BufLsts::PersistMte2Lst>;
    using UsedMte2Lst = std::conditional_t<cache, TypeList<>, typename BufLsts::Mte2Lst>;
    using Mte2 = PriorityGetFirst_t<UsedPersistMte2Lst, UsedMte2Lst, UsedTmpLst, UsedPongMte3Lst>;
    // update
    using Mte2LstNext = PopFrontIfEqual_t<typename BufLsts::Mte2Lst, Mte2>;
    using TmpLstNext = PopFrontIfEqual_t<typename BufLsts::TmpLst, Mte2>;
    using PongMte3LstNext = PopFrontIfEqual_t<typename BufLsts::PongMte3Lst, Mte2>;
    using PersistMte2LstNext = PopFrontIfEqual_t<typename BufLsts::PersistMte2Lst, Mte2>;

public:
    using Type = TypeList<
        PersistMte2LstNext, Mte2LstNext, typename BufLsts::PersistMte3Lst, typename BufLsts::Mte3Lst,
        typename BufLsts::PersistTmpLst, TmpLstNext, PongMte3LstNext, Mte2>;
};

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
using AllocMte2_t = typename AllocMte2<BufLstLst, memLvl, cache>::Type;

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
struct AllocTempBuffer {
private:
    using BufLsts = BufLstLstDecoder<BufLstLst>;
    // pop front
    using UsedPongMte3Lst =
        std::conditional_t<memLvl != MemLevel::LEVEL_0 || cache, TypeList<>, typename BufLsts::PongMte3Lst>;
    using UsedMte2Lst = std::conditional_t<memLvl == MemLevel::LEVEL_2 || cache, TypeList<>, typename BufLsts::Mte2Lst>;
    using UsedPersistTmpLst = std::conditional_t<!cache, TypeList<>, typename BufLsts::PersistTmpLst>;
    using UsedTmpLst = std::conditional_t<cache, TypeList<>, typename BufLsts::TmpLst>;
    using Tmp = PriorityGetFirst_t<UsedPersistTmpLst, UsedTmpLst, UsedPongMte3Lst, UsedMte2Lst>;
    // update
    using TmpLstNext = PopFrontIfEqual_t<typename BufLsts::TmpLst, Tmp>;
    using PongMte3LstNext = PopFrontIfEqual_t<typename BufLsts::PongMte3Lst, Tmp>;
    using Mte2LstNext = PopFrontIfEqual_t<typename BufLsts::Mte2Lst, Tmp>;
    using PersistTmpLstNext = PopFrontIfEqual_t<typename BufLsts::PersistTmpLst, Tmp>;

public:
    using Type = TypeList<
        typename BufLsts::PersistMte2Lst, Mte2LstNext, typename BufLsts::PersistMte3Lst, typename BufLsts::Mte3Lst,
        PersistTmpLstNext, TmpLstNext, PongMte3LstNext, Tmp>;
};

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
using AllocTempBuffer_t = typename AllocTempBuffer<BufLstLst, memLvl, cache>::Type;

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
struct AllocMte3 {
private:
    using BufLsts = BufLstLstDecoder<BufLstLst>;
    // pop front
    using UsedTmpLst = std::conditional_t<memLvl != MemLevel::LEVEL_0 || cache, TypeList<>, typename BufLsts::TmpLst>;
    using UsedPongMte3Lst =
        std::conditional_t<memLvl != MemLevel::LEVEL_0 || cache, TypeList<>, typename BufLsts::PongMte3Lst>;
    using UsedMte2Lst = std::conditional_t<memLvl != MemLevel::LEVEL_0 || cache, TypeList<>, typename BufLsts::Mte2Lst>;
    using UsedPersistMte3Lst = std::conditional_t<!cache, TypeList<>, typename BufLsts::PersistMte3Lst>;
    using UsedMte3Lst = std::conditional_t<cache, TypeList<>, typename BufLsts::Mte3Lst>;
    using Mte3 = PriorityGetFirst_t<UsedPersistMte3Lst, UsedMte3Lst, UsedTmpLst, UsedPongMte3Lst, UsedMte2Lst>;
    // update
    using Mte3LstNext = PopFrontIfEqual_t<typename BufLsts::Mte3Lst, Mte3>;
    using TmpLstNext = PopFrontIfEqual_t<typename BufLsts::TmpLst, Mte3>;
    using PongMte3LstNext = PopFrontIfEqual_t<typename BufLsts::PongMte3Lst, Mte3>;
    using Mte2LstNext = PopFrontIfEqual_t<typename BufLsts::Mte2Lst, Mte3>;
    using PersistMte3LstNext = PopFrontIfEqual_t<typename BufLsts::PersistMte3Lst, Mte3>;

public:
    using Type = TypeList<
        typename BufLsts::PersistMte2Lst, Mte2LstNext, PersistMte3LstNext, Mte3LstNext, typename BufLsts::PersistTmpLst,
        TmpLstNext, PongMte3LstNext, Mte3>;
};

template <typename BufLstLst, MemLevel memLvl, bool cache = false>
using AllocMte3_t = typename AllocMte3<BufLstLst, memLvl, cache>::Type;

// Release input buffers of OpX @opPos to @BufLstLst
// and remove from @ToReleaseLst
template <typename OpLst, typename BufLstLst, typename ToReleaseLst, bool cacheBrc, std::size_t opPos>
struct ReleaseAndUpdateLst {
private:
    using BufLsts = BufLstLstDecoder<BufLstLst>;
    // release & update
    using PersistMte2Lst = typename BufLsts::PersistMte2Lst;
    using Mte2LstNext = Concatenate_t<
        typename BufLsts::Mte2Lst, typename ReleaseBufferByUsage<OpLst, ToReleaseLst, opPos, BUF_MTE2, cacheBrc>::Type>;
    using PersistMte3Lst = typename BufLsts::PersistMte3Lst;
    using Mte3LstNext = Concatenate_t<
        typename BufLsts::Mte3Lst, typename ReleaseBufferByUsage<OpLst, ToReleaseLst, opPos, BUF_MTE3, cacheBrc>::Type>;
    using PersistTmpLst = typename BufLsts::PersistTmpLst;
    using TmpLstNext = Concatenate_t<
        typename BufLsts::TmpLst, typename ReleaseBufferByUsage<OpLst, ToReleaseLst, opPos, BUF_TEMP, cacheBrc>::Type>;
    using PongMte3LstNext = Concatenate_t<
        typename BufLsts::PongMte3Lst,
        typename ReleaseBufferByUsage<OpLst, ToReleaseLst, opPos, BUF_MTE3, cacheBrc, BUF_PONG>::Type>;
    // update ToReleaseLst to speedup next release
    using ToReleaseLstNext = typename ReleaseUnusedInput<OpLst, ToReleaseLst, opPos, cacheBrc>::Type;

public:
    using Type = TypeList<
        PersistMte2Lst, Mte2LstNext, PersistMte3Lst, Mte3LstNext, PersistTmpLst, TmpLstNext, PongMte3LstNext,
        ToReleaseLstNext>;
};

#if !defined(__ATVOSS_HOST_ONLY__)
template <typename TL, std::size_t N, BufType bt, int start = 0>
__aicore__ auto GetBufferId(bool isPing = true)
{
    if constexpr (start < Size_v<TL>) {
        using bufMap = Get_t<TL, start>;
        if constexpr (bufMap::paramNum == N && bufMap::bufType == bt) {
            return isPing ? bufMap::pingBufId : bufMap::pongBufId;
        } else {
            return GetBufferId<TL, N, bt, start + 1>(isPing);
        }
    } else {
        static_assert(start < Size_v<TL>, "Param or LocalVar Id invalid.");
    }
};
#endif

/*
 * Generate buffer id of each Expression savedin @OpLst as per @memLvl
 * Template Arguments：
 *   1. OpLst: Ordered full Expression / Compute list
 *   2. OutLst: Full list of Output
 *   3. BufLstLst：Unused MTE2/MTE3/TmpBuffer List
 *   4. pongOffset: Offset of Pong buffer id to Ping buffer
 *   5. memLvl: memory level (policy) 0/1/2
 *   6. useNddma: whether use NDDMA from CopyInBrc operator
 *   7. cacheBrc: whether cache CopyInBrc & VecBrc operation outputs
 *   8. AllocLst: Allocated buffer list of `BufferWrapper` currently
 *   9. ToReleaseLst: To release list of `Mapping<Bind, TypeList<BufferWrapper...>>`
 *  10. scalarIdx: index or placehold of next `Scalar` operation output
 *  11. opPos: Position of current Expression / Compute
 * Return：
 *   1. 2*Size_v<OpLst> integer matrix as `const int32_t* const*`
 */
template <
    typename OpLst, typename OutLst, typename BufLstLst, uint32_t pongOffset, MemLevel memLvl = MemLevel::LEVEL_2,
    bool useNddma = true, bool cacheBrc = false, typename AllocLst = TypeList<>, typename ToReleaseLst = TypeList<>,
#ifdef __ATP_UT__
    int32_t scalarIdx = 50,
#else
    int32_t scalarIdx = 0,
#endif
    std::size_t opPos = 0>
static constexpr const int32_t* const* GenerateBufferIdOrder()
{
    if constexpr (opPos < Size_v<OpLst>) {
        using Op = Get_t<OpLst, opPos>;
        if constexpr (Op::isScalarOp) {
            using Buf = BufferWrapper<scalarIdx, BUF_SCALAR>;
            using AllocLstNext = Append_t<AllocLst, Buf>;
            return GenerateBufferIdOrder<
                OpLst, OutLst, BufLstLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLst,
                scalarIdx + 1, opPos + 1>();
        } else if constexpr (false /*IsBindOfOp_v<OpCopyInBrc, Op>*/ && !useNddma) { // TODO
            // use copyIn + ubBrc to implement nddma.
            using NextLst0 = AllocMte2_t<BufLstLst, memLvl, cacheBrc>;
            using Mte2 = Get_t<NextLst0, BUF_ALLOCATED_IDX>;
            if constexpr (ConnectToAny_v<OutLst, Op>) {
                // if CopyInBrc -> CopyOut, should use MTE3
                using NextLst = AllocMte3_t<NextLst0, memLvl, cacheBrc>;
                using Mte3 = Get_t<NextLst, BUF_ALLOCATED_IDX>;
                // add to alloc list
                using Buf = CombinedBufferWrappers<Mte3, Mte2>;
                using AllocLstNext = Append_t<AllocLst, Buf>;
                using ToReleaseLstNext = Append_t<ToReleaseLst, Mapping<Op, Mte3, Mte2>>;
                // next
                return GenerateBufferIdOrder<
                    OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                    scalarIdx, opPos + 1>();
            } else {
                using NextLst = AllocTempBuffer_t<NextLst0, memLvl, cacheBrc>;
                using Tmp = Get_t<NextLst, BUF_ALLOCATED_IDX>;
                // add to alloc list
                using Buf = CombinedBufferWrappers<Tmp, Mte2>;
                using AllocLstNext = Append_t<AllocLst, Buf>;
                using ToReleaseLstNext = Append_t<ToReleaseLst, Mapping<Op, Tmp, Mte2>>;
                // next
                return GenerateBufferIdOrder<
                    OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                    scalarIdx, opPos + 1>();
            }
        } else if constexpr (IsBindOfOp_v<OpCopyIn, Op>) {
            // activate cache when CopyInBrc = NDDMA
            using NextLst = AllocMte2_t<BufLstLst, memLvl, IsCacheBind_v<cacheBrc, Op>>;
            using Mte2 = Get_t<NextLst, BUF_ALLOCATED_IDX>;
            // add to alloc list
            using AllocLstNext = Append_t<AllocLst, Mte2>;
            using ToReleaseLstNext = Append_t<ToReleaseLst, Mapping<Op, Mte2>>;
            // next
            return GenerateBufferIdOrder<
                OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                scalarIdx, opPos + 1>();
        } else if constexpr (IsBindOfOp_v<OpCopyOut, Op>) {
            using NextLst = typename ReleaseAndUpdateLst<OpLst, BufLstLst, ToReleaseLst, cacheBrc, opPos>::Type;
            // release and update
            using ToReleaseLstNext = Get_t<NextLst, BUF_TO_RELEASE_IDX>;
            // fill -1 when no buffer is needed.
            using Buf = BufferWrapper<-1, BUF_PLACEHOLDER>;
            using AllocLstNext = Append_t<AllocLst, Buf>;
            // next
            return GenerateBufferIdOrder<
                OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                scalarIdx, opPos + 1>();
        } else {
            if constexpr (ConnectToAny_v<OutLst, Op>) {
                // activate cache when current Node is VecBrc
                using NextLst0 = AllocMte3_t<BufLstLst, memLvl, IsCacheBind_v<cacheBrc, Op>>;
                using Mte3 = Get_t<NextLst0, BUF_ALLOCATED_IDX>;
                // add to alloc list
                using AllocLstNext = Append_t<AllocLst, Mte3>;
                // release and update
                using NextLst = typename ReleaseAndUpdateLst<OpLst, NextLst0, ToReleaseLst, cacheBrc, opPos>::Type;
                using ToReleaseLstNext = Append_t<Get_t<NextLst, BUF_TO_RELEASE_IDX>, Mapping<Op, Mte3>>;
                // next
                return GenerateBufferIdOrder<
                    OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                    scalarIdx, opPos + 1>();
            } else {
                // activate cache when current Node is VecBrc
                using NextLst0 = AllocTempBuffer_t<BufLstLst, memLvl, IsCacheBind_v<cacheBrc, Op>>;
                using Tmp = Get_t<NextLst0, BUF_ALLOCATED_IDX>;
                // add to alloc list
                using AllocLstNext = Append_t<AllocLst, Tmp>;
                // release and update
                using NextLst = typename ReleaseAndUpdateLst<OpLst, NextLst0, ToReleaseLst, cacheBrc, opPos>::Type;
                using ToReleaseLstNext = Append_t<Get_t<NextLst, BUF_TO_RELEASE_IDX>, Mapping<Op, Tmp>>;
                // next
                return GenerateBufferIdOrder<
                    OpLst, OutLst, NextLst, pongOffset, memLvl, useNddma, cacheBrc, AllocLstNext, ToReleaseLstNext,
                    scalarIdx, opPos + 1>();
            }
        }
    } else {
        // generate buffer id list for ping-pong as per AllocLst.
        static_assert(Size_v<OpLst> == Size_v<AllocLst>, "Size_v<OpLst> == Size_v<AllocLst>");
        return ExtractBufferId<AllocLst, pongOffset>::Value;
    }
}

} // namespace Atvoss::Tile
#endif // ATVOSS_GRAPH_BUFFER_H