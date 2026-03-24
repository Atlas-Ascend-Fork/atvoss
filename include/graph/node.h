/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_GRAPH_NODE_H
#define ATVOSS_GRAPH_NODE_H

#include "graph/bind.h"

namespace Atvoss::Tile {

using Atvoss::Util::Append_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Contains_v;
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::ForEach;
using Atvoss::Util::Get_t;
using Atvoss::Util::IsSpecializationOf_v;
using Atvoss::Util::Map_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::Unique_t;

constexpr static uint8_t MAX_DTYPE_BYTES = 32;

template <typename OutputList>
struct TempCalcNodeChecker {
private:
    template <typename B>
    struct UseTempBuffer : std::bool_constant<!(IsBindOfOp_v<OpCopyIn, B> || ConnectToAny_v<OutputList, B>)> {};

public:
    template <typename B>
    using Type = UseTempBuffer<B>;
};

/*
 * Collect count of `CopyIn` before first vector calculation.
 * Template Arguments:
 *   1. OpList: Ordered full Expression / Compute list
 *   2. RsvList: Cache node list
 *   3. start：Position of current Expression / Compute
 *   4. Acc: Nodes of `CopyIn` so far
 */
template <typename OpList, typename RsvList = TypeList<>, std::size_t start = 0, typename Acc = TypeList<>>
__host_aicore__ constexpr std::size_t GetCopyInCountBeforeFirstCalcNode()
{
    /* need to skip `Scalar` operation & cache nodes */
    if constexpr (start < Size_v<OpList>) {
        using bind = Get_t<OpList, start>;
        if constexpr (IsBindOfOp_v<OpCopyIn, bind>) {
            using Next = std::conditional_t<Contains_v<RsvList, bind>, Acc, Append_t<Acc, bind>>;
            return GetCopyInCountBeforeFirstCalcNode<OpList, RsvList, start + 1, Next>();
        } else if constexpr (IsBindOfOp_v<OpCopyOut, bind>) {
            return GetCopyInCountBeforeFirstCalcNode<OpList, RsvList, start + 1, Acc>();
        } else {
            return Size_v<Acc>;
        }
    }
    return Size_v<Acc>;
};

template <typename OpList, std::size_t start>
struct WillNotUsed {
private:
    template <typename B>
    struct NotUsing : std::bool_constant<IsAbleToFree<OpList, B, false, start>()> {};

public:
    template <typename B>
    using Type = NotUsing<B>;
};

struct DagMaxAliveInfo {
    std::size_t aliveNode = 0;    // max alive node size
    std::size_t tempCalcNode = 0; // nodes temporarily acllcated

    constexpr DagMaxAliveInfo() : aliveNode(0), tempCalcNode(0)
    {}

    constexpr DagMaxAliveInfo(const DagMaxAliveInfo& v) : aliveNode(v.aliveNode), tempCalcNode(v.tempCalcNode)
    {}
};

template <typename T>
static constexpr const T& Max(const T& a, const T& b)
{
    return a > b ? a : b;
}

/*
 * Calculate max alive nodes excluding cache nodes saved in @RsvList
 * Template Arguments：
 *   1. OpList: Ordered full Expression / Compute list
 *   2. OutList: Full list of output
 *   3. RsvList: Cache node list
 *   4. start: Position of current Expression / Compute
 *   5. Acc: Alive nodes so far
 * Return:
 *   1. Max Alive Node information saved in `DagMaxAliveInfo`
 */
template <
    typename OpList, typename OutList, typename RsvList = TypeList<>, std::size_t start = 0, typename Acc = TypeList<>>
constexpr DagMaxAliveInfo MaxAliveNode(DagMaxAliveInfo info)
{
    if constexpr (start < Size_v<OpList>) {
        using Op = Get_t<OpList, start>;

        // Dependencies of current node: inputs & outputs
        using InOutNodes = std::conditional_t<
            IsBindOfOp_v<OpCopyOut, Op>, typename Op::InNonScalarOps,
            Append_t<typename Op::InNonScalarOps, typename Op::BindType>>;
        // Union alive nodes so far saved in @Acc.
        // Cache nodes saved in `RsvLst` will be removed.
        using AliveNodes = Difference_t<Unique_t<Concatenate_t<Acc, InOutNodes>>, RsvList>;
        using TempCalcNodes = Filter_t<TempCalcNodeChecker<OutList>::template Type, AliveNodes>;

        // alive node size
        constexpr auto AliveNodeSize = Size_v<AliveNodes>;
        constexpr auto TempCalcNodeSize = Size_v<TempCalcNodes>;
        info.aliveNode = Max<std::size_t>(AliveNodeSize, info.aliveNode);
        info.tempCalcNode = Max<std::size_t>(TempCalcNodeSize, info.tempCalcNode);

        // Collect unused inputs of current node.
        using DelVar = Filter_t<WillNotUsed<OpList, start + 1>::template Type, typename Op::InNonScalarOps>;
        // Delete unused inputs of current node.
        using Next = Difference_t<AliveNodes, DelVar>;

        return MaxAliveNode<OpList, OutList, RsvList, start + 1, Next>(info);
    }
    return info;
};

/*
 * To collect node information in @OpList
 */
template <typename OpList /*TypeList of Bind*/, typename OutList /*TypeList of Bind*/>
struct DagNodeInfo {
public:
    // Save input template argument @OpList
    using SavedOpList = OpList;
    using SavedOutList = OutList;

    // Collect In/Out Parameters including ScalarTensor.
    using AllParams = Unique_t<Concatenate_t<ExtractBindParams_t<OpList>, Map_t<ExtractBindAssignTo, OutList>>>;
    using InParams = Filter_t<IsInParam, AllParams>;
    using OutParams = Filter_t<IsOutParam, AllParams>;

    // Input/Output GM size
    constexpr static std::size_t inSize = Size_v<InParams>;
    constexpr static std::size_t outSize = Size_v<OutParams>;

    // TODO: Input GM size without InScalarHolders
    constexpr static std::size_t inSizeWoScalar = inSize;

    // Max alive node information for normal scenario.
    constexpr static auto maxAliveNodeInfo = MaxAliveNode<OpList, OutList>(DagMaxAliveInfo());

private:
    // CopyInNodes . TODO: without Scalar-CopyIn
    using CopyInNodes = Filter_t<BindOpChecker<OpCopyIn>::template Type, OpList>;
    // Collect CopyIn Nodes connecting to CopyOut Nodes.
    using CopyInNodesLinkCopyOut = Filter_t<ConnectToAny<OutList>::template Type, CopyInNodes>;

private:
    __host_aicore__ constexpr static std::size_t GetMaxAliveNodeSize()
    {
        return maxAliveNodeInfo.aliveNode;
    }

    __host_aicore__ constexpr static std::size_t GetNonPersistInputSize()
    {
        return inSizeWoScalar;
    }

public:
    // Get GM count before first calc node skipping Cached CopyInBrc
    __host_aicore__ constexpr static std::size_t GetGMCountBeforeFirstCalcNode()
    {
        return GetCopyInCountBeforeFirstCalcNode<OpList>();
    }

    // Get Persist MTE2 number
    __host_aicore__ constexpr static std::size_t GetPersistMte2Num()
    {
        return 0;
    }

    // Get Persist MTE3 number
    __host_aicore__ constexpr static std::size_t GetPersistMte3Num()
    {
        return 0;
    }

    // Get Persist Temp buffer number
    __host_aicore__ constexpr static std::size_t GetPersistTempCalcBufNum()
    {
        return 0;
    }

    // Get temp calculation node (without CopyIn/Out/Cache) size
    // according to the scenario.
    __host_aicore__ constexpr static std::size_t GetTempCalcNodeSize()
    {
        return maxAliveNodeInfo.tempCalcNode;
    }

    // Get the count of CopyOut node before first CopyOut node
    // without considering Cache Nodes.
    // Normally it is 1.
    __host_aicore__ constexpr static std::size_t GetFirstCopyOutNodeGMCount()
    {
        constexpr auto maxAliveNodeSize = GetMaxAliveNodeSize();
        return maxAliveNodeSize > GetGMCountBeforeFirstCalcNode() ? 1 : 0;
    }

    // Get the count of L1/L2 MTE3 according to the scenario. (without Cache)
    __host_aicore__ constexpr static std::size_t GetLvl12Mte3Count()
    {
        constexpr auto allOutSize = Size_v<OutList>;
        constexpr auto persistMte3Size = GetPersistMte3Num();
        constexpr auto mte2AsMte3Size = Size_v<CopyInNodesLinkCopyOut>;
        return allOutSize - persistMte3Size - mte2AsMte3Size;
    }

    // Get the count of L1 temp calculation nodes (without CopyIn/Out/Cache).
    __host_aicore__ constexpr static std::size_t GetLvl1TmpSize()
    {
        constexpr auto maxAliveNodeSize = GetMaxAliveNodeSize();
        constexpr auto tempCalcNodeSize = GetTempCalcNodeSize();
        constexpr auto nonPersistInputSize = GetNonPersistInputSize();
        return tempCalcNodeSize > 0 ?
                   (maxAliveNodeSize > nonPersistInputSize ? maxAliveNodeSize - nonPersistInputSize : 0) :
                   0;
    }

    // Get the count of L0 temp calculation nodes (without CopyIn/Out/Cache).
    __host_aicore__ constexpr static std::size_t GetLvl0TmpSize()
    {
        constexpr auto maxAliveNodeSize = GetMaxAliveNodeSize();
        constexpr auto firstCopyOutNodeGMCount = GetFirstCopyOutNodeGMCount();
        return maxAliveNodeSize - (GetGMCountBeforeFirstCalcNode() + firstCopyOutNodeGMCount);
    }

    // Get the total count of L0 buffer (with Cache).
    __host_aicore__ constexpr static std::size_t GetBufferNumLevel0()
    {
        // 2 means ping-pong
        return GetMaxAliveNodeSize() + GetPersistTempCalcBufNum() + GetGMCountBeforeFirstCalcNode() +
               GetPersistMte2Num() * 2 + GetFirstCopyOutNodeGMCount() + GetPersistMte3Num() * 2;
    }

    // Get the total count of L1 buffer (with Cache).
    __host_aicore__ constexpr static std::size_t GetBufferNumLevel1()
    {
        // 2 means ping-pong
        return GetLvl1TmpSize() + GetPersistTempCalcBufNum() + inSizeWoScalar * 2 + GetLvl12Mte3Count() * 2 +
               GetPersistMte3Num() * 2;
    }

    // Get the total count of L2 buffer (with Cache).
    __host_aicore__ constexpr static std::size_t GetBufferNumLevel2()
    {
        // 2 means ping-pong
        return GetTempCalcNodeSize() + GetPersistTempCalcBufNum() + inSizeWoScalar * 2 + GetLvl12Mte3Count() * 2 +
               GetPersistMte3Num() * 2;
    }
};

} // namespace Atvoss::Tile

#endif // ATVOSS_GRAPH_NODE_H