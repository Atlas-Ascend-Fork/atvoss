/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_GRAPH_NODE_H
#define ATVOSS_REDUCE_GRAPH_NODE_H

#include <cstddef>
#include "common.h"
#include "utils/utility.h"

namespace Atvoss::Reduce::Graph {

using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Difference_t;
using Atvoss::Util::Filter_t;
using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::Unique_t;

struct DagMaxAliveInfo {
    std::size_t maxAliveNodes = 0;
    std::size_t maxTempCalcNodes = 0;

    constexpr DagMaxAliveInfo() = default;

    constexpr DagMaxAliveInfo(std::size_t alive, std::size_t temp) : maxAliveNodes(alive), maxTempCalcNodes(temp)
    {}
};

template <typename ExprList, typename AliveNodes = TypeList<>, std::size_t pos = 0>
struct MaxAliveNodeHelper {
    using CurrentOpAssign = Get_t<ExprList, pos>;
    using LHS = typename CurrentOpAssign::LhsType;
    using RHS = typename CurrentOpAssign::RhsType;
    using CurrentInputs = typename ExtractInputs<RHS>::Type;
    using RemoveRetNode = Difference_t<AliveNodes, TypeList<LHS>>;
    using CurrentAllNodes = Concatenate_t<TypeList<LHS>, CurrentInputs>;
    using NewAliveNodes = Unique_t<Concatenate_t<RemoveRetNode, CurrentAllNodes>>;
    using TempCalcNodes = Filter_t<IsLocalVar, NewAliveNodes>;
    using ReleasableNodes = Filter_t<CanRelease<ExprList, pos>::template Type, NewAliveNodes>;
    using NextAliveNodes = Difference_t<NewAliveNodes, ReleasableNodes>;
};

template <typename ExprList, typename AliveNodes = TypeList<>, std::size_t pos = 0>
constexpr DagMaxAliveInfo MaxAliveNode()
{
    if constexpr (pos < Size_v<ExprList>) {
        using Helper = MaxAliveNodeHelper<ExprList, AliveNodes, pos>;

        constexpr auto nextInfo = MaxAliveNode<ExprList, typename Helper::NextAliveNodes, pos + 1>();

        return DagMaxAliveInfo(
            Max(Size_v<typename Helper::NewAliveNodes>, nextInfo.maxAliveNodes),
            Max(Size_v<typename Helper::TempCalcNodes>, nextInfo.maxTempCalcNodes));
    } else {
        return DagMaxAliveInfo(Size_v<AliveNodes>, Size_v<Filter_t<IsLocalVar, AliveNodes>>);
    }
}

template <typename ExprList>
struct DagNodeInfo {
private:
    using AllParams = Params_t<ExprList>;
    using InParams = Filter_t<IsInParam, AllParams>;
    using OutParams = Filter_t<IsOutParam, AllParams>;

    constexpr static std::size_t inSize = Size_v<InParams>;
    constexpr static std::size_t outSize = Size_v<OutParams>;

public:
    constexpr static auto maxAliveNodeInfo = MaxAliveNode<ExprList>();

    __host_aicore__ constexpr static std::size_t GetInputNum()
    {
        return inSize;
    }

    __host_aicore__ constexpr static std::size_t GetOutputNum()
    {
        return outSize;
    }

    __host_aicore__ static constexpr std::size_t GetTempCalcNodeSize()
    {
        return maxAliveNodeInfo.maxTempCalcNodes;
    }
};

} // namespace Atvoss::Reduce::Graph
#endif
