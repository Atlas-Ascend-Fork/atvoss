/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ELE_WISE_H
#define BLOCK_ELE_WISE_H

#include <functional>
#include <type_traits>
#include "block/block_tensor.h"
#include "block/block_tail_tensor.h"
#include "common/tuple_tool.h"
#include "utils/buf_pool/loopbuf.h"
#include "utils/layout/layout.h"
#include "utils/layout/shape.h"
#include "block_schedule.h"
namespace Atvoss::EleWise {
struct UbAssign {
    uint32_t ubInCnt = 1;               // The expression requires the number of space to be input.
    uint32_t ubOutCnt = 1;              // The expression requires the number of space to be output.
    uint32_t ubTmpCnt = 0;              // The expression requires the number of temporary space.
    uint32_t eleNumSingleTensor = 1024; // The number of elements in a local tensor.
};
struct BlockConfig {
    uint32_t wholeLoop = 0;    // The number of entire tiles in the current block(excluding the tail tile).
    uint32_t tileCnt = 0;      // The number of elements processed when the current tile is the last one. Zero when the
                               // current tile is entire.
    uint32_t basicNum = 0;     // The number of elements processed by the entire tile.
    uint32_t totalElemCnt = 0; // Total number of elements processed in the current block.
    UbAssign ubAssign;         // UB space allocation strategy.
};

template <typename Shape>
struct BlockPolicy {
    using TileShape = Shape;
    uint32_t ubSizeMax = 190 * 1024; // Maximum UB space size.
    Shape tileShape{};
};

using TileShape = Atvoss::Shape<1, 32>;
static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicyDefault{190 * 1024, TileShape{}};

/*!
 * BlockBuilder: The task for a single block is broken down into multiple tiles, completing each data transfer and
 * computation.
 */
template <typename Compute, const auto& Policy = blockPolicyDefault, typename ScheduleCfg = BlockConfig,
          class Schedule = Block::DefaultSchedule<Compute, Policy, ScheduleCfg>>
class BlockBuilder {
public:
    using ScheduleClz = Schedule;
    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        Schedule schedule;
        schedule.Run(cfg, argTuple);
    }
};

} // namespace Atvoss::Block

#endif