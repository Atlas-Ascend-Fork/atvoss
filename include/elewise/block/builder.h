/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ELEWISE_H
#define BLOCK_ELEWISE_H

#include "utils/layout/shape.h"
#include "common/arch.h"
#include "schedule.h"

namespace Atvoss::Ele {
struct DefaultBlockConfig {
    uint32_t wholeLoop = 0;    // The number of entire tiles in the current block(excluding the tail tile).
    uint32_t tileCnt = 0;      // The number of elements processed when the current tile is the last one. Zero when the
                               // current tile is entire.
    uint32_t basicNum = 0;     // The number of elements processed by the entire tile.
    uint32_t totalElemCnt = 0; // Total number of elements processed in the current block.
};

template <typename Shape>
struct DefaultBlockPolicy {
    using TileShape = Shape;
    Shape tileShape{};
    Atvoss::MemMngPolicy memPolicy = Atvoss::MemMngPolicy::AUTO;
};

static constexpr uint32_t DEFAULT_SHAPE = 4096;
using TileShape = Atvoss::Shape<1, DEFAULT_SHAPE>;
static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> defaultBlockPolicy{TileShape{}};

/*!
 * BlockBuilder: The task for a single block is broken down into multiple tiles, completing each data transfer and
 * computation.
 */
template <
    typename Compute, typename ArchTagcfg = Atvoss::Arch::DAV_3510, const auto& Policy = defaultBlockPolicy,
    typename ScheduleCfg = DefaultBlockConfig,
    template <typename, const auto&, typename, typename> class Schedule = DefaultBlockSchedule>
class BlockBuilder {
public:
    using ScheduleCfgClz = ScheduleCfg;
    using ScheduleClz = Schedule<Compute, Policy, ScheduleCfg, ArchTagcfg>;
    using BlockTileShape = typename ScheduleClz::TileShape;
#if !defined(__ATVOSS_HOST_ONLY__)
    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        ScheduleClz schedule;
        schedule.Run(cfg, argTuple);
    }
#endif
};

} // namespace Atvoss::Ele

#endif