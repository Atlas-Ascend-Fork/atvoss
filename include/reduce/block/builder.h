/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_REDUCE_H
#define BLOCK_REDUCE_H

#include "utils/layout/shape.h"
#include "common/arch.h"
#include "schedule.h"

namespace Atvoss::Reduce {

constexpr size_t MAX_DIMaaa = 8;
struct DefaultBlockConfig {
    uint32_t patternID;                  // Origin pattern
    uint64_t ubFactorA = 0;              // Ub split factor of A-axis
    uint64_t ubFactorR = 0;              // Ub split factor of R-axis
    uint64_t tileACntPerCore = 0;        // Number of TileA count processed per core
    uint64_t tileRCntPerCore = 0;        // Number of TileR count processed per core
    uint32_t basicBlock = 0;             // The block size of elements processed by the entire tile.
    uint32_t resultBlock = 0;            // The result block size of reduced-elements processed by the entire tile.
    uint32_t useNddma = 0;               // MTE2 whether to use NDDMA
    uint64_t shape[MAX_DIMaaa] = {0, 0, 0, 0, 0, 0, 0, 0};          // shape of tensor
    uint64_t srcStride[MAX_DIMaaa] = {0, 0, 0, 0, 0, 0, 0, 0};      // input stride of tensor
    uint64_t dstStride[MAX_DIMaaa] = {0, 0, 0, 0, 0, 0, 0, 0};      // output stride of tensor
};

template <typename Shape>
struct DefaultBlockPolicy {
    using TileShape = Shape;        // TileShape contains two dimensions: the first dimension in order is TileA,
                                    // and the second dimension is TileB.
    static constexpr bool EnableBinaryAcc = true;    // Whether to enable binary accumulation, enabled by default.
    Shape tileShape{};
    Atvoss::MemMngPolicy memPolicy = Atvoss::MemMngPolicy::AUTO;
};

static constexpr uint32_t DEFAULT_TILE_A = 32;
static constexpr uint32_t DEFAULT_TILE_R = 32;
using TileShape = Atvoss::Shape<DEFAULT_TILE_A, DEFAULT_TILE_R>;
static constexpr Atvoss::Reduce::DefaultBlockPolicy<TileShape> defaultBlockPolicy{TileShape{}};

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
    using ScheduleClz = Schedule<Compute, Policy, ScheduleCfg, ArchTagcfg>;
    using ScheduleCfgClz = ScheduleCfg;
#if !defined(__ATVOSS_HOST_ONLY__)
    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        ScheduleClz schedule;
        schedule.Run(cfg, argTuple);
    }
#endif
};

} // namespace Atvoss::Reduce

#endif