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

#include "reduce/block/policy.h"
#include "schedule.h"

namespace Atvoss::Reduce {

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
    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        ScheduleClz schedule;
        schedule.template Run<SchLoopInfo>(cfg, argTuple);
    }
#endif
};

} // namespace Atvoss::Reduce

#endif