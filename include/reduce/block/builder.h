/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file builder.h
 * @brief Reduce Block 构建模块
 */

#ifndef BLOCK_REDUCE_H
#define BLOCK_REDUCE_H

#include "reduce/block/policy.h"
#include "schedule.h"

namespace Atvoss::Reduce {

/**
 * @brief Block 构建器: 将单核任务分解为多个 tile, 完成数据搬运和计算调度
 * @tparam Compute 计算表达式类型 (如 ReduceSumCompute)
 * @tparam ArchTagcfg 架构标签 (默认 DAV_3510)
 * @tparam Policy Block 策略配置 (DefaultBlockPolicy)
 * @tparam ScheduleCfg Tiling 配置类型 (DefaultBlockConfig)
 * @tparam Schedule 调度类型 (DefaultBlockSchedule)
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
    /**
     * @brief Block 层执行入口
     * @tparam SchLoopInfo 调度循环信息
     * @tparam ArgTup 参数 tuple 类型 (GM 偏移后的地址)
     * @param cfg Block 层 Tiling 配置
     * @param argTuple 参数 tuple
     */
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
