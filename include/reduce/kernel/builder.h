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
 * @brief Reduce 算子 Kernel 构建模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_BUILDER_H
#define ATVOSS_REDUCE_KERNEL_BUILDER_H

#include "policy.h"
#include "schedule.h"
#include "reduce/device/tilingdata.h"

namespace Atvoss::Reduce {

/**
 * @brief Kernel 构建器
 * @tparam BlockOp Block 操作类型
 * @tparam Policy Kernel 策略配置
 * @tparam ScheduleCfg Tiling 配置类型
 * @tparam Schedule Kernel 调度类型
 */
template <
    typename BlockOp, const auto& Policy = defaultKernelPolicy, typename ScheduleCfg = Atvoss::Reduce::ReduceTilingData,
    template <typename, const auto&, typename> class Schedule = DefaultKernelSchedule>
class KernelBuilder {
public:
    /**
     * @brief Kernel 操作参数结构体
     */
    struct OpParam {
        ScheduleCfg tilingData;
    };
    using ScheduleClz = Schedule<BlockOp, Policy, ScheduleCfg>;
    using ScheduleCfgClz = OpParam;
    using TilingData = ScheduleCfg;

#if !defined(__ATVOSS_HOST_ONLY__)
    /**
     * @brief Kernel 层执行函数
     * @tparam SchLoopInfo loopInfo 操作类型（指针）
     * @tparam OpParam 操作参数类型
     * @tparam Args 可变参数类型
     * @param cfg Tiling 配置参数
     * @param args 可变参数（包含 GM 地址等）
     */
    template <auto SchLoopInfo, typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        // AscendC::printf("kernel builder run: %d\n", 0x1);
        static_assert(
            (... && (std::is_scalar_v<Args> || Util::IsTensor_v<Args>)),
            "KernelBuilder::Run only accepts scalar or Tensor types");
        ScheduleClz schedule;
        schedule.template Run<SchLoopInfo>(cfg, args...);
    }
#endif
};
} // namespace Atvoss::Reduce
#endif // ATVOSS_REDUCE_KERNEL_BUILDER_H