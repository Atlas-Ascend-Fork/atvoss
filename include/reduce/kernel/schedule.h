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
 * @file schedule.h
 * @brief Reduce 算子 Kernel 调度模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_SCHEDULE_H
#define ATVOSS_REDUCE_KERNEL_SCHEDULE_H

#include "param_handler.h"
#include "policy.h"
#include "reduce/utils/reduce_utils.h"

#if !defined(__ATVOSS_HOST_ONLY__)
template <typename T>
using KernelTensor = AscendC::GlobalTensor<T>;
#else
template <typename T>
struct KernelTensor {
    using Type = T;
};
#endif

namespace Atvoss::Reduce {

/**
 * @brief Kernel 调度基类：负责根据 Block ID 计算 GM 地址偏移，并调度 Block 执行
 * @tparam BlockOp Block 操作类型
 * @tparam Policy Kernel 策略配置
 * @tparam ScheduleCfg Tiling 配置类型
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker;
    using BlockTemplate = BlockOp;
    using ParamStruct = ScheduleCfg;
    using Params = typename Atvoss::ParamsDeriver<ExprMaker, KernelTensor>::Params;
    using BlockSched = typename BlockOp::ScheduleClz;
    using PreReduceParams = typename BlockSched::PreReduceParams;
    using PostReduceParams = typename BlockSched::PostReduceParams;

    /**
     * @brief 生成调度配置（Host 端调用）
     * @tparam Args 参数类型
     * @param arguments 算子输入参数
     * @param kernelParam 输出的 Kernel 配置参数
     * @return 是否生成成功
     */
    template <typename Args>
    static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
    {
        kernelParam.coreNum = 49;
        kernelParam.tilingKey = 2;
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseKernelSchedule() = default;

    /**
     * @brief Kernel 调度执行函数
     * @tparam SchLoopInfo loopInfo 操作类型（指针）
     * @tparam OpParam 操作参数类型
     * @tparam Args 可变参数类型
     * @param cfg Tiling 配置参数
     * @param args 可变参数（包含 GM 地址等）
     */
    template <auto SchLoopInfo, typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        typename BlockOp::ScheduleClz::ParamStruct blockConfig = cfg.blockParam;
        // in and out gm address
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        
        // 根据当前coreNum计算gm起始地址, 按 DAG 阶段自动区分 PreReduce / PostReduce
        auto params = KernelParamHandler::PrepareParamsByPhase<
            PreReduceParams, PostReduceParams>(cfg.kernelParam, argTuple);
        // args和params参数合并
        auto convertArgs = KernelParamHandler::ConvertArgs<Params>(params, argTuple);
 
        if constexpr (SchLoopInfo->loopRCount == 0) {
            BlockOp blockOp{};
            blockOp.template Run<SchLoopInfo>(blockConfig, convertArgs);
        } else {
            BlockOp blockOpStepOne{};
            blockOpStepOne.template Run<SchLoopInfo>(blockConfig, convertArgs);
            AscendC::SyncAll();
            TilingForGroup(cfg);
            BlockOp blockOpStepTwo{};
            blockOpStepTwo.template Run<SchLoopInfo>(blockConfig, convertArgs);
        }
    }

    /**
     * @brief Group Reduce 的 Tiling 处理
     * @tparam OpParam 操作参数类型
     * @param cfg Tiling 配置参数
     */
    template <typename OpParam>
    __aicore__ inline auto TilingForGroup(OpParam& cfg)
    {
        return;
    }
#endif
};

/**
 * @brief 默认 Kernel 调度类
 * @tparam BlockOp Block 操作类型
 * @tparam Policy Kernel 策略配置
 * @tparam ScheduleCfg Tiling 配置类型
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_KERNEL_SCHEDULE_H