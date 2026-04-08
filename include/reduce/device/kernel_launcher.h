/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file kernel_launcher.h
 * @brief Reduce 算子 Kernel 启动模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_LAUNCHER_H
#define ATVOSS_REDUCE_KERNEL_LAUNCHER_H

#include <tuple>

#include "acl/acl.h"
#include "common/platform_info.h"
#include "param_handler.h"
#include "reduce/utils/reduce_tiling_key.h"

namespace Atvoss::Reduce {

/**
 * @brief Kernel 内部包装器：展开参数 tuple 并调用 KernelOp::Run
 * @tparam SchLoopInfo loopInfo 操作类型
 * @tparam KernelOp Kernel 操作类型（如 KernelBuilder<BlockOp, Policy, Config>）
 * @tparam OpParam 操作参数类型（包含 kernelParam 和 blockParam）
 * @tparam ArgTuple 参数 tuple 类型
 * @tparam Is 索引序列（编译时生成）
 * @param cfg Tiling 配置参数（包含 kernel 和 block 的调度信息）
 * @param args 参数 tuple（已转换为指针的参数）
 */
template <auto SchLoopInfo, class KernelOp, typename OpParam, typename ArgTuple, std::size_t... Is>
__aicore__ inline void KernelWrapper(OpParam& cfg, ArgTuple args, AscendC::Std::index_sequence<Is...>)
{   
    AscendC::printf("PatternID: %d", SchLoopInfo->patternID);
    KernelOp op;
    op.template Run<SchLoopInfo>(cfg, AscendC::Std::get<Is>(args)...);
}

/**
 * @brief Kernel 入口函数：AI Core 执行的起点
 * @tparam ReduceTilingKey Kernel tilingKey类型
 * @tparam KernelOp Kernel 操作类型
 * @tparam OpParam 操作参数类型
 * @tparam ArgTuple 参数 tuple 类型
 * @param cfg Tiling 配置参数（从 Host 端传入）
 * @param args 参数 tuple（已转换为指针）
 */
template <typename ReduceTilingKey, class KernelOp, typename OpParam, typename ArgTuple>
__global__ __aicore__ void KernelCustom(OpParam cfg, ArgTuple args)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // AscendC::printf("PatternID: %d", ReduceTilingKey::PATTERN_ID);
    // AscendC::printf("LoopARCount: %d", ReduceTilingKey::LOOP_AR_COUNT);
    // AscendC::printf("LoopInnerARCount: %d", ReduceTilingKey::LOOP_INNER_AR_COUNT);
    constexpr static ReduceSchLoopInfo SchLoopInfo = GetSchLoopInfo<ReduceTilingKey::PATTERN_ID,
        ReduceTilingKey::LOOP_AR_COUNT, ReduceTilingKey::LOOP_INNER_AR_COUNT>();
    // 展开 tuple 参数
    KernelWrapper<&SchLoopInfo, KernelOp, OpParam, ArgTuple>(
        cfg, args, AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTuple>>{});
}

/**
 * @brief Kernel 启动函数：Host 端启动 Kernel 的接口
 * @tparam ReduceTilingKey Kernel tilingKey类型
 * @tparam KernelOp Kernel 操作类型
 * @tparam OpParam 操作参数类型
 * @tparam ArgTup 参数 tuple 类型
 * @param coreNum 启动的 AI Core 核数（从 Tiling 计算得到）
 * @param stream ACL Stream（用于异步执行）
 * @param cfg Tiling 配置参数
 * @param argTuple 参数 tuple（可能包含 DeviceTensor）
 */
template <typename ReduceTilingKey, class KernelOp, typename OpParam, typename ArgTup>
void LaunchKernel(uint32_t coreNum, aclrtStream stream, OpParam& cfg, const ArgTup& argTuple)
{
    // Step 1: 参数转换
    // 将 DeviceTensor 转换为指针，生成新的参数 tuple
    auto transformedArgs = TransformArgsToPtr(argTuple);
    
    // Step 2: 启动 Kernel
    KernelCustom<ReduceTilingKey, KernelOp, OpParam><<<coreNum, nullptr, stream>>>(cfg, transformedArgs);
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_KERNEL_LAUNCHER_H