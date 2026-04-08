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
 * @file tiling.h
 * @brief Reduce 算子 Tiling 参数计算模块
 */

#ifndef ATVOSS_REDUCE_TILING_H
#define ATVOSS_REDUCE_TILING_H

namespace Atvoss::Reduce {

/**
 * @brief 计算 Reduce 算子的 Tiling 配置参数
 * @tparam KernelOp Kernel 操作类型（如 KernelBuilder<...>）
 * @tparam Args 用户输入参数类型（通常为 tuple 类型）
 * @param args 用户输入参数，包含输入 Tensor、输出 Tensor、Reduce axes 等
 * @param cfg 输出参数，Kernel 和 Block 的配置结构体
 * @return bool 计算是否成功，true 表示成功，false 表示失败
 */
template <typename KernelOp, typename Args>
bool CalculateTiling(const Args& args, typename KernelOp::ScheduleCfgClz& cfg)
{
    // 获取 Block 操作类型
    using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;

    // Step 1: 计算 Kernel 层配置
    if (!KernelOp::ScheduleClz::MakeScheduleConfig(args, cfg.kernelParam)) {
        printf("[ERROR]: [Atvoss][Reduce][Device] MakeScheduleConfig for kernel failed!\n");
        return false;
    }

    // Step 2: 计算 Block 层配置
    if (!BlockOp::ScheduleClz::MakeScheduleConfig(args, cfg.kernelParam, cfg.blockParam)) {
        printf("[ERROR]: [Atvoss][Reduce][Device] MakeScheduleConfig for block failed!\n");
        return false;
    }

    return true;
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_TILING_H