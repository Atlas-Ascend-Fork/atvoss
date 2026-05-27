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
 * @file policy.h
 * @brief Reduce 算子 Kernel 策略配置模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_POLICY_H
#define ATVOSS_REDUCE_KERNEL_POLICY_H

#include <cstdint>

namespace Atvoss::Reduce {

/**
 * @brief 多核 Tiling 策略枚举
 */
enum class MultiCoreTilingPolicy
{
    MemoryFirst = 0U,   // For bandwidth bound operators, prioritize fully utilizing the UB
    ComputeFirst,       // For compute bound operators, prioritize fully utilizing all cores
};

/**
 * @brief Kernel 默认策略配置
 */
struct DefaultKernelPolicy {
    MultiCoreTilingPolicy multiCoreTilingPolicy = MultiCoreTilingPolicy::MemoryFirst;
};

static constexpr DefaultKernelPolicy defaultKernelPolicy{MultiCoreTilingPolicy::MemoryFirst};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_KERNEL_POLICY_H