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
 * @file gm_offset.h
 * @brief Reduce 算子 Kernel GM 偏移计算模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_GM_OFFSET_H
#define ATVOSS_REDUCE_KERNEL_GM_OFFSET_H

#include "expression/expr_template.h"

namespace Atvoss::Reduce::KernelGmOffset {

/**
 * @brief 计算输入 GM 偏移量
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @param blockId 当前核 ID
 * @return GM 偏移量（元素数量）
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetForInput(ScheduleCfg& cfg, int32_t blockId)
{
    return 0;
}

/**
 * @brief 计算输出 GM 偏移量
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @param blockId 当前核 ID
 * @return GM 偏移量（元素数量）
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetForOutput(ScheduleCfg& cfg, int32_t blockId)
{
    return 0;
}

/**
 * @brief 根据参数用途计算 GM 偏移量
 * @tparam Usage 参数用途（输入/输出/输入输出）
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量（元素数量）
 */
template <Atvoss::ParamUsage Usage, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetByUsage(ScheduleCfg& cfg)
{
    int32_t blockId = AscendC::GetBlockIdx();
    if constexpr (Usage == Atvoss::ParamUsage::IN || Usage == Atvoss::ParamUsage::IN_OUT) {
        return CalGMOffsetForInput(cfg, blockId);
    } else {
        return CalGMOffsetForOutput(cfg, blockId);
    }
}

} // namespace Atvoss::Reduce::KernelGmOffset

#endif // ATVOSS_REDUCE_KERNEL_GM_OFFSET_H_