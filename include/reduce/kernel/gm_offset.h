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

using Atvoss::Reduce::ReducePhase;
using Atvoss::Reduce::ReducePhase::PreReduce;
using Atvoss::Reduce::ReducePhase::PostReduce;

/**
 * @brief PreReduce 阶段 + 输入参数 的 GM 偏移
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalPreReduceInputOffset(ScheduleCfg& cfg)
{
    return 0;
}

/**
 * @brief PostReduce 阶段 + 输入参数 的 GM 偏移
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalPostReduceInputOffset(ScheduleCfg& cfg)
{
    return 0;
}

/**
 * @brief PreReduce 阶段 + 输出参数 的 GM 偏移
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalPreReduceOutputOffset(ScheduleCfg& cfg)
{
    return 0;
}

/**
 * @brief PostReduce 阶段 + 输出参数 的 GM 偏移
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalPostReduceOutputOffset(ScheduleCfg& cfg)
{
    return 0;
}

/**
 * @brief 统一 GM 偏移分发: 根据 Usage + Phase 选择正确的偏移公式
 * @tparam Usage 参数用途 (IN / OUT / IN_OUT)
 * @tparam Phase Reduce 阶段 (PreReduce / PostReduce)
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <Atvoss::ParamUsage Usage, ReducePhase Phase, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetImpl(ScheduleCfg& cfg)
{
    int32_t blockId = AscendC::GetBlockIdx();

    if constexpr (Usage == Atvoss::ParamUsage::IN) {
        if constexpr (Phase == ReducePhase::PreReduce) {
            return CalPreReduceInputOffset(cfg);
        } else {
            return CalPostReduceInputOffset(cfg);
        }
    } else if constexpr (Usage == Atvoss::ParamUsage::IN_OUT) {
        // IN_OUT: PreReduce 按输入算, PostReduce 按输出算
        if constexpr (Phase == ReducePhase::PreReduce) {
            return CalPreReduceInputOffset(cfg);
        } else {
            return CalPostReduceOutputOffset(cfg);
        }
    } else {
        // OUT
        if constexpr (Phase == ReducePhase::PreReduce) {
            return CalPreReduceOutputOffset(cfg);
        } else {
            return CalPostReduceOutputOffset(cfg);
        }
    }
}

/**
 * @brief 统一入口: 根据 ParamType 和 Phase 计算 GM 偏移
 * @tparam ParamType 参数类型 (Param<N, T, Usage, RN>)
 * @tparam Phase Reduce 阶段 (PreReduce / PostReduce)
 * @tparam ScheduleCfg Tiling 配置类型
 * @param cfg Kernel 配置参数
 * @return GM 偏移量 (元素数量)
 */
template <typename ParamType, ReducePhase Phase, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffset(ScheduleCfg& cfg)
{
    return CalGMOffsetImpl<ParamType::usage, Phase>(cfg);
}

} // namespace Atvoss::Reduce::KernelGmOffset

#endif // ATVOSS_REDUCE_KERNEL_GM_OFFSET_H_
