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
 * @file reduce_tiling_key.h
 * @brief reduce tilingKey配置模块
 */

#ifndef ATVOSS_REDUCE_TILING_KEY_H
#define ATVOSS_REDUCE_TILING_KEY_H

#include "reduce_utils.h"

namespace Atvoss::Reduce {

/**
 * @brief Kernel 启动策略模板
 * @tparam PatternID Reduce 模式标识 Pattern * 10 + InnerPattern
 * @tparam LoopARCount 核间A/R轴循环次数 LoopACount * 10 + LoopRCount
 * @tparam LoopInnerARCount 核内A/R轴循环次数 LoopInnerACount * 10 + LoopInnerRCount
 */
template<uint32_t PatternID = 0, uint32_t LoopARCount = 0, uint32_t LoopInnerARCount = 0>
struct ReduceTilingKey {
    static constexpr uint32_t PATTERN_ID = PatternID;
    static constexpr uint32_t LOOP_AR_COUNT = LoopARCount;
    static constexpr uint32_t LOOP_INNER_AR_COUNT = LoopInnerARCount;
};

using DefaultReduceTilingKey = ReduceTilingKey<0, 0, 0>;
} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_TILING_KEY_H