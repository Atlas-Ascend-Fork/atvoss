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

struct ReduceSchLoopInfo {
    int32_t patternID;
    int32_t innerPatternID;
    int32_t loopACount;
    int32_t loopAAxis[MAX_LOOP_DIM];
    int32_t loopRCount;
    int32_t loopRAxis[MAX_DIM];
    int32_t loopInnerACount;
    int32_t loopInnerAAxis[MAX_DIM];
    int32_t loopInnerRCount;
    int32_t loopInnerRAxis[MAX_DIM];
};

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo0()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo1()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo2()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo3()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo4()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo5()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PatternID / CONST10,
                                           .innerPatternID = PatternID % CONST10,
                                           .loopACount = LoopARCount / CONST10,
                                           .loopAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopRCount = LoopARCount % CONST10,
                                           .loopRAxis = {DIM0, DIM2, DIM4, DIM6, DIM8, DIM1, DIM3, DIM5, DIM7},
                                           .loopInnerACount = LoopInnerARCount / CONST10,
                                           .loopInnerAAxis = {DIM0, DIM2, DIM4, DIM6, DIM8},
                                           .loopInnerRCount = LoopInnerARCount % CONST10,
                                           .loopInnerRAxis = {DIM1, DIM3, DIM5, DIM7}};
    return schInfo;
}

__aicore__ inline constexpr ReduceSchLoopInfo GetGroupSchLoopInfo()
{
    constexpr ReduceSchLoopInfo schInfo = {.patternID = PATTERN_RA,
                                           .innerPatternID = PATTERN_RA,
                                           .loopACount = CONST1,
                                           .loopAAxis = {DIM1},
                                           .loopRCount = CONST0,
                                           .loopRAxis = {},
                                           .loopInnerACount = CONST0,
                                           .loopInnerAAxis = {},
                                           .loopInnerRCount = CONST1,
                                           .loopInnerRAxis = {DIM0}};
    return schInfo;
}

template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo()
{
    if constexpr (LoopARCount / CONST10 == CONST0) {
        return GetSchLoopInfo0<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST1) {
        return GetSchLoopInfo1<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST2) {
        return GetSchLoopInfo2<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST3) {
        return GetSchLoopInfo3<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST4) {
        return GetSchLoopInfo4<PatternID, LoopARCount, LoopInnerARCount>();
    } else if constexpr (LoopARCount / CONST10 == CONST5) {
        return GetSchLoopInfo5<PatternID, LoopARCount, LoopInnerARCount>();
    }
}
} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_TILING_KEY_H