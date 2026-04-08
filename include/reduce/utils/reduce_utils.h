/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVOSS_REDUCE_UTILS_H
#define ATVOSS_REDUCE_UTILS_H

namespace Atvoss::Reduce {

constexpr uint32_t DIM0 = 0;
constexpr uint32_t DIM1 = 1;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t DIM4 = 4;
constexpr uint32_t DIM5 = 5;
constexpr uint32_t DIM6 = 6;
constexpr uint32_t DIM7 = 7;
constexpr uint32_t DIM8 = 8;
constexpr uint32_t DIM9 = 9;

constexpr uint32_t CONST0 = 0;
constexpr uint32_t CONST1 = 1;
constexpr uint32_t CONST2 = 2;
constexpr uint32_t CONST3 = 3;
constexpr uint32_t CONST4 = 4;
constexpr uint32_t CONST5 = 5;
constexpr uint32_t CONST6 = 6;
constexpr uint32_t CONST7 = 7;
constexpr uint32_t CONST8 = 8;
constexpr uint32_t CONST9 = 9;
constexpr uint32_t CONST10 = 10;

constexpr uint32_t MAX_DIM = 9;
constexpr uint32_t MAX_LOOP_DIM = (MAX_DIM + 1) / 2;
constexpr uint32_t PATTERN_A = 10;
constexpr uint32_t PATTERN_EMPTY = 9;
constexpr uint32_t PATTERN_RA = 0;
constexpr uint32_t PATTERN_AR = 1;
constexpr uint32_t PATTERN_ARA = 2;
constexpr uint32_t PATTERN_ARAR = 3;
constexpr uint32_t PATTERN_ARARA = 4;
constexpr uint32_t PATTERN_ARARAR = 5;
constexpr uint32_t PATTERN_ARARARA = 6;
constexpr uint32_t PATTERN_ARARARAR = 7;
constexpr uint32_t PATTERN_ARARARARA = 8;




} // namespace Atvoss::Reduce
#endif // ATVOSS_REDUCE_UTILS_H