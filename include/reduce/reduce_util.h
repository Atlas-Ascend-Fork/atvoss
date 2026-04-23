/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_REDUCE_REDUCE_UTIL_H_
#define ATVOSS_INCLUDE_REDUCE_REDUCE_UTIL_H_

#include <stdint.h>

namespace Atvoss {
namespace Reduce {

constexpr int32_t CONST0 = 0;
constexpr int32_t CONST1 = 1;
constexpr int32_t CONST2 = 2;
constexpr int32_t CONST3 = 3;
constexpr int32_t CONST4 = 4;
constexpr int32_t CONST5 = 5;
constexpr int32_t CONST6 = 6;
constexpr int32_t CONST7 = 7;
constexpr int32_t CONST8 = 8;
constexpr int32_t CONST9 = 9;
constexpr int32_t CONST10 = 10;

constexpr int32_t MAX_DIM = 9;
constexpr int32_t MAX_LOOP_DIM = (MAX_DIM + 1) / 2;
constexpr int32_t PATTERN_A = 10;
constexpr int32_t PATTERN_RA = 0;
constexpr int32_t PATTERN_AR = 1;
constexpr int32_t PATTERN_ARA = 2;
constexpr int32_t PATTERN_ARAR = 3;
constexpr int32_t PATTERN_ARARA = 4;
constexpr int32_t PATTERN_ARARAR = 5;
constexpr int32_t PATTERN_ARARARA = 6;
constexpr int32_t PATTERN_ARARARAR = 7;
constexpr int32_t PATTERN_ARARARARA = 8;

namespace __reducePattern {

template <uint32_t id, bool firstA, bool tailA, int32_t dim>
struct PatternConstInfo {
    constexpr static uint32_t ID = id;
    constexpr static bool FirstA = firstA;
    constexpr static bool TailA = tailA;
    constexpr static int32_t Dim = dim;
};

struct A : public PatternConstInfo<PATTERN_A, true, true, CONST1> {
};
struct RA : public PatternConstInfo<PATTERN_RA, false, true, CONST2> {
};
struct AR : public PatternConstInfo<PATTERN_AR, true, false, CONST2> {
};
struct ARA : public PatternConstInfo<PATTERN_ARA, true, true, CONST3> {
};
struct ARAR : public PatternConstInfo<PATTERN_ARAR, true, false, CONST4> {
};
struct ARARA : public PatternConstInfo<PATTERN_ARARA, true, true, CONST5> {
};
struct ARARAR : public PatternConstInfo<PATTERN_ARARAR, true, false, CONST6> {
};
struct ARARARA : public PatternConstInfo<PATTERN_ARARARA, true, true, CONST7> {
};
struct ARARARAR : public PatternConstInfo<PATTERN_ARARARAR, true, false, CONST8> {
};
struct ARARARARA : public PatternConstInfo<PATTERN_ARARARARA, true, true, CONST9> {
};

}  // namespace __reducePattern

}  // namespace Reduce
}  // namespace Atvoss

#endif  // ATVOSS_INCLUDE_REDUCE_REDUCE_UTIL_H_
