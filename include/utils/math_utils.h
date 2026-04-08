/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_UTILS_MATH_UTILS_H
#define ATVOSS_UTILS_MATH_UTILS_H

#include <type_traits>

namespace Atvoss {
template <typename T>
__aicore__ inline constexpr T CeilDiv(T a, T b)
{
    using type = typename std::conditional<
        sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline constexpr T CeilAlign(T a, T b)
{
    using type = typename std::conditional<
        sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline constexpr T FloorDiv(T a, T b)
{
    return a / b;
}

template <typename T>
__aicore__ inline constexpr T FloorAlign(T a, T b)
{
    using type = typename std::conditional<
        sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
    type res = static_cast<type>(a) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}
} // namespace Atvoss
#endif // ATVOSS_UTILS_MATH_UTILS_H
