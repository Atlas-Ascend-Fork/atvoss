/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_DEV_PATTERNS_H
#define ATVOSS_DEV_PATTERNS_H
namespace Atvoss {
enum class Pattern
{
    AR,
    RA,
    AB,
    BA
};

enum class CastMode
{
    CAST_NONE = 0,
    CAST_RINT,
    CAST_FLOOR,
    CAST_CEIL,
    CAST_ROUND,
    CAST_TRUNC,
    CAST_ODD
};

enum class MemMngPolicy : uint8_t
{
    MANUAL = 0,
    AUTO
};

enum class MemLevel : uint8_t
{
    LEVEL_0 = 0,
    LEVEL_1,
    LEVEL_2
};
} // namespace Atvoss
#endif // ATVOSS_DEV_PATTERNS_H
