/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_COMMON_ARCH_H
#define ATVOSS_COMMON_ARCH_H

#if (defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102))
#define _ATVOSS_ARCH35_ 1
#else
#define _ATVOSS_ARCH35_ 0
#endif

namespace Atvoss::Arch {

struct DAV_3510 {
    static constexpr uint32_t CORE_NUM = 56;
    static constexpr uint32_t UB_SIZE = 240 * 1024;
};

} // namespace Atvoss::Arch

#endif // ATVOSS_COMMON_ARCH_H
