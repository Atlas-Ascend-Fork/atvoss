/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_EXAMPLE_COMMON_H
#define Atvoss_EXAMPLE_COMMON_H
#include "acl/acl.h"

namespace {
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

static constexpr float REL_TOL = 1e-3f;
static constexpr float ABS_TOL = 1e-5f;

bool IsClose(float a, float b)
{
    const float eps = 1e-40f;
    float diff = std::abs(a - b);
    return (diff <= ABS_TOL) || (diff <= REL_TOL * std::max(std::abs(a), std::abs(b) + eps));
}

template <typename T>
bool VerifyResults(const std::vector<T> &golden, const std::vector<T> &output)
{
    for (int32_t i = 0; i < golden.size(); i++) {
        if (!IsClose(golden[i], output[i])) {
            printf("Accuracy verification failed! The expected value of element "
                   "in index [%d] is %f, but actual value is %f.\n",
                i,
                static_cast<float>(golden[i]),
                static_cast<float>(output[i]));
            return false;
        }
    }
    return true;
}
}  // namespace

#endif