/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_STRING_UTILS_H
#define ATVOSS_STRING_UTILS_H

#include <string>
#include <sstream>
#include <vector>

namespace Atvoss {

template <typename T>
std::string VectorToString(const std::vector<T>& s)
{
    std::stringstream ss;
    for (auto iter = s.begin(); iter != s.end(); ++iter) {
        ss << *iter;
        if (iter != s.end() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

template <typename T, typename U>
std::string VectorToString(const T* s, U size)
{
    std::stringstream ss;
    for (U i = 0; i < size; i++) {
        ss << s[i];
        if (i != size - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

} // namespace Atvoss

#endif // ATVOSS_STRING_UTILS_H