/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_UTILS_SHAPE_H_
#define ATVOSS_INCLUDE_UTILS_SHAPE_H_

#include <tuple>

namespace Atvoss {

template <int... a>
class Shape {
public:
    using Types = std::tuple<std::integral_constant<size_t, a>...>;

    template <size_t N>
    using get_type = typename std::tuple_element_t<N, Types>;

    using size = std::integral_constant<size_t, sizeof...(a)>;
};

} // namespace Atvoss

#endif // ATVOSS_INCLUDE_UTILS_SHAPE_H_
