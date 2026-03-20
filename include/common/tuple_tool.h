/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TUPLE_UTIL_H
#define ATVOSS_TUPLE_UTIL_H

namespace Atvoss {
template <class T>
struct ArgSize {
    constexpr static int arg = 0;
};

template <class... Args>
struct ArgSize<AscendC::Std::tuple<Args...>> {
    constexpr static int arg = sizeof...(Args);
};

namespace TupleUtils {
template <typename T>
__aicore__ inline void CalOneEleOffset(T& tensor, uint32_t offset)
{
    tensor = tensor[offset];
}

template <typename T, std::size_t... Ints>
__aicore__ inline void CalOffsetImpl(T& argTuple, uint32_t offset, AscendC::Std::index_sequence<Ints...>)
{
    (CalOneEleOffset(AscendC::Std::get<Ints>(argTuple), offset), ...);
}

template <typename T>
__aicore__ inline void CalOffset(T& argTuple, uint32_t offset)
{
    constexpr uint32_t tupeSize = ArgSize<T>::arg;
    CalOffsetImpl(argTuple, offset, AscendC::Std::make_index_sequence<tupeSize>{});
}
} // namespace TupleUtils
} // namespace Atvoss
#endif