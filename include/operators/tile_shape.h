/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_TILE_LAYOUT_H
#define ATVOSS_TILE_LAYOUT_H
#include "utils/layout/layout.h"

#if !defined(__ATVOSS_HOST_ONLY__)
#define atvoss_std AscendC::Std
#else
#define atvoss_std std
#endif

namespace Atvoss::Ele::Tile {

enum class Operation
{
    Unary = 1, // Unary operation, return uint32_t
    Binary,    // Binary operation that returns an array of uint32_t
    Ternary,   // Ternary operation (reserve)
};

template <typename T, typename = void>
struct LayoutImpl : std::false_type {
    using Layout = atvoss_std::tuple<>;
};

template <typename T>
struct LayoutImpl<T, std::void_t<typename T::layout>> : std::true_type {
    using Layout = typename T::layout;
};

template <typename T>
using Layout_t = typename LayoutImpl<T>::Layout;

template <typename T, typename = void>
struct ShapeImpl {
    using type = typename T::TileShape;
};

template <typename T>
struct ShapeImpl<T, std::void_t<typename std::enable_if<(atvoss_std::tuple_size<Layout_t<T>>::value > 0)>::type>> {
    using type = typename T::TileShape;
};

template <typename T>
using Shape_t = typename ShapeImpl<T>::type;

template <typename T>
using Stride_t = atvoss_std::conditional_t<
    (atvoss_std::tuple_size<Layout_t<T>>::value > 1),
    typename atvoss_std::tuple_element<atvoss_std::tuple_size<Layout_t<T>>::value - 1, Layout_t<T>>::type,
    atvoss_std::tuple<>>;

template <typename ShapeType>
struct ShapeSize {
    static constexpr size_t value = ShapeType::size::value;
};

template <typename... ShapeType>
struct ShapeSize<atvoss_std::tuple<ShapeType...>> {
    static constexpr size_t value = sizeof...(ShapeType);
};

template <size_t N, typename T1>
static constexpr inline int32_t GetTotal(uint32_t eleCntInTensor = 1, int defaultSize = 1)
{
    constexpr size_t tupleSize = T1::size::value;
    if constexpr (tupleSize == 0) {
        return eleCntInTensor;
    } else if constexpr (N > tupleSize) {
        return eleCntInTensor;
    } else {
        using TValueType = typename T1::template get_type<N>;
        int32_t TValue = TValueType::value;
        defaultSize = defaultSize * TValue;
        if constexpr (N < tupleSize - 1) {
            return GetTotal<N + 1, T1>(defaultSize);
        }
        return defaultSize;
    }
}

/*!
 * \brief Calculation layout's total length.
 * \param[in] eleCntInTensor, Init-val.
 * \param[in] defaultSize, Init-val.
 * \return int32_t, Shape's total
 */
template <size_t N, typename T>
static constexpr inline int32_t GetTotalElement(uint32_t eleCntInTensor = 1, int defaultSize = 1)
{
    using shape = typename T::TileShape;
    return GetTotal<N, shape>(eleCntInTensor, defaultSize);
}

#if !defined(__ATVOSS_HOST_ONLY__)
template <Operation op = Operation::Unary, typename ArgTup>
__aicore__ inline static constexpr auto GetShape(ArgTup& args)
{
    auto blockTensor = atvoss_std::get<0>(args);
    using LayoutType = typename decltype(blockTensor)::LayoutType;
    if constexpr (std::is_same_v<LayoutType, Atvoss::Layout::VariableRankExtents<1>>) {
        if constexpr (op == Operation::Unary) {
            return blockTensor.GetLayout().GetUnaryShape();
        }
        if constexpr (op == Operation::Binary) {
            return blockTensor.GetLayout().GetBinaryShape();
        }
    } else {
        if constexpr (op == Operation::Unary) {
            return LayoutType::ShapeType::UNARY_SHAPE;
        }
        if constexpr (op == Operation::Binary) {
            return LayoutType::ShapeType::BINARY_SHAPE;
        }
    }
}
#endif

} // namespace Atvoss::Ele::Tile
#endif