/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVOSS_ARGUMENTS_H
#define ATVOSS_ARGUMENTS_H

#include <tuple>
#include <string>
#include <type_traits>
#include <vector>
#include "utils/utility.h"

namespace Atvoss {

template <typename Key, typename Value>
struct AttrMap {
    Key key;
    Value value;
};

template <typename Key, typename Value>
constexpr auto MakeAttr(Key key, Value value)
{
    return AttrMap<Key, Value>{key, value};
}

template <typename Tuple1, typename Tuple2>
struct ConcatTuples;

template <typename... T1, typename... T2>
struct ConcatTuples<std::tuple<T1...>, std::tuple<T2...>> {
    using type = std::tuple<T1..., T2...>;
};

template <typename Tuple1, typename Tuple2>
using ConcatTuplesT = typename ConcatTuples<Tuple1, Tuple2>::type;

template <typename InputOutputTuple>
struct InputOutputCollector {
    InputOutputTuple inputOutput;

    template <typename... NewInputOutput>
    constexpr auto AddInputOutput(NewInputOutput&... newInputOutput) const
    {
        auto newTuple = std::tuple_cat(inputOutput, std::forward_as_tuple(newInputOutput...));
        using NewInputOutputTuple = ConcatTuplesT<InputOutputTuple, std::tuple<NewInputOutput&...>>;
        return InputOutputCollector<NewInputOutputTuple>{newTuple};
    }
};

template <typename AttrsTuple>
struct AttrCollector {
    AttrsTuple attrs;

    template <typename Key, typename Value>
    constexpr auto AddAttr(Key key, Value value) const
    {
        auto newTuple = std::tuple_cat(attrs, std::make_tuple(MakeAttr(key, value)));
        using NewAttrsTuple = ConcatTuplesT<AttrsTuple, std::tuple<AttrMap<Key, Value>>>;
        return AttrCollector<NewAttrsTuple>{newTuple};
    }
};

template <typename InOutCollector, typename AttrCollector>
struct ArgumentsBuilderImpl {
    InOutCollector inOutCollector;
    AttrCollector attrCollector;

    ArgumentsBuilderImpl(InOutCollector ic, AttrCollector ac) : inOutCollector(ic), attrCollector(ac)
    {}

    template <typename... NewInputOutput>
    constexpr auto inputOutput(NewInputOutput&... newInputOutput) const
    {
        auto newInOutCollector = inOutCollector.AddInputOutput(newInputOutput...);
        return ArgumentsBuilderImpl<decltype(newInOutCollector), AttrCollector>{newInOutCollector, attrCollector};
    }

    template <typename Key, typename Value>
    constexpr auto attr(Key key, Value value) const
    {
        auto newAttrCollector = attrCollector.AddAttr(key, value);
        return ArgumentsBuilderImpl<InOutCollector, decltype(newAttrCollector)>{inOutCollector, newAttrCollector};
    }

    constexpr auto build() const
    {
        return std::make_tuple(inOutCollector.inputOutput, attrCollector.attrs);
    }
};

template <typename InOutCollector, typename AttrCollector>
ArgumentsBuilderImpl(InOutCollector, AttrCollector) -> ArgumentsBuilderImpl<InOutCollector, AttrCollector>;

struct ArgumentsBuilder {
    template <typename... InitialInputOutput>
    constexpr auto inputOutput(InitialInputOutput&&... inputOutput) const
    {
        // 参数必须为Tensor和非指针类的scalar类型
        static_assert(
            (... && !std::is_pointer_v<InitialInputOutput>), "Pointer types are not allowed in inputOutput parameters");
        static_assert(
            (... && (Util::IsSpecializationOf_v<Atvoss::Tensor, std::decay_t<InitialInputOutput>> ||
                     std::is_scalar_v<std::decay_t<InitialInputOutput>>)),
            "Only Atvoss::Tensor and scalar types are allowed in inputOutput parameters");

        auto initialInputOutput = std::forward_as_tuple(inputOutput...);
        auto inOutCollector = InputOutputCollector<std::tuple<InitialInputOutput&...>>{initialInputOutput};
        auto attrCollector = AttrCollector<std::tuple<>>{};

        return ArgumentsBuilderImpl{inOutCollector, attrCollector};
    }

    template <typename Key, typename Value>
    constexpr auto attr(Key key, Value value) const
    {
        auto inOutCollector = InputOutputCollector<std::tuple<>>{};
        auto initialAttrs = std::make_tuple(MakeAttr(key, value));
        auto attrCollector = AttrCollector<std::tuple<AttrMap<Key, Value>>>{initialAttrs};

        return ArgumentsBuilderImpl{inOutCollector, attrCollector};
    }
};
} // namespace Atvoss
#endif // ATVOSS_ARGUMENTS_H