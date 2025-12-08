/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef Atvoss_ARGUMENTS_H
#define Atvoss_ARGUMENTS_H

#include <tuple>
#include <string>
#include <type_traits>
#include <vector>

namespace Atvoss {

template<typename Key, typename Value>
struct AttrMap {
    Key key;
    Value value;
};

template<typename Key, typename Value>
constexpr auto MakeAttr(Key key, Value value) {
    return AttrMap<Key, Value>{key, value};
}

template<typename Tuple1, typename Tuple2>
struct ConcatTuples;

template<typename... T1, typename... T2>
struct ConcatTuples<std::tuple<T1...>, std::tuple<T2...>> {
    using type = std::tuple<T1..., T2...>;
};

template<typename Tuple1, typename Tuple2>
using ConcatTuplesT = typename ConcatTuples<Tuple1, Tuple2>::type;

template<typename InputsTuple>
struct InputCollector {
    InputsTuple inputs;
    
    template<typename... NewInputs>
    constexpr auto AddInputs(NewInputs&... newInputs) const {
        auto newTuple = std::tuple_cat(inputs, std::forward_as_tuple(newInputs...));
        using NewInputsTuple = ConcatTuplesT<InputsTuple, std::tuple<NewInputs& ...>>;
        return InputCollector<NewInputsTuple>{newTuple};
    }
};

template<typename OutputsTuple>
struct OutputCollector {
    OutputsTuple outputs;
    
    template<typename... NewOutputs>
    constexpr auto AddOutputs(NewOutputs&... newOutputs) const {
        auto newTuple = std::tuple_cat(outputs, std::forward_as_tuple(newOutputs...));
        using NewOutputsTuple = ConcatTuplesT<OutputsTuple, std::tuple<NewOutputs& ...>>;
        return OutputCollector<NewOutputsTuple>{newTuple};
    }
};

template<typename AttrsTuple>
struct AttrCollector {
    AttrsTuple attrs;
    
    template<typename Key, typename Value>
    constexpr auto AddAttr(Key key, Value value) const {
        auto newTuple = std::tuple_cat(attrs, std::make_tuple(MakeAttr(key, value)));
        using NewAttrsTuple = ConcatTuplesT<AttrsTuple, std::tuple<AttrMap<Key, Value>>>;
        return AttrCollector<NewAttrsTuple>{newTuple};
    }
};

template<typename InCollector, typename OutCollector, typename AttrCollector>
struct ArgumentsBuilderImpl {
    InCollector inputCollector;
    OutCollector outputCollector;
    AttrCollector attrCollector;
    
    ArgumentsBuilderImpl(InCollector ic, OutCollector oc, AttrCollector ac) 
        : inputCollector(ic), outputCollector(oc), attrCollector(ac) {}
    
    template<typename... NewInputs>
    constexpr auto input(NewInputs&... newInputs) const {
        auto newInputCollector = inputCollector.AddInputs(newInputs...);
        return ArgumentsBuilderImpl<decltype(newInputCollector), OutCollector, AttrCollector>{
            newInputCollector, outputCollector, attrCollector
        };
    }
    
    template<typename... NewOutputs>
    constexpr auto output(NewOutputs&... newOutputs) const {
        auto newOutputCollector = outputCollector.AddOutputs(newOutputs...);
        return ArgumentsBuilderImpl<InCollector, decltype(newOutputCollector), AttrCollector>{
            inputCollector, newOutputCollector, attrCollector
        };
    }
    
    template<typename Key, typename Value>
    constexpr auto attr(Key key, Value value) const {
        auto newAttrCollector = attrCollector.AddAttr(key, value);
        return ArgumentsBuilderImpl<InCollector, OutCollector, decltype(newAttrCollector)>{
            inputCollector, outputCollector, newAttrCollector
        };
    }
    
    constexpr auto build() const {
        return std::make_tuple(
            inputCollector.inputs,
            outputCollector.outputs,
            attrCollector.attrs
        );
    }
};

template<typename InCollector, typename OutCollector, typename AttrCollector>
ArgumentsBuilderImpl(InCollector, OutCollector, AttrCollector) -> ArgumentsBuilderImpl<InCollector, OutCollector, AttrCollector>;

struct ArgumentsBuilder {
    template<typename... InitialInputs>
    constexpr auto input(InitialInputs&... inputs) const {
        auto initialInputs = std::forward_as_tuple(inputs...);
        auto inputCollector = InputCollector<std::tuple<InitialInputs& ...>>{initialInputs};
        auto outputCollector = OutputCollector<std::tuple<>>{};
        auto attrCollector = AttrCollector<std::tuple<>>{};
        
        return ArgumentsBuilderImpl{
            inputCollector, 
            outputCollector, 
            attrCollector
        };
    }
    
    template<typename... InitialOutputs>
    constexpr auto output(InitialOutputs&... outputs) const {
        auto initialOutputs = std::forward_as_tuple(outputs...);
        auto inputCollector = InputCollector<std::tuple<>>{};
        auto outputCollector = OutputCollector<std::tuple<InitialOutputs& ...>>{initialOutputs};
        auto attrCollector = AttrCollector<std::tuple<>>{};
        
        return ArgumentsBuilderImpl{
            inputCollector, 
            outputCollector, 
            attrCollector
        };
    }
    
    template<typename Key, typename Value>
    constexpr auto attr(Key key, Value value) const {
        auto inputCollector = InputCollector<std::tuple<>>{};
        auto outputCollector = OutputCollector<std::tuple<>>{};
        auto initialAttrs = std::make_tuple(MakeAttr(key, value));
        auto attrCollector = AttrCollector<std::tuple<AttrMap<Key, Value>>>{initialAttrs};
        
        return ArgumentsBuilderImpl{
            inputCollector, 
            outputCollector, 
            attrCollector
        };
    }
};
} // namespace Atvoss
#endif  // Atvoss_ARGUMENTS_H