/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_PARAM_TRANSFORM_H
#define ATVOSS_REDUCE_PARAM_TRANSFORM_H

#include <tuple>

#include "device_tensor.h"
#include "utils/utility.h"

namespace Atvoss::Reduce {

template <typename ParamType, typename ArgTup, template <typename> class TensorT>
constexpr auto ConstructParam(ArgTup& args)
{
    using ArgType = std::decay_t<std::tuple_element_t<ParamType::number - 1, ArgTup>>;
    if constexpr (
        std::is_scalar_v<typename ParamType::Type> && Atvoss::Util::IsSpecializationOf_v<Atvoss::Tensor, ArgType>) {
        return TensorT<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
    } else {
        return typename std::decay_t<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
    }
}

template <typename Params, typename ArgTup, template <typename> class TensorT, std::size_t... Ints>
constexpr auto PrepareParamsImpl(ArgTup& args, std::index_sequence<Ints...>)
{
    return std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>, ArgTup, TensorT>(args)...);
}

template <typename Params, template <typename> class TensorT, typename ArgTup>
constexpr auto PrepareParams(ArgTup& argTuple)
{
    return PrepareParamsImpl<Params, ArgTup, TensorT>(
        argTuple, std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
}

template <typename Params, std::size_t Index, typename ParamTup, typename ArgTup>
constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
{
    constexpr auto pos = Atvoss::Util::Find_v<Atvoss::CheckVarNum<Index + 1>::template Checker, Params>;
    if constexpr (pos < Atvoss::Util::Size_v<Params>) {
        return std::get<pos>(params);
    } else {
        return std::get<Index>(args);
    }
}

template <typename Params, typename ParamTup, typename ArgTup, std::size_t... Ints>
constexpr auto ConvertArgsImpl(ParamTup& params, ArgTup& args, std::index_sequence<Ints...>)
{
    return std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
}

template <typename Params, typename ParamTup, typename ArgTup>
auto ConvertArgs(ParamTup& params, ArgTup& args)
{
    return ConvertArgsImpl<Params>(params, args, std::make_index_sequence<std::tuple_size_v<ArgTup>>{});
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_PARAM_TRANSFORM_H