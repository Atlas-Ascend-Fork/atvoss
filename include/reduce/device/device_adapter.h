/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_DEVICE_ADAPTER_H
#define ATVOSS_REDUCE_DEVICE_ADAPTER_H

#include <functional>

#include "acl/acl.h"
#include "common/compile_info.h"
#include "device_tensor.h"
#include "tiling.h"
#include "utils/arguments/arguments.h"
#include "utils/utility.h"

namespace Atvoss::Reduce {

template <class KernelOp, typename OpParam, typename ArgTuple, std::size_t... Is>
__aicore__ inline void KernelWrapper(OpParam& cfg, ArgTuple args, AscendC::Std::index_sequence<Is...>)
{
    KernelOp op;
    op.Run(cfg, AscendC::Std::get<Is>(args)...);
}

template <class KernelOp, typename OpParam, typename ArgTuple>
__global__ __aicore__ void KernelCustom(OpParam cfg, ArgTuple args)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelWrapper<KernelOp, OpParam, ArgTuple>(
        cfg, args, AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTuple>>{});
}

template <typename T>
auto TransformArgs(T&& value)
{
    static_assert(
        std::is_scalar_v<std::decay_t<T>> || Util::IsTensor_v<std::decay_t<T>>,
        "TransformArgs only accepts scalar types or Tensor specializations");
    if constexpr (std::is_scalar_v<std::decay_t<T>>) {
        return std::forward<T>(value);
    } else {
        return value.GetPtr();
    }
}

template <class KernelOp, typename OpParam, typename ArgTup>
void LaunchKernelWithDataTuple(uint32_t blockNum, aclrtStream& stream, OpParam& cfg, const ArgTup& argTuple)
{
    static constexpr auto size = std::tuple_size_v<ArgTup>;
    auto transformedArgs = std::apply(
        [](auto&&... elements) {
            return AscendC::Std::make_tuple(TransformArgs(std::forward<decltype(elements)>(elements))...);
        },
        argTuple);

    KernelCustom<KernelOp, OpParam><<<blockNum, nullptr, stream>>>(cfg, transformedArgs);
}

template <typename KernelOp>
class DeviceAdapter {
public:
    using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;
    using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;
    using OpParam = typename KernelOp::ScheduleCfgClz;

    template <typename T>
    using Tensor = DeviceTensor<T>;

    DeviceAdapter() {}

    template <typename Args>
    int64_t Run(const Args& arguments, aclrtStream stream = nullptr)
    {
        auto expr = ToLinearizerExpr(ExprMaker{}.template Compute<Tensor>());
        using Expr = typename decltype(expr)::Type;
        using Params = Atvoss::Params_t<Expr>;

        auto argTuple = std::get<0>(arguments);

        auto params = PrepareParams<Params>(argTuple);

        OpParam opParam;
        if (!CalculateTiling<KernelOp>(arguments, opParam)) {
            printf("[ERROR]: [Atvoss][Reduce][Device] CalculateTiling failed!\n");
            return -1;
        }

        auto convertArgs = ConvertArgs<Params>(params, argTuple);

#if ATVOSS_DEBUG_MODE == 2
        for (auto i = 0; i < 200; i++) {
            LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
        }
#else
        LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
#endif

        return 0;
    }

private:
    template <typename ParamType, typename ArgTup>
    constexpr auto ConstructParam(ArgTup& args)
    {
        using ArgType = std::decay_t<std::tuple_element_t<ParamType::number - 1, ArgTup>>;
        if constexpr (
            std::is_scalar_v<typename ParamType::Type> && Atvoss::Util::IsSpecializationOf_v<Atvoss::Tensor, ArgType>) {
            return Tensor<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
        } else {
            return typename std::decay_t<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
        }
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    constexpr auto PrepareParamsImpl(ArgTup& args, std::index_sequence<Ints...>)
    {
        return std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>>(args)...);
    }

    template <typename Params, typename ArgTup>
    constexpr auto PrepareParams(ArgTup& argTuple)
    {
        return PrepareParamsImpl<Params>(argTuple, std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
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
};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_DEVICE_ADAPTER_H