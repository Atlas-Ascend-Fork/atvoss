/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_DEVICE_ADAPTER_H
#define ATVOSS_DEVICE_ADAPTER_H
#include <functional>
#include "acl/acl.h"
#include "device_tensor.h"
#include "common/compile_info.h"
#include "utils/utility.h"
#include "utils/arguments/arguments.h"
#include "tiling.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

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

namespace Atvoss {

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

/*!
 * DeviceAdapter: DeviceAdapter is a generic adapter that provides a host-side generic interface for different operator
 * invacation. It encapsulates Acl-related resource management internally and automatically handles kernel invocation.
 */
template <typename KernelOp>
class DeviceAdapter {
public:
    using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;
    using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;
    // using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;
    // using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct;
    using OpParam = typename KernelOp::ScheduleCfgClz;
    template <typename T>
    using Tensor = DeviceTensor<T>;

    /*!
     * \brief The constructor interface of DeviceAdapter class.
     */
    DeviceAdapter() {};

    /*!
     * \brief The external running interface of DeviceAdapter mainly completes resource initialization,
     *        data transfer between host and device, and kernel launch.
     * \param[in] arguments
     */
    template <typename Args>
    int64_t Run(const Args& arguments, aclrtStream stream = nullptr)
    {
        auto expr = ToLinearizerExpr(ExprMaker{}.template Compute<Tensor>());
        using Expr = typename decltype(expr)::Type;
        using Params = Atvoss::Params_t<Expr>;

        auto argTuple = std::get<0>(arguments);
        // 1. prepare Param
        auto params = PrepareParams<Params>(argTuple);

        // 2. calc dynamic param （tiling / worksapce）
        OpParam opParam;
        if (!CalculateTiling<KernelOp>(arguments, opParam)) {
            printf("[ERROR]: [Atvoss][Device] CalcParam failed!\n");
            return -1;
        }
        // 3. kernel launch
        auto convertArgs = ConvertArgs<Params>(params, argTuple);
#if ATVOSS_DEBUG_MODE == 2
        for (auto i = 0; i < 200; i++) { // 200 : profiling run times
            LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
        }
#else
        LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
#endif
        return 0;
    }

private:
    // calc kernel/block tiling and workspace.
    template <typename Args>
    bool CalcParam(const Args& arguments, OpParam& opParam)
    {
        if (!KernelOp::ScheduleClz::MakeScheduleConfig(arguments, opParam.kernelParam)) {
            printf("[ERROR]: [Atvoss][Device] MakeScheduleConfig for kernel failed!\n");
            return false;
        }
        if (!BlockOp::ScheduleClz::MakeScheduleConfig(arguments, opParam.kernelParam, opParam.blockParam)) {
            printf("[ERROR]: [Atvoss][Device] MakeScheduleConfig for block failed!\n");
            return false;
        }
        return true;
    }

    template <typename Params, typename ParamTup>
    auto GetInParams(ParamTup& params)
    {
        constexpr auto size = Atvoss::Util::Size_v<Params>;
        static_assert(
            size == std::tuple_size_v<ParamTup>,
            "[ERROR]: [Atvoss][Device] Size must match the number of element num in ParamTup!\n");
        return GetInParamsImpl<Params>(params, std::make_index_sequence<size>{});
    }

    template <typename Params, typename ParamTup>
    auto GetOutParams(ParamTup& params)
    {
        constexpr auto size = Atvoss::Util::Size_v<Params>;
        static_assert(
            size == std::tuple_size_v<ParamTup>,
            "[ERROR]: [Atvoss][Device] Size must match the number of element num in ParamTup!\n");
        return GetOutParamsImpl<Params>(params, std::make_index_sequence<size>{});
    }

    template <typename InParams, typename InParamTup, typename ArgTup>
    void CopyIn(InParamTup& inParams, ArgTup& args)
    {
        constexpr auto size = Atvoss::Util::Size_v<InParams>;
        static_assert(
            size == std::tuple_size_v<InParamTup>,
            "[ERROR]: [Atvoss][Device] Size must match the number of element num in InParamTup!\n");
        CopyInImpl<InParams>(inParams, args, std::make_index_sequence<size>{});
    }

    template <typename OutParams, typename OutParamTup, typename ArgTup>
    void CopyOut(OutParamTup& outParams, ArgTup& args)
    {
        constexpr auto size = Atvoss::Util::Size_v<OutParams>;
        static_assert(
            size == std::tuple_size_v<OutParamTup>,
            "[ERROR]: [Atvoss][Device] Size must match the number of element num in OutParamTup!\n");
        CopyOutImpl<OutParams>(outParams, args, std::make_index_sequence<size>{});
    }

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

    template <typename Params, std::size_t Index, Atvoss::ParamUsage... usages, typename ParamTup>
    constexpr auto GetOneParam(ParamTup& params)
    {
        using Param = Atvoss::Util::Get_t<Params, Index>;
        if constexpr (((Param::usage == usages) || ...)) {
            return std::forward_as_tuple(std::get<Index>(params));
        } else {
            return std::tuple<>{};
        }
    }

    template <typename Params, typename ParamTup, std::size_t... Ints>
    constexpr auto GetInParamsImpl(ParamTup& params, std::index_sequence<Ints...>)
    {
        return std::tuple_cat(GetOneParam<Params, Ints, Atvoss::ParamUsage::IN, Atvoss::ParamUsage::IN_OUT>(params)...);
    }

    template <typename Params, typename ParamTup, std::size_t... Ints>
    constexpr auto GetOutParamsImpl(ParamTup& params, std::index_sequence<Ints...>)
    {
        return std::tuple_cat(
            GetOneParam<Params, Ints, Atvoss::ParamUsage::OUT, Atvoss::ParamUsage::IN_OUT>(params)...);
    }

    template <typename InParams, std::size_t Index, typename T, typename ArgTup>
    void CopyInOneParam(T& param, ArgTup& args)
    {
        using Param = Atvoss::Util::Get_t<InParams, Index>;
        param.CopyIn();
    }

    template <typename InParams, typename InParamTup, typename ArgTup, std::size_t... Ints>
    void CopyInImpl(InParamTup& inParams, ArgTup& args, std::index_sequence<Ints...>)
    {
        (CopyInOneParam<InParams, Ints>(std::get<Ints>(inParams), args), ...);
    }

    template <typename OutParams, std::size_t Index, typename T, typename ArgTup>
    void CopyOutOneParam(T& param, ArgTup& args)
    {
        using Param = Atvoss::Util::Get_t<OutParams, Index>;
        param.CopyOut();
    }

    template <typename OutParams, typename OutParamTup, typename ArgTup, std::size_t... Ints>
    void CopyOutImpl(OutParamTup& outParams, ArgTup& args, std::index_sequence<Ints...>)
    {
        (CopyOutOneParam<OutParams, Ints>(std::get<Ints>(outParams), args), ...);
    }
};

} // namespace Atvoss
#endif