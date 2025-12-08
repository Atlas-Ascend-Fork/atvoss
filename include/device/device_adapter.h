/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEVICE_ADAPTER_H
#define Atvoss_DEVICE_ADAPTER_H
#include <functional>
#include "acl/acl.h"
#include "device_tensor.h"
#include "common/compile_info.h"
#include "utils/arguments/arguments.h"
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

void InitializeACL(aclrtContext &context, aclrtStream &stream, int32_t deviceId)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
}

void CleanACL(aclrtStream &stream, aclrtContext &context, int32_t deviceId)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x);
}
template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y);
}

template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y, z);
}
template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR m)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y, z, m);
}
template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR m, GM_ADDR n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y, z, m, n);
}
template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y, z, a, b, c);
}

namespace Atvoss {

template<class KernelOp, typename OpParam, typename ArgTup>
void LaunchKernelWithDataTuple(uint32_t blockNum, aclrtStream& stream, OpParam& cfg, ArgTup& argTuple)
{
    static constexpr auto size = std::tuple_size_v<ArgTup>;
    if constexpr (size == 1) { // the number of args is 1
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x);
    } else if constexpr (size == 2) { // the number of args is 2
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        uint8_t *y = std::get<1>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x, y);
    } else if constexpr (size == 3) { // the number of args is 3
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        uint8_t *y = std::get<1>(argTuple).GetPtr();
        uint8_t *z = std::get<2>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x, y, z);
    } else if constexpr (size == 4) { // the number of args is 4
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        uint8_t *y = std::get<1>(argTuple).GetPtr();
        uint8_t *z = std::get<2>(argTuple).GetPtr();
        uint8_t *m = std::get<3>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x, y, z, m);
    } else if constexpr (size == 5) { // the number of args is 5
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        uint8_t *y = std::get<1>(argTuple).GetPtr();
        uint8_t *z = std::get<2>(argTuple).GetPtr();
        uint8_t *m = std::get<3>(argTuple).GetPtr();
        uint8_t *n = std::get<4>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x, y, z, m, n);
    } else if constexpr (size == 6) { // the number of args is 6
        uint8_t *x = std::get<0>(argTuple).GetPtr();
        uint8_t *y = std::get<1>(argTuple).GetPtr();
        uint8_t *z = std::get<2>(argTuple).GetPtr();
        uint8_t *a = std::get<3>(argTuple).GetPtr();
        uint8_t *b = std::get<4>(argTuple).GetPtr();
        uint8_t *c = std::get<5>(argTuple).GetPtr();
        KernelCustom<KernelOp><<<blockNum, nullptr, stream>>>(cfg, x, y, z, a, b, c);
    }
}

/*!
 * DeviceAdapter: DeviceAdapter is a generic adapter that provides a host-side generic interface for different operator
 * invacation. It encapsulates Acl-related resource management internally and automatically handles kernel invocation.
 */
template <typename KernelOp>
class DeviceAdapter
{
public:
    using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;
    using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;
    using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;
    using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct;
    struct OpParam {
        KernelParamStruct kernelParam;
        BlockParamStruct blockParam;
    };

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
    int64_t Run(const Args& arguments)
    {
        auto expr = ExprMaker{}.template Compute<DeviceTensor>();
        using Expr = typename decltype(expr)::Type;
        using Params = Atvoss::Params_t<Expr>;
        using InParams = Atvoss::InParams_t<Expr>;
        using OutParams = Atvoss::OutParams_t<Expr>;

        // 1. Init Resource
        InitializeACL(context_, stream_, deviceId_);
        // 2. Init ShapeInfo
        using InputTuple = std::decay_t<decltype(std::get<0>(arguments))>;
        if constexpr (std::tuple_size_v<InputTuple> > 0) {
            shapeInfo_ = std::get<0>(std::get<0>(arguments)).shape_vector();
            for (auto dim : shapeInfo_) {
                if (dim == 0) {
                    printf("[ERROR]: [Atvoss][Device] Empty input tensor not supported (shape contains 0)!\n");
                    return -1;
                }
            }
        } else {
            printf("[ERROR]: [Atvoss][Device] No input tensor, shape info obtaining failed!\n");
            return -1;
        }
        auto argTuple = std::tuple_cat(std::get<0>(arguments), std::get<1>(arguments));
        // 2. prepare Param
        auto params = PrepareParams<Params>(argTuple);
        auto inParams = GetInParams<Params>(params);
        auto outParams = GetOutParams<Params>(params);
        CopyIn<InParams>(inParams, argTuple);
        // 3. calc dynamic param （tiling / worksapce）
        OpParam opParam;
        if (!CalcParam(opParam)) {
            printf("[ERROR]: [Atvoss][Device] CalcParam failed!\n");
            return -1;
        }
        // 4. kernel launch
        auto convertArgs = ConvertArgs<Params>(params, argTuple);
#if Atvoss_DEBUG_MODE == 2
        for(auto i = 0; i < 200; i++) { // 200 : profiling run times
             LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream_, opParam, convertArgs);
        }
#else
        LaunchKernelWithDataTuple<KernelOp>(opParam.kernelParam.blockNum, stream_, opParam, convertArgs);
#endif
        CHECK_ACL(aclrtSynchronizeStream(stream_));
        CopyOut<OutParams>(outParams, argTuple);
        CleanACL(stream_, context_, deviceId_);
        return 0;
    }

private:

    // calc kernel/block tiling and workspace.
    bool CalcParam(OpParam &opParam)
    {
        if (!KernelOp::ScheduleClz::MakeKernelParam(shapeInfo_, opParam.kernelParam)) {
            printf("[ERROR]: [Atvoss][Device] MakeKernelParam failed!\n");
            return false;
        }
        if (!BlockOp::ScheduleClz::MakeBlockParam(opParam.blockParam)){
            printf("[ERROR]: [Atvoss][Device] MakeBlockParam failed!\n");
            return false;
        }
        return true;
    }

    template <typename Params, typename ParamTup>
    auto GetInParams(ParamTup& params)
    {
        constexpr auto size = Util::TMP::Size_v<Params>;
        static_assert(size == std::tuple_size_v<ParamTup>, "[ERROR]: [Atvoss][Device] Size must match the number of element num in ParamTup!\n");
        return GetInParamsImpl<Params>(
            params, std::make_index_sequence<size>{});
    }

    template <typename Params, typename ParamTup>
    auto GetOutParams(ParamTup& params)
    {
        constexpr auto size = Util::TMP::Size_v<Params>;
        static_assert(size == std::tuple_size_v<ParamTup>, "[ERROR]: [Atvoss][Device] Size must match the number of element num in ParamTup!\n");
        return GetOutParamsImpl<Params>(
            params, std::make_index_sequence<size>{});
    }

    template <typename InParams, typename InParamTup, typename ArgTup>
    void CopyIn(InParamTup& inParams, ArgTup& args)
    {
        constexpr auto size = Util::TMP::Size_v<InParams>;
        static_assert(size == std::tuple_size_v<InParamTup>, "[ERROR]: [Atvoss][Device] Size must match the number of element num in InParamTup!\n");
        CopyInImpl<InParams>(inParams, args,
                                    std::make_index_sequence<size>{});
    }

    template <typename OutParams, typename OutParamTup, typename ArgTup>
    void CopyOut(OutParamTup& outParams, ArgTup& args)
    {
        constexpr auto size = Util::TMP::Size_v<OutParams>;
        static_assert(size == std::tuple_size_v<OutParamTup>, "[ERROR]: [Atvoss][Device] Size must match the number of element num in OutParamTup!\n");
        CopyOutImpl<OutParams>(outParams, args,
                                    std::make_index_sequence<size>{});
    }

    template <typename ParamType, typename ArgTup>
    constexpr auto ConstructParam(ArgTup& args)
    {
        return typename std::decay_t<typename ParamType::Type>(
            std::get<ParamType::number - 1>(args));
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    constexpr auto PrepareParamsImpl(ArgTup& args, std::index_sequence<Ints...>)
    {
        return std::make_tuple(
            ConstructParam<Util::TMP::Get_t<Params, Ints>>(args)...);
    }

    template <typename Params, typename ArgTup>
    constexpr auto PrepareParams(ArgTup& argTuple)
    {
        return PrepareParamsImpl<Params>(
            argTuple, std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename Params, std::size_t Index, typename ParamTup, typename ArgTup>
    constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos =
            Util::TMP::Find_v<Atvoss::CheckVarNum<Index + 1>::template Checker, Params>;
        if constexpr (pos < Util::TMP::Size_v<Params>) {
            return std::get<pos>(params);
        } else {
            return std::get<Index>(args);
        }
    }

    template <typename Params, typename ParamTup, typename ArgTup, std::size_t... Ints>
    constexpr auto ConvertArgsImpl(ParamTup& params, ArgTup& args,
                        std::index_sequence<Ints...>)
    {
        return std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
    }

    template <typename Params, typename ParamTup, typename ArgTup>
    auto ConvertArgs(ParamTup& params, ArgTup& args)
    {
        return ConvertArgsImpl<Params>(
            params, args,
            std::make_index_sequence<std::tuple_size_v<ArgTup>>{});
    }

    template <typename Params, std::size_t Index, Atvoss::ParamUsage... usages,
          typename ParamTup>
    constexpr auto GetOneParam(ParamTup& params)
    {
        using Param = Util::TMP::Get_t<Params, Index>;
        if constexpr (((Param::usage == usages) || ...)) {
            return std::forward_as_tuple(std::get<Index>(params));
        } else {
            return std::tuple<>{};
        }
    }

    template <typename Params, typename ParamTup, std::size_t... Ints>
    constexpr auto GetInParamsImpl(ParamTup& params, std::index_sequence<Ints...>)
    {
        return std::tuple_cat(
            GetOneParam<Params, Ints, Atvoss::ParamUsage::in, Atvoss::ParamUsage::in_out>(params)...);
    }

    template <typename Params, typename ParamTup, std::size_t... Ints>
    constexpr auto GetOutParamsImpl(ParamTup& params, std::index_sequence<Ints...>)
    {
        return std::tuple_cat(
            GetOneParam<Params, Ints, Atvoss::ParamUsage::out, Atvoss::ParamUsage::in_out>(params)...);
    }

    template <typename InParams, std::size_t Index, typename T, typename ArgTup>
    void CopyInOneParam(T& param, ArgTup& args)
    {
        using Param = Util::TMP::Get_t<InParams, Index>;
        param.CopyIn();
    }

    template <typename InParams, typename InParamTup, typename ArgTup,
            std::size_t... Ints>
    void CopyInImpl(InParamTup& inParams, ArgTup& args,
                    std::index_sequence<Ints...>)
    {
        (CopyInOneParam<InParams, Ints>(std::get<Ints>(inParams), args), ...);
    }

    template <typename OutParams, std::size_t Index, typename T, typename ArgTup>
    void CopyOutOneParam(T& param, ArgTup& args)
    {
        using Param = Util::TMP::Get_t<OutParams, Index>;
        param.CopyOut();
    }

    template <typename OutParams, typename OutParamTup, typename ArgTup,
            std::size_t... Ints>
    void CopyOutImpl(OutParamTup& outParams, ArgTup& args,
                    std::index_sequence<Ints...>)
    {
        (CopyOutOneParam<OutParams, Ints>(std::get<Ints>(outParams), args), ...);
    }

private:
    aclrtContext context_ = nullptr;	
    int32_t deviceId_ = 0;	
    aclrtStream stream_ = nullptr;
    std::vector<uint32_t> shapeInfo_;
};

} // namespace Atvoss::Device
#endif