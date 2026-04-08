/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_
#define ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_
#include <functional>
#include "common/arch.h"
#include "utils/reduce_utils.h"

#if !defined(__ATVOSS_HOST_ONLY__)
template <typename T>
using KernelTensor = AscendC::GlobalTensor<T>;
#else
template <typename T>
struct KernelTensor {
    using Type = T;
};
#endif

namespace Atvoss::Reduce {

template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker;
    using ParamStruct = ScheduleCfg;    // DefaultKernelConfig
    static constexpr auto EXPRESSION = ToLinearizerExpr(ExprMaker{}.template Compute<KernelTensor>());
    using Expr = typename decltype(EXPRESSION)::Type;
    using Params = Atvoss::Params_t<Expr>;

    template <typename Args>
    static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
    {
        // Tiling处理
        auto& attrs = std::get<1>(arguments); // axes

        kernelParam.blockNum = 49;
        kernelParam.groupR = 1;
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseKernelSchedule() = default;

    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        typename BlockOp::ScheduleClz::ParamStruct blockConfig = cfg.blockParam;
        // in and out gm address
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);

        // 根据当前blockNum计算gm起始地址
        auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
        // args和params参数合并
        // 示例：用户传入额外参数
        // KernelSchedule::Run(cfg, in_gm, out_gm, axes_ptr, scale_value);
        // args... = (in_gm, out_gm, axes_ptr, scale_value)
        // Params = TypeList<Param<1>, Param<2>>  // 只有输入和输出
        // ConvertArgs 处理：
        // Index=0: 找到 Param<1> → 用 params[0] (偏移后的in_gm)
        // Index=1: 找到 Param<2> → 用 params[1] (偏移后的out_gm)
        // Index=2: 没找到 Param<3> → 用 args[2] (axes_ptr原值)
        // Index=3: 没找到 Param<4> → 用 args[3] (scale_value原值)
        // convertArgs = tuple<偏移in_gm, 偏移out_gm, axes_ptr, scale_value>
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        uint64_t groupR = cfg.kernelParam.groupR;
        if (groupR == 1) {
            BlockOp blockOp{};
            blockOp.Run(blockConfig, convertArgs);  
        } else {
            BlockOp blockOpStepOne{};
            blockOpStepOne.Run(blockConfig, convertArgs);
            AscendC::SyncAll();
            TilingForGroup(blockConfig);
            BlockOp blockOpStepTwo{};
            blockOpStepTwo.Run(blockConfig, convertArgs);
        }
    }

    __aicore__ inline auto TilingForGroup(ScheduleCfg& cfg)
    {
        return;
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto params =
            PrepareParamsImpl<Params>(cfg, argTuple, AscendC::Std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
        return params;
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(ScheduleCfg& cfg, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>>(cfg, args)...);
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline auto ConstructParam(ScheduleCfg& cfg, ArgTup& args)
    {   
        // 取出args中对应位置的GM地址
        auto arg = AscendC::Std::get<ParamType::number - 1>(args);
        // 张量入参，计算偏移
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DTypeTmp = typename ParamType::Type::PrimType;
            uint64_t offset = CalGMOffsetByUsage<ParamType::usage>(cfg);
            auto ptr = (uint64_t)(arg) + sizeof(DTypeTmp) * offset;
            return reinterpret_cast<__gm__ uint8_t*>(ptr);
        } else {    // 标量入参，直接返回或者解引用
            using argType = decltype(arg);
            if constexpr (std::is_pointer_v<argType>) {
                return *arg;
            } else {
                return arg;
            }
        }
    }

    template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
    __aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos = Atvoss::Util::Find_v<Atvoss::CheckVarNum<Index + 1>::template Checker, Params>;
        if constexpr (pos < Atvoss::Util::Size_v<Params>) {
            return AscendC::Std::get<pos>(params);
        } else {
            return AscendC::Std::get<Index>(args);
        }
    }

    template <typename Params, typename ParamTup, typename ArgTup, size_t... Ints>
    __aicore__ inline constexpr auto ConvertArgsImpl(
        ParamTup& params, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
    }

    template <typename Params, typename ParamTup, typename ArgTup>
    __aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args)
    {
        return ConvertArgsImpl<Params>(
            params, args, AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTup>>{});
    }

    template <Atvoss::ParamUsage Usage>
    __aicore__ inline uint64_t CalGMOffsetByUsage(ScheduleCfg& cfg)
    {
        int32_t blockId = AscendC::GetBlockIdx();
        if constexpr (Usage == Atvoss::ParamUsage::IN || Usage == Atvoss::ParamUsage::IN_OUT) {
            return CalGMOffsetForInput(cfg, blockId);
        } else {
            return CalGMOffsetForOutput(cfg, blockId);
        }
        return 0;
    }

    __aicore__ inline uint64_t CalGMOffsetForInput(ScheduleCfg& cfg, int32_t blockId)
    {
        return 0;
    }

    __aicore__ inline uint64_t CalGMOffsetForOutput(ScheduleCfg& cfg, int32_t blockId)
    {
        return 0;
    }
#endif
};

template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_