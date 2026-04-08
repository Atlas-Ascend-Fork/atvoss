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
    using ParamStruct = ScheduleCfg;
    static constexpr auto EXPRESSION = ToLinearizerExpr(ExprMaker{}.template Compute<KernelTensor>());
    using Expr = typename decltype(EXPRESSION)::Type;
    using Params = Atvoss::Params_t<Expr>;

    template <typename Args>
    static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
    {
        // Tiling处理
        auto& attrs = std::get<1>(arguments); // axes

        kernelParam.patternID = PATTERN_AR;
        kernelParam.coreNum = 64;
        kernelParam.groupR = 1;
        kernelParam.useNddma = 0;
        
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseKernelSchedule() = default;

    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        typename BlockOp::ScheduleClz::ParamStruct configBlock = cfg.blockParam;
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        uint32_t patternID =  kernelParam.patternID;
        uint64_t groupR = kernelParam.groupR;

        if (patternID == 9) {
            BlockOp blockOpEmpty{};
            blockOpEmpty.Run(configBlock, convertArgs);
        } else if (groupR == 1) {
            BlockOp blockOp{};
            blockOp.Run(configBlock, convertArgs);  
        } else {
            BlockOp blockOp1{};
            blockOp1.Run(configBlock, convertArgs);
            TilingForGroup(configBlock);
            BlockOp blockOp2{};
            blockOp2.Run(configBlock, convertArgs);
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
    __aicore__ inline auto PrepareParamsImpl(ScheduleCfg& config, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>>(config, args)...);
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline auto ConstructParam(ScheduleCfg& config, ArgTup& args)
    {
        auto arg = AscendC::Std::get<ParamType::number - 1>(args);
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DTypeTmp = typename ParamType::Type::PrimType;
            uint64_t offset = CalGMOffset(config);
            auto ptr = (uint64_t)(arg) + sizeof(DTypeTmp) * offset;
            return reinterpret_cast<__gm__ uint8_t*>(ptr);
        } else {
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

    __aicore__ inline auto CalGMOffset(ScheduleCfg& config)
    {
        if (AscendC::GetBlockIdx() < config.moreUnitCoreNum) {
            return AscendC::GetBlockIdx() * (config.unitNumPerCore * config.unitNum + config.unitNum);
        }
        return config.unitNumPerCore * config.unitNum * AscendC::GetBlockIdx() +
               config.moreUnitCoreNum * config.unitNum;
    }
#endif
};

template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_KERNEL_SCHEDULE_H_