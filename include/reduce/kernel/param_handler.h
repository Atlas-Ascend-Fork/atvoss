/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file param_handler.h
 * @brief Reduce 算子 Kernel 参数处理模块
 */

#ifndef ATVOSS_REDUCE_KERNEL_PARAM_HANDLER_H_
#define ATVOSS_REDUCE_KERNEL_PARAM_HANDLER_H_

#include "gm_offset.h"
#include "reduce/utils/reduce_utils.h"

namespace Atvoss::Reduce::KernelParamHandler {

using Atvoss::Reduce::ReducePhase;

/**
 * @brief 构造单个参数（张量计算偏移，标量直接返回）
 * @tparam Phase Reduce 阶段 (PreReduce / PostReduce)
 * @tparam ParamType 参数类型
 * @tparam ScheduleCfg Tiling 配置类型
 * @tparam ArgTup 参数 tuple 类型
 * @param cfg Kernel 配置参数
 * @param args 参数 tuple
 * @return 构造后的参数
 */
template <ReducePhase Phase, typename ParamType, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto ConstructParam(ScheduleCfg& cfg, ArgTup& args)
{
    auto arg = AscendC::Std::get<ParamType::number - 1>(args);
    if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
        using DType = typename ParamType::Type::PrimType;
        uint64_t offset = KernelGmOffset::CalGMOffset<ParamType, Phase>(cfg);
        auto ptr = (uint64_t)(arg) + sizeof(DType) * offset;
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

/**
 * @brief 准备参数列表实现（展开索引序列）
 * @tparam Phase Reduce 阶段 (PreReduce / PostReduce)
 * @tparam Params 参数类型列表
 * @tparam ScheduleCfg Tiling 配置类型
 * @tparam ArgTup 参数 tuple 类型
 * @tparam Is 索引序列
 * @param cfg Kernel 配置参数
 * @param args 参数 tuple
 * @return 偏移后的参数列表
 */
template <ReducePhase Phase, typename Params, typename ScheduleCfg, typename ArgTup, std::size_t... Is>
__aicore__ inline auto PrepareParamsImpl(ScheduleCfg& cfg, ArgTup& args, AscendC::Std::index_sequence<Is...>)
{
    return AscendC::Std::make_tuple(ConstructParam<Phase, Atvoss::Util::Get_t<Params, Is>>(cfg, args)...);
}

/**
 * @brief 准备参数列表（计算 GM 地址偏移）
 * @tparam Phase Reduce 阶段 (PreReduce / PostReduce)
 * @tparam Params 参数类型列表
 * @tparam ScheduleCfg Tiling 配置类型
 * @tparam ArgTup 参数 tuple 类型
 * @param cfg Kernel 配置参数
 * @param argTuple 参数 tuple
 * @return 偏移后的参数列表
 */
template <ReducePhase Phase, typename Params, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
{
    return PrepareParamsImpl<Phase, Params>(cfg, argTuple,
        AscendC::Std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
}

/**
 * @brief 转换单个参数（优先使用偏移后的参数，否则使用原参数）
 * @tparam Params 参数类型列表
 * @tparam Index 参数索引
 * @tparam ParamTup 偏移后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @param params 偏移后的参数 tuple
 * @param args 原始参数 tuple
 * @return 转换后的参数
 */
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

/**
 * @brief 转换参数列表实现（展开索引序列）
 * @tparam Params 参数类型列表
 * @tparam ParamTup 偏移后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @tparam Is 索引序列
 * @param params 偏移后的参数 tuple
 * @param args 原始参数 tuple
 * @return 转换后的参数 tuple
 */
template <typename Params, typename ParamTup, typename ArgTup, size_t... Is>
__aicore__ inline constexpr auto ConvertArgsImpl(ParamTup& params, ArgTup& args, AscendC::Std::index_sequence<Is...>)
{
    return AscendC::Std::make_tuple(ConvertOneArg<Params, Is>(params, args)...);
}

/**
 * @brief 转换参数列表（合并偏移参数和原参数）
 * @tparam Params 参数类型列表
 * @tparam ParamTup 偏移后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @param params 偏移后的参数 tuple
 * @param args 原始参数 tuple
 * @return 转换后的参数 tuple
 */
template <typename Params, typename ParamTup, typename ArgTup>
__aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args)
{
    return ConvertArgsImpl<Params>(params, args, 
        AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTup>>{});
}

} // namespace Atvoss::Reduce::KernelParamHandler

#endif // ATVOSS_REDUCE_KERNEL_PARAM_HANDLER_H