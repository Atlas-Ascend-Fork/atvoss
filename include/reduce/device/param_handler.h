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
 * @brief Reduce 算子参数处理模块
 */

#ifndef ATVOSS_REDUCE_PARAM_HANDLER_H
#define ATVOSS_REDUCE_PARAM_HANDLER_H

#include <tuple>

#include "acl/acl.h"
#include "utils/utility.h"
#include "utils/tensor/device_tensor.h"

namespace Atvoss::Reduce {

/**
 * @brief 单参数转换：Tensor → 指针，标量保持原值
 * @tparam T 参数类型（标量或 DeviceTensor）
 * @param value 输入参数值
 * @return 转换后的值：标量原值，Tensor 返回其 GetPtr() 指针
 */
template <typename T>
auto TransformArgToPtr(T&& value)
{
    static_assert(
        std::is_scalar_v<std::decay_t<T>> || Util::IsTensor_v<std::decay_t<T>>,
        "TransformArgToPtr only accepts scalar types or Tensor specializations");

    // 编译时分支：标量直接返回，Tensor 调用 GetPtr()
    if constexpr (std::is_scalar_v<std::decay_t<T>>) {
        return std::forward<T>(value);
    } else {
        return value.GetPtr();
    }
}

/**
 * @brief 参数组转换：批量转换 tuple 中所有参数
 * @tparam ArgTup 参数 tuple 类型
 * @param argTuple 输入参数 tuple
 * @return 转换后的参数 tuple（所有 DeviceTensor 已转为指针）
 */
template <typename ArgTup>
auto TransformArgsToPtr(const ArgTup& argTuple)
{
    // 使用 std::apply 展开 tuple，对每个元素调用 TransformArgToPtr
    return std::apply(
        [](auto&&... elements) {
            return AscendC::Std::make_tuple(TransformArgToPtr(std::forward<decltype(elements)>(elements))...);
        },
        argTuple);
}

/**
 * @brief 构造单个参数：从用户输入提取并构造 DeviceTensor 或标量
 * @tparam ParamType 参数类型描述（包含 number, Type, usage 等信息）
 * @tparam ArgTup 参数 tuple 类型
 * @tparam TensorT Tensor 模板类型（如 DeviceTensor）
 * @param args 用户输入参数 tuple
 * @return 构造后的参数：DeviceTensor 或标量值
 */
template <typename ParamType, typename ArgTup, template <typename> class TensorT>
constexpr auto ConstructParam(ArgTup& args)
{
    // 获取 args 中对应位置的参数类型
    using ArgType = std::decay_t<std::tuple_element_t<ParamType::number - 1, ArgTup>>;
    
    if constexpr (
        std::is_scalar_v<typename ParamType::Type> && Util::IsSpecializationOf_v<Atvoss::Tensor, ArgType>) {
        // 构造 DeviceTensor：将 Host Tensor 转为设备端 Tensor
        return TensorT<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
    } else {
        // 直接返回：已经是标量或指针类型
        return typename std::decay_t<typename ParamType::Type>(std::get<ParamType::number - 1>(args));
    }
}

/**
 * @brief PrepareParams 的实现函数（带 index_sequence）
 * @tparam Params 参数类型列表（TypeList<Param<1>, Param<2>, ...>）
 * @tparam ArgTup 参数 tuple 类型
 * @tparam TensorT Tensor 模板类型
 * @tparam Is 索引序列
 * @param args 用户输入参数 tuple
 * @return 构造后的参数 tuple
 */
template <typename Params, typename ArgTup, template <typename> class TensorT, std::size_t... Ints>
constexpr auto PrepareParamsImpl(ArgTup& args, std::index_sequence<Ints...>)
{
    // 展开 Params 类型列表，构造所有参数
    return std::make_tuple(ConstructParam<Util::Get_t<Params, Ints>, ArgTup, TensorT>(args)...);
}

/**
 * @brief 准备所有参数：根据 Params 类型列表构造参数 tuple
 * @tparam Params 参数类型列表（TypeList<Param<1>, Param<2>, ...>）
 * @tparam TensorT Tensor 模板类型（如 DeviceTensor）
 * @tparam ArgTup 参数 tuple 类型
 * @param argTuple 用户输入参数 tuple
 * @return 构造后的参数 tuple：tuple<DeviceTensor<T1>, DeviceTensor<T2>, ...>
 */
template <typename Params, template <typename> class TensorT, typename ArgTup>
constexpr auto PrepareParams(ArgTup& argTuple)
{
    return PrepareParamsImpl<Params, ArgTup, TensorT>(
        argTuple, std::make_index_sequence<Util::Size_v<Params>>{});
}

/**
 * @brief 合并单个参数：决定使用构造参数还是原始参数
 * @tparam Params 参数类型列表
 * @tparam Index 当前处理的位置索引（从 0 开始）
 * @tparam ParamTup 构造后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @param params 构造后的参数 tuple（PrepareParams 结果）
 * @param args 原始参数 tuple（用户输入）
 * @return 合并后的单个参数值
 */
template <typename Params, std::size_t Index, typename ParamTup, typename ArgTup>
constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
{
    // 查找 Params 中是否有 number=Index+1 的参数
    constexpr auto pos = Util::Find_v<CheckVarNum<Index + 1>::template Checker, Params>;

    // - pos < Size_v：找到了，使用 params[pos]
    // - 否则：未定义，使用 args[Index]（额外参数）
    if constexpr (pos < Util::Size_v<Params>) {
        return std::get<pos>(params);
    } else {
        return std::get<Index>(args);
    }
}

/**
 * @brief ConvertArgs 的实现函数（带 index_sequence）
 * @tparam Params 参数类型列表
 * @tparam ParamTup 构造后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @tparam Is 索引序列
 * @param params 构造后的参数 tuple
 * @param args 原始参数 tuple
 * @return 合并后的参数 tuple
 */
template <typename Params, typename ParamTup, typename ArgTup, std::size_t... Ints>
constexpr auto ConvertArgsImpl(ParamTup& params, ArgTup& args, std::index_sequence<Ints...>)
{
    return std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
}

/**
 * @brief 合并参数：将构造参数与原始参数合并
 * @tparam Params 参数类型列表（定义了哪些参数需要构造）
 * @tparam ParamTup 构造后的参数 tuple 类型
 * @tparam ArgTup 原始参数 tuple 类型
 * @param params PrepareParams 的结果，包含构造后的 DeviceTensor
 * @param args 用户原始输入，包含所有参数和额外参数
 * @return 合并后的参数 tuple，长度与 args 相同
 */
template <typename Params, typename ParamTup, typename ArgTup>
auto ConvertArgs(ParamTup& params, ArgTup& args)
{
    return ConvertArgsImpl<Params>(params, args, std::make_index_sequence<std::tuple_size_v<ArgTup>>{});
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_PARAM_HANDLER_H