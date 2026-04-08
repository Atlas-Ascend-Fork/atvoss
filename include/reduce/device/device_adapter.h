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
 * @file device_adapter.h
 * @brief Reduce 算子设备适配器模块
 */

#ifndef ATVOSS_REDUCE_DEVICE_ADAPTER_H
#define ATVOSS_REDUCE_DEVICE_ADAPTER_H

#include "acl/acl.h"
#include "kernel_launcher.h"
#include "param_handler.h"
#include "utils/tiling.h"

namespace Atvoss::Reduce {

/**
 * @brief Reduce 算子设备适配器
 * @tparam KernelOp Kernel 操作类型（如 KernelBuilder<...>）
 */
template <typename KernelOp>
class DeviceAdapter {
public:
    DeviceAdapter() = default;

    /**
     * @brief 表达式生成器类型
     */
    using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;

    /**
     * @brief 操作参数类型，包含两部分配置：kernelParam和blockParam
     */
    using OpParam = typename KernelOp::ScheduleCfgClz;

    /**
     * @brief Tensor 类型别名
     * @tparam T 数据类型
     */
    template <typename T>
    using Tensor = DeviceTensor<T>;

    /**
     * @brief 执行 Reduce 算子
     * @tparam Args 用户输入参数类型
     * @param arguments 用户输入参数，通常为 tuple 类型
     * @param stream ACL Stream，用于异步执行（可选）
     * @return int64_t 执行结果：0 表示成功，-1 表示失败
     */
    template <typename Args>
    int64_t Run(const Args& arguments, aclrtStream stream = nullptr)
    {
        // ========== Step 1: 表达式推导 ==========
        // 使用 ExprMaker 生成表达式，推导 Params 类型列表
        // ToLinearizerExpr 将表达式转换为可分析的线性化形式
        auto expr = ToLinearizerExpr(ExprMaker{}.template Compute<Tensor>());
        
        // 提取表达式类型
        using Expr = typename decltype(expr)::Type;
        
        // 从表达式推导参数类型列表
        // Params_t<Expr> 返回 TypeList<Param<1>, Param<2>, ...>
        using Params = Atvoss::Params_t<Expr>;

        // ========== Step 2: 参数准备 ==========
        // 提取数据参数 tuple（第一个元素）
        // arguments 结构: tuple<tuple<in, out>, axes>
        auto argTuple = std::get<0>(arguments);
        
        // 根据 Params 类型列表构造参数
        // PrepareParams 会将 Tensor 转换为 DeviceTensor
        // 结果: tuple<DeviceTensor<T1>, DeviceTensor<T2>, ...>
        auto params = PrepareParams<Params, Tensor>(argTuple);

        // ========== Step 3: Tiling 计算 ==========
        OpParam opParam;
        // 计算 Kernel 和 Block 层的调度配置
        if (!CalculateTiling<KernelOp>(arguments, opParam)) {
            printf("[ERROR]: [Atvoss][Reduce][Device] CalculateTiling failed!\n");
            return -1;
        }

        // ========== Step 4: 参数合并 ==========
        // 合成构造参数和原始额外参数
        // params: 构造的 DeviceTensor 参数
        // argTuple: 原始用户输入（包含额外参数）
        // 结果: tuple<DeviceTensor, DeviceTensor, axes_ptr, scale, ...>
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        // ========== Step 5: Kernel 启动 ==========
#if ATVOSS_DEBUG_MODE == 2
        for (auto i = 0; i < 200; i++) {
            LaunchKernel<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
        }
#else
        LaunchKernel<KernelOp>(opParam.kernelParam.blockNum, stream, opParam, convertArgs);
#endif
        return 0;
    }
};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_DEVICE_ADAPTER_H