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
 * @brief Reduce 协调模块
 */

#ifndef ATVOSS_REDUCE_DEVICE_ADAPTER_H
#define ATVOSS_REDUCE_DEVICE_ADAPTER_H

#include "acl/acl.h"
#include "kernel_launcher.h"
#include "param_handler.h"
#include "utils/expr_deriver.h"
#include "tiling.h"

namespace Atvoss::Reduce {

/**
 * @brief Reduce 算子协调
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
    using Tensor = Atvoss::DeviceTensor<T>;

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
        using Params = typename Atvoss::ParamsDeriver<ExprMaker, Tensor>::Params;

        // ========== Step 2: 参数准备 ==========
        auto argTuple = std::get<0>(arguments);
        // 根据 Params 类型列表构造参数，Tensor 转换为 DeviceTensor
        // 结果: tuple<DeviceTensor<T1>, DeviceTensor<T2>, ...>
        auto params = PrepareParams<Params, Tensor>(argTuple);

        // ========== Step 3: 参数合并 ==========
        // 合成构造参数和原始额外参数
        // 结果: tuple<DeviceTensor, DeviceTensor, axes_ptr, scale, ...>
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        // ========== Step 4: Tiling 计算 ==========
        OpParam opParam;
        // 计算 Kernel 和 Block 层的配置参数
        if (!Atvoss::CalculateTiling<KernelOp>(arguments, opParam)) {
            printf("[ERROR]: [Atvoss][Reduce][Device] CalculateTiling failed!\n");
            return -1;
        }

        // ========== Step 5: Kernel 启动 ==========
        switch (opParam.kernelParam.tilingKey) {
            case 0:
                LaunchKernel<ReduceTilingKey<0, 0, 0>, KernelOp>(
                    opParam.kernelParam.coreNum, stream, opParam, convertArgs);
                break;
            case 1:
                LaunchKernel<ReduceTilingKey<1, 1, 1>, KernelOp>(
                    opParam.kernelParam.coreNum, stream, opParam, convertArgs);
                break;
            case 2:
                LaunchKernel<ReduceTilingKey<11, 10, 1>, KernelOp>(
                    opParam.kernelParam.coreNum, stream, opParam, convertArgs);
                break;
            default:
                break;
        }
        return 0;
    }
};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_DEVICE_ADAPTER_H