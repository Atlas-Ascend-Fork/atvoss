/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \brief 计算Reduce算子的Tiling数据
 *
 * 调用Kernel层的MakeScheduleConfig一次性计算所有Tiling数据，
 * Block层直接从 cfg.tilingData 读取需要的参数。
 *
 * \tparam KernelOp Kernel操作类型，如 KernelBuilder<BlockBuilder<...>>
 * \tparam Args 用户输入参数类型
 *
 * \param[in] args 用户输入参数，结构如下：
 *                 std::tuple<
 *                     std::tuple<Tensor<float>&, Tensor<float>&, ...>,  // 第0元素：输入输出Tensor tuple
 *                     std::tuple<AttrMap<const char*, std::vector<int>>, ...>  // 第1元素：属性tuple（如reduce axes）
 *                 >
 *                 - Tensor引用：包含shape、dtype、数据指针等信息
 *                 - AttrMap：属性键值对，如 {"axes", [1, 2]} 表示在第1、2轴做reduce
 *
 * \param[out] cfg Kernel操作参数，类型为 KernelOp::ScheduleCfgClz（即 KernelBuilder::OpParam），结构如下：
 *                 struct OpParam {
 *                     ReduceTilingData tilingData;  // 统一的Tiling数据
 *                 };
 *                 Block层可直接读取 cfg.tilingData 中的字段
 *
 */
#ifndef ATVOSS_REDUCE_TILING_H
#define ATVOSS_REDUCE_TILING_H

namespace Atvoss::Reduce {

template <typename KernelOp, typename Args>
bool CalculateTiling(const Args& args, typename KernelOp::ScheduleCfgClz& cfg)
{
    typename KernelOp::ScheduleClz kernelScheduler;
    if (!kernelScheduler.MakeScheduleConfig(args, cfg.tilingData)) {
        printf("[ERROR]: [Atvoss][Device] MakeScheduleConfig for kernel failed!\n");
        return false;
    }
    return true;
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_TILING_H