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

#include "acl/acl.h"
#include "kernel_launcher.h"
#include "param_transform.h"
#include "tiling.h"
#include "utils/utility.h"

namespace Atvoss::Reduce {

template <typename KernelOp>
class DeviceAdapter {
public:
    using ExprMaker = typename KernelOp::ScheduleClz::ExprMaker;
    using BlockOp = typename KernelOp::ScheduleClz::BlockTemplate;
    using OpParam = typename KernelOp::ScheduleCfgClz;

    template <typename T>
    using Tensor = DeviceTensor<T>;

    DeviceAdapter() = default;

    template <typename Args>
    int64_t Run(const Args& arguments, aclrtStream stream = nullptr)
    {
        auto expr = ToLinearizerExpr(ExprMaker{}.template Compute<Tensor>());
        using Expr = typename decltype(expr)::Type;
        using Params = Atvoss::Params_t<Expr>;

        auto argTuple = std::get<0>(arguments);

        auto params = PrepareParams<Params, Tensor>(argTuple);

        OpParam opParam;
        if (!CalculateTiling<KernelOp>(arguments, opParam)) {
            printf("[ERROR]: [Atvoss][Reduce][Device] CalculateTiling failed!\n");
            return -1;
        }

        auto convertArgs = ConvertArgs<Params>(params, argTuple);

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