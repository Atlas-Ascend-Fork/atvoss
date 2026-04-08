/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_KERNEL_LAUNCHER_H
#define ATVOSS_REDUCE_KERNEL_LAUNCHER_H

#include <tuple>

#include "acl/acl.h"
#include "common/compile_info.h"
#include "param_handler.h"

namespace Atvoss::Reduce {

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

template <class KernelOp, typename OpParam, typename ArgTup>
void LaunchKernel(uint32_t blockNum, aclrtStream stream, OpParam& cfg, const ArgTup& argTuple)
{
    auto transformedArgs = TransformArgsToPtr(argTuple);
    KernelCustom<KernelOp, OpParam><<<blockNum, nullptr, stream>>>(cfg, transformedArgs);
}

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_KERNEL_LAUNCHER_H