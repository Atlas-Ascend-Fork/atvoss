/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_UT_ABS_CONFIG_H
#define ATVOSS_UT_ABS_CONFIG_H

#include "utils/tensor.h"              // Compute表达式Tensor依赖
#include "utils/arguments/arguments.h" // 构造tiling计算函数的arguments需要
#include "elewise/block/builder.h"     // 计算图解析需要
#include "elewise/kernel/builder.h"    // 计算图解析需要

template <typename T, int32_t TILE_SIZE>
struct AbsConfig {
    using Dtype = T;

    using TileShape = Atvoss::Shape<TILE_SIZE>;
    struct AbsCompute {
        template <template <typename> class Tensor>
        constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<Dtype>, Atvoss::ParamUsage::IN>();
            auto out1 = Atvoss::PlaceHolder<2, Tensor<Dtype>, Atvoss::ParamUsage::OUT>();
            return (out1 = Abs(in1));
        };
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};

    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};

    using ArchTag = Atvoss::Arch::DAV_3510;

    using BlockOp = Atvoss::Ele::BlockBuilder<AbsCompute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig>;

    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;
};

#endif // ATVOSS_ABS_CONFIG_H