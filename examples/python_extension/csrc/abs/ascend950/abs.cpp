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
 * \file add.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

#include "atvoss.h"

// Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("abs(Tensor x) -> Tensor");
}

// Meta function implementation of Abs
torch::Tensor abs_meta(const torch::Tensor& x)
{
    auto y = torch::empty_like(x);
    return y;
}

// Register the Meta implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("abs", abs_meta);
}

static constexpr int32_t TILE_SIZE = 32;

template <typename T>
struct AbsConfig {
    using Dtype = T;

    using TileShape = Atvoss::Shape<TILE_SIZE>;
    struct AbsCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
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

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

torch::Tensor abs_npu(const torch::Tensor& x)
{
    auto y = abs_meta(x);
    uint32_t shape[2] = {};
    std::copy(x.sizes().begin(), x.sizes().end(), shape);

    Atvoss::Tensor<float> t1(x.data_ptr<float>(), shape);
    Atvoss::Tensor<float> t2(y.data_ptr<float>(), shape);

    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2).build();
    using Config = AbsConfig<float>;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    Config::DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);
    return y;
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("abs", abs_npu);
}
