/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "utils/layout/shape.h"

static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;

template <typename T1, typename T2, typename T3>
struct RmsNormConfig {
    using DtypeV1 = T1;
    using DtypeV2 = T2;
    using DtypeV3 = T3;
    using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;

    struct RmsNormCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::in>();
            auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV3>, Atvoss::ParamUsage::out>();
            auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

            return (temp = in1 * in1,
                    out = ReduceSum<Atvoss::Patterns::Pattern::AR>(temp),
                    out = Broadcast<Atvoss::Patterns::Pattern::AB>(out),
                    temp = Divs<WIDTH>(out),
                    out = Sqrt(temp),
                    temp = in1 / out,
                    out = in2 * temp);
        }
    };

    static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {
        190 * 1024,
        TileShape{}
    };
    static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {
        48,
        Atvoss::EleWise::KernelPolicySegment::UniformSegment
    };

    using BlockOp = Atvoss::EleWise::BlockBuilder<
        RmsNormCompute,
        blockPolicy,
        Atvoss::EleWise::BlockConfig>;

    using KernelOp = Atvoss::EleWise::KernelBuilder<
        BlockOp,
        kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

int main()
{
    using Config = RmsNormConfig<float, float, float>;
    std::vector<float> v1(32 * 32, 1.0F);
    std::vector<float> v2(32 * 32, 2.0F);
    std::vector<float> v3(32 * 32);
    std::vector<float> golden(32 * 32, 2.0f);
    uint32_t shape[2] = {32, 32};

    Atvoss::Tensor<float> t1(v1.data(), shape);
    Atvoss::Tensor<float> t2(v2.data(), shape);
    Atvoss::Tensor<float> t3(v3.data(), shape);

    // 生成入参信息
    auto arguments = Atvoss::ArgumentsBuilder{}.input(t1, t2).output(t3).build();

    Config::DeviceOp deviceOp;
    deviceOp.Run(arguments);

    if (!VerifyResults(golden, v3)) {
        return -1;
    }

    printf("Accuracy verification passed.\n");
    return 0;
}