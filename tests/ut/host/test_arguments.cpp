/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "kernel_operator.h"
#include "utils/log.h"
#include "atvoss.h"
#include "../../../include/operators/math_expression.h"
#include "utils/layout/shape.h"

static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;
static constexpr int32_t MAX_DIM = 8;

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
            auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::IN>();
            auto in3 = Atvoss::PlaceHolder<3, float, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<4, Tensor<DtypeV3>, Atvoss::ParamUsage::OUT>();
            auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

            auto _1 = in1 * in1;
            auto _2 = ReduceSum<Atvoss::Pattern::AR>(_1);
            auto _3 = Broadcast<Atvoss::Pattern::AB>(_2);
            auto _4 = Divs<WIDTH>(_3);
            auto _5 = _4 * in3;
            auto _6 = in3 * _5;
            auto _7 = in3 * _6;
            auto _8 = Atvoss::Sqrt(_7);
            auto _9 = in1 / _8;
            auto _10 = in2 * _9;
            auto _11 = in1 * _10;
            return out = _11;
        }
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<RmsNormCompute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig>;

    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

using KernelOp = typename RmsNormConfig<float, float, float>::KernelOp;
using OpParam = KernelOp::ScheduleCfgClz;

class AtvossArgumentsTest : public testing::Test {
protected:
    void SetUp()
    {
        std::cout << "AtvossArgumentsTest SetUp" << std::endl;
    }

    void TearDown()
    {
        std::cout << "AtvossArgumentsTest TearDown" << std::endl;
    }
};

TEST_F(AtvossArgumentsTest, AtvossArgumentsTestCase)
{
    uint64_t shapeArray[MAX_DIM] = {0};
    std::vector<uint64_t> shape{32, 32};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<float> t1(nullptr, shapeArray, shape.size());
    Atvoss::Tensor<float> t2(nullptr, shapeArray, shape.size());
    Atvoss::Tensor<float> t3(nullptr, shapeArray, shape.size());
    float a = 1.0f;
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, a, t3).build();
    auto arg = std::get<2>(std::get<0>(arguments));
    EXPECT_EQ(arg, 1.0f);
}
