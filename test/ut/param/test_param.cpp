/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "include/utils/expression/common.h"
#include "kernel/kernel_elewise.h"
#include "tile/tile_elewise.h"
#include "tile/tile_evaluator_common.h"
#include "tile/tile_evaluator.h"
#include "block/block_elewise.h"

namespace Atvoss {
static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;
using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;
struct RmsNormCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<float>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<2, Tensor<float>, Atvoss::ParamUsage::out>();
        return (out = in1 * in1);
    }
};
class AtvossParamTest : public testing::Test {
protected:
     void SetUp()
    {
        AscendC::SetGCoreType(2U);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvossParamTest, AtvossParamTestSuccessCase) {
    std::cout << "[TEST] Running AtvossParamTestSuccessCase - Just printing info." << std::endl;
    static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {190 * 1024, TileShape{}};
    static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {48, Atvoss::EleWise::KernelPolicySegment::UniformSegment};
    using BlockOp = Atvoss::EleWise::BlockBuilder<RmsNormCompute, blockPolicy, Atvoss::EleWise::BlockConfig>;
    using KernelOp = Atvoss::EleWise::KernelBuilder<BlockOp, kernelPolicy>;
    using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;
    using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct;
    KernelParamStruct kernelParam;
    BlockParamStruct blockParam;
    std::vector<uint32_t> shape1{32*32};
    auto ret = KernelOp::ScheduleClz::MakeKernelParam(shape1, kernelParam);
    EXPECT_EQ(true, ret);
    std::vector<uint32_t> shape4{1, 1024};
    ret = KernelOp::ScheduleClz::MakeKernelParam(shape4, kernelParam);
    EXPECT_EQ(true, ret);
    ret = BlockOp::ScheduleClz::MakeBlockParam(blockParam);
    EXPECT_EQ(true, ret);
}


TEST_F(AtvossParamTest, AtvossParamTestFailedCase) {
    std::cout << "[TEST] Running AtvossParamTestFailedCase - Just printing info." << std::endl;
    static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {0, TileShape{}};
    static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {48, Atvoss::EleWise::KernelPolicySegment::UniformSegment};
    using BlockOp = Atvoss::EleWise::BlockBuilder<RmsNormCompute, blockPolicy, Atvoss::EleWise::BlockConfig>;
    using KernelOp = Atvoss::EleWise::KernelBuilder<BlockOp, kernelPolicy>;
    using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;
    using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct;
    KernelParamStruct kernelParam;
    BlockParamStruct blockParam;
    std::vector<uint32_t> shape1{};
    auto ret = KernelOp::ScheduleClz::MakeKernelParam(shape1, kernelParam);
    EXPECT_EQ(false, ret);
    std::vector<uint32_t> shape2{0};
    ret = KernelOp::ScheduleClz::MakeKernelParam(shape2, kernelParam);
    EXPECT_EQ(false, ret);
    ret = BlockOp::ScheduleClz::MakeBlockParam(blockParam);
    EXPECT_EQ(false, ret);
}

}  // namespace Atvoss
