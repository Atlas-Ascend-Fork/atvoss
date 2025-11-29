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
struct RmsNormCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<float>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<2, Tensor<float>, Atvoss::ParamUsage::out>();
        return (out = in1 * in1);
    }
};

using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;
static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {190 * 1024, TileShape{}};
static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {48, Atvoss::EleWise::KernelPolicySegment::UniformSegment};
using BlockOp = Atvoss::EleWise::BlockBuilder<RmsNormCompute, blockPolicy, Atvoss::EleWise::BlockConfig>;
using KernelOp = Atvoss::EleWise::KernelBuilder<BlockOp, kernelPolicy>;
using KernelParamStruct = typename KernelOp::ScheduleClz::ParamStruct;
using BlockParamStruct = typename KernelOp::ScheduleClz::BlockTemplate::ScheduleClz::ParamStruct;

struct OpParam {
    KernelParamStruct kernelParam;
    BlockParamStruct blockParam;
};
template<class KernelOp, typename OpParam>
__global__ __aicore__ void KernelCustom(OpParam cfg, GM_ADDR x, GM_ADDR y)
{
    AscendC::TPipe pipeIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelOp op;
    op.Run(cfg, x, y);
}

bool CalcParam(std::vector<uint32_t> shape, OpParam &opParam)
{   
    KernelOp::ScheduleClz::MakeKernelParam(shape, opParam.kernelParam);
    BlockOp::ScheduleClz::MakeBlockParam(opParam.blockParam);
}


void TestKernelCustom(uint32_t *a, uint32_t *b, std::vector<uint32_t> shape)
{
    OpParam opParam;
    CalcParam(shape, opParam);
    KernelCustom<KernelOp, OpParam>(opParam,
        reinterpret_cast<uint8_t *>(a),
        reinterpret_cast<uint8_t *>(b));
}

class AtvossKernelTest : public testing::Test {  
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

TEST_F(AtvossKernelTest, AtvossKernelTestCase) {
    std::cout << "[TEST] Running AtvossKernelTest - Just printing info." << std::endl;
    std::vector<uint32_t> shape{32,32};
    uint32_t eleNum = 1U;
    for(int i = 0; i < shape.size(); i++){
        eleNum = shape[i] * eleNum;
    }
    uint32_t xGm[eleNum] = {1};
    uint32_t yGm[eleNum] = {0};
    TestKernelCustom(xGm, yGm, shape);
    EXPECT_EQ(yGm[0], 0);
}


}// namespace ATVOSS