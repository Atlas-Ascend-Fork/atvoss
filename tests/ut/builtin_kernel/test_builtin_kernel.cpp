/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iosfwd>
#include <algorithm>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "tiling_data_stub.h"
#include "elewise/device/tiling.h" // 这里是为了得到tiling的值需要elewise模板的tiling计算函数，真实的builtin算子不需要
#include "abs_config.h"

class ElewiseKenelTest : public testing::Test {};

template <uint32_t flag>
__global__ __aicore__ void AbsKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using KernelOp = AbsConfig<float, 32>::KernelOp;
    using CfgType = typename KernelOp::ScheduleCfgClz;
    REGISTER_TILING_DEFAULT(CfgType);
    GET_TILING_DATA_WITH_STRUCT(CfgType, tilingData, tiling);
    KernelOp op;
    op.Run(tilingData, x, y);
}

TEST_F(ElewiseKenelTest, kernel_test)
{
    using KernelOp = AbsConfig<float, 32>::KernelOp;
    using CfgType = typename KernelOp::ScheduleCfgClz;
    CfgType cfg;
    std::vector<int64_t> shape = {32, 32};
    Atvoss::Tensor<float> t1(nullptr, shape.data(), shape.size());
    Atvoss::Tensor<float> t2(nullptr, shape.data(), shape.size());
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2).build();

    auto result = Atvoss::CalculateTiling<KernelOp>(arguments, cfg);
    ASSERT_TRUE(result);
    EXPECT_EQ(sizeof(cfg), 56);
    constexpr size_t dataCount = 32 * 32;
    std::vector<float> x(dataCount, -2.1f);
    std::vector<float> y(dataCount, -2.1f);
    constexpr size_t dataSize = dataCount * sizeof(float);
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto x_data = std::unique_ptr<uint8_t, void (*)(void*)>((uint8_t*)AscendC::GmAlloc(dataSize), AscendC::GmFree);
    auto y_data = std::unique_ptr<uint8_t, void (*)(void*)>((uint8_t*)AscendC::GmAlloc(dataSize), AscendC::GmFree);
    std::copy(x.begin(), x.end(), reinterpret_cast<float*>(x_data.get()));

    ICPU_RUN_KF(
        AbsKernel<0>, cfg.kernelParam.blockNum, x_data.get(), y_data.get(), nullptr, reinterpret_cast<uint8_t*>(&cfg));

    std::copy(y_data.get(), y_data.get() + dataSize, reinterpret_cast<uint8_t*>(y.data()));

    std::vector<float> expectedValue(dataCount, 2.1f);
    EXPECT_EQ(y, expectedValue);
}