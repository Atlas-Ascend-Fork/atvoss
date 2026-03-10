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
#include "../builtin_kernel/abs_config.h"
#include "elewise/device/tiling.h" // elewise模板的tiling计算函数

class ElewiseTilingTest : public testing::Test {};

TEST_F(ElewiseTilingTest, UniqueTest)
{
    using KernelOp = AbsConfig<float, 32>::KernelOp;
    using TilingData = typename KernelOp::ScheduleCfgClz;
    TilingData tilingData;
    std::vector<int64_t> shape = {32, 32};
    Atvoss::Tensor<float> t1(nullptr, shape.data(), shape.size());
    Atvoss::Tensor<float> t2(nullptr, shape.data(), shape.size());
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2).build();

    auto result = Atvoss::CalculateTiling<KernelOp>(arguments, tilingData);
    ASSERT_TRUE(result);
    EXPECT_EQ(sizeof(tilingData), 56);
}