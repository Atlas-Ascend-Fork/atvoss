/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "operators/reduce_expression.h"
#include "operators/reduce_evaluator.h"
#include "reduce/policy.h"
#include "reduce/builder.h"

static void TestReduceExpression() {
    auto in = Atvoss::PlaceHolder<1, float, Atvoss::ParamUsage::IN>();
    auto out = Atvoss::PlaceHolder<2, float, Atvoss::ParamUsage::OUT>();
    
    auto expr1 = Atvoss::ReduceSum<0>(in);
    auto assign1 = (out = Atvoss::ReduceSum<0>(in));
}

struct TestReduceConfig {
    using TileShape = Atvoss::Shape<32, 16>;
    
    struct ReduceSumCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const {
            auto in = Atvoss::PlaceHolder<1, Tensor<float>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<float>, Atvoss::ParamUsage::OUT>();
            return (out = Atvoss::ReduceSum<0>(in));
        }
    };
    
    static constexpr auto blockPolicy = 
        Atvoss::Reduce::DefaultBlockPolicy<TileShape>{TileShape{}};
    static constexpr auto kernelPolicy = 
        Atvoss::Reduce::DefaultKernelPolicy<0>{};
    
    using DeviceOp = typename Atvoss::Reduce::ReduceBuilder<
        ReduceSumCompute, 0, Atvoss::Arch::DAV_3510,
        blockPolicy, kernelPolicy>::DeviceOp;
};

int main() {
    TestReduceExpression();
    return 0;
}