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
#include "kernel_operator.h"
#include "../../include/atvoss.h"
#include "../../include/graph/expr_linearizer.h"

class ExprLinearizerTest : public testing::Test {};

TEST_F(ExprLinearizerTest, ToLinearizerExpr)
{
    constexpr uint32_t WIDTH = 32;
    auto in1 = Atvoss::PlaceHolder<1, AscendC::GlobalTensor<float>, Atvoss::ParamUsage::IN>();
    auto in2 = Atvoss::PlaceHolder<2, AscendC::GlobalTensor<float>, Atvoss::ParamUsage::IN>();
    auto in3 = Atvoss::PlaceHolder<3, float, Atvoss::ParamUsage::IN>();
    auto out = Atvoss::PlaceHolder<4, AscendC::GlobalTensor<float>, Atvoss::ParamUsage::OUT>();
    auto out2 = Atvoss::PlaceHolder<5, AscendC::GlobalTensor<float>, Atvoss::ParamUsage::OUT>();
    auto out3 = Atvoss::PlaceHolder<6, AscendC::GlobalTensor<float>, Atvoss::ParamUsage::OUT>();
    auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);
    auto temp1 = Atvoss::PlaceHolderTmpLike<2>(in1);
    auto temp2 = Atvoss::PlaceHolderTmpLike<3>(in1);
    auto temp3 = Atvoss::PlaceHolderTmpLike<4>(in1);
    auto temp4 = Atvoss::PlaceHolderTmpLike<5>(in1);
    auto temp5 = Atvoss::PlaceHolderTmpLike<6>(in1);
    auto temp6 = Atvoss::PlaceHolderTmpLike<7>(in1);
    auto temp7 = Atvoss::PlaceHolderTmpLike<8>(in1);
    auto temp8 = Atvoss::PlaceHolderTmpLike<9>(in1);

    auto _1 = in1 * in1;
    auto _2 = Atvoss::ReduceSum<Atvoss::Pattern::AR>(_1);
    auto _3 = Atvoss::Broadcast<Atvoss::Pattern::AB>(_2);
    auto _4 = Atvoss::Divs<WIDTH>(_3);
    auto _4x = _4 * in3;
    auto _4y = _4x + _1;
    auto _5 = in1 / (Atvoss::Sqrt(_4y) + _1);
    auto xx = (out = in2 * _5, out2 = in2 + _5, out3 = in2 / _4y);

    auto xx1 =
        (temp = in1 * in1, temp1 = Atvoss::ReduceSum<Atvoss::Pattern::AR>(temp),
         temp2 = Atvoss::Broadcast<Atvoss::Pattern::AB>(temp1), temp3 = Atvoss::Divs<WIDTH>(temp2), temp4 = temp3 * in3,
         temp5 = temp4 + temp, temp6 = Atvoss::Sqrt(temp5), temp7 = temp6 + temp, temp8 = in1 / temp7,
         out = in2 * temp8, out2 = in2 + temp8, out3 = in2 / temp5);

    constexpr bool sameType = std::is_same_v<decltype(xx1), decltype(Atvoss::ToLinearizerExpr(xx))>;
    EXPECT_TRUE(sameType);
}