
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "utils/log.h"
#include "atvoss.h"
#include "../../../../include/expression/expr_template.h"
#include "../../../../include/utils/arguments/arguments.h"
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
            auto temp1 = Atvoss::PlaceHolderTmpLike<1>(in1);

            auto _1 = Atvoss::BindBuff(temp1, in1 * in1);
            auto _2 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_1));
            auto _3 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_2));
            auto _4 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_3));
            auto _5 = Atvoss::BindBuff(temp1, _4 * in3);
            auto _6 = Atvoss::BindBuff(temp1, in3 * _5);
            auto _7 = Atvoss::BindBuff(temp1, in3 * _6);
            auto _8 = Atvoss::BindBuff(out, Sqrt(_7));
            auto _9 = Atvoss::BindBuff(temp1, in1 / _8);
            auto _10 = Atvoss::BindBuff(out, in2 * _9); // 10个expression
            auto _11 = Atvoss::BindBuff(temp1, in1 * _10);
            auto _12 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_11));
            auto _13 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_12));
            auto _14 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_13));
            auto _15 = Atvoss::BindBuff(temp1, _14 * in3);
            auto _16 = Atvoss::BindBuff(temp1, in3 * _15);
            auto _17 = Atvoss::BindBuff(temp1, in3 * _16);
            auto _18 = Atvoss::BindBuff(out, Sqrt(_17));
            auto _19 = Atvoss::BindBuff(temp1, in1 / _18);
            auto _20 = Atvoss::BindBuff(out, in2 * _19); // 20个expression
            auto _21 = Atvoss::BindBuff(temp1, in1 * _20);
            auto _22 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_21));
            auto _23 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_22));
            auto _24 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_23));
            auto _25 = Atvoss::BindBuff(temp1, _24 * in3);
            auto _26 = Atvoss::BindBuff(temp1, in3 * _25);
            auto _27 = Atvoss::BindBuff(temp1, in3 * _26);
            auto _28 = Atvoss::BindBuff(out, Sqrt(_27));
            auto _29 = Atvoss::BindBuff(temp1, in1 / _28);
            auto _30 = Atvoss::BindBuff(out, in2 * _29); // 30个expression
            auto _31 = Atvoss::BindBuff(temp1, in1 * _30);
            auto _32 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_31));
            auto _33 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_32));
            auto _34 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_33));
            auto _35 = Atvoss::BindBuff(temp1, _34 * in3);
            auto _36 = Atvoss::BindBuff(temp1, in3 * _35);
            auto _37 = Atvoss::BindBuff(temp1, in3 * _36);
            auto _38 = Atvoss::BindBuff(out, Sqrt(_37));
            auto _39 = Atvoss::BindBuff(temp1, in1 / _38);
            auto _40 = Atvoss::BindBuff(out, in2 * _39); // 40个expression
            auto _41 = Atvoss::BindBuff(temp1, in1 * _40);
            auto _42 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_41));
            auto _43 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_42));
            auto _44 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_43));
            auto _45 = Atvoss::BindBuff(temp1, _44 * in3);
            auto _46 = Atvoss::BindBuff(temp1, in3 * _45);
            auto _47 = Atvoss::BindBuff(temp1, in3 * _46);
            auto _48 = Atvoss::BindBuff(out, Sqrt(_47));
            auto _49 = Atvoss::BindBuff(temp1, in1 / _48);
            auto _50 = Atvoss::BindBuff(out, in2 * _49); // 50个expression
            auto _51 = Atvoss::BindBuff(temp1, in1 * _50);
            auto _52 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_51));
            auto _53 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_52));
            auto _54 = Atvoss::BindBuff(temp1, Divs<WIDTH>(_53));
            auto _55 = Atvoss::BindBuff(temp1, _54 * in3);
            auto _56 = Atvoss::BindBuff(temp1, in3 * _55);
            auto _57 = Atvoss::BindBuff(temp1, in3 * _56);
            auto _58 = Atvoss::BindBuff(out, Sqrt(_57));
            auto _59 = Atvoss::BindBuff(temp1, in1 / _58);
            // auto _60 = Atvoss::BindBuff(out, in2 * _59); // 60个expression
            // auto _61 = Atvoss::BindBuff(temp1, in1 * _60);
            // auto _62 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_61));
            // auto _63 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_62));
            // auto _64 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_63));
            // auto _65 = Atvoss::BindBuff(temp1, _64 * in3);
            // auto _66 = Atvoss::BindBuff(temp1, in3 * _65);
            // auto _67 = Atvoss::BindBuff(temp1, in3 * _66);
            // auto _68 = Atvoss::BindBuff(out, Sqrt(_67));
            // auto _69 = Atvoss::BindBuff(temp1, in1 / _68);
            // auto _70 = Atvoss::BindBuff(out, in2 * _69); // 70个expression
            // auto _71 = Atvoss::BindBuff(temp1, in1 * _70);
            // auto _72 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_71));
            // auto _73 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_72));
            // auto _74 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_73));
            // auto _75 = Atvoss::BindBuff(temp1, _74 * in3);
            // auto _76 = Atvoss::BindBuff(temp1, in3 * _75);
            // auto _77 = Atvoss::BindBuff(temp1, in3 * _76);
            // auto _78 = Atvoss::BindBuff(out, Sqrt(_77));
            // auto _79 = Atvoss::BindBuff(temp1, in1 / _78);
            // auto _80 = Atvoss::BindBuff(out, in2 * _79); // 80个expression
            // auto _81 = Atvoss::BindBuff(temp1, in1 * _80);
            // auto _82 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_81));
            // auto _83 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_82));
            // auto _84 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_83));
            // auto _85 = Atvoss::BindBuff(temp1, _84 * in3);
            // auto _86 = Atvoss::BindBuff(temp1, in3 * _85);
            // auto _87 = Atvoss::BindBuff(temp1, in3 * _86);
            // auto _88 = Atvoss::BindBuff(out, Sqrt(_87));
            // auto _89 = Atvoss::BindBuff(temp1, in1 / _88);
            // auto _90 = Atvoss::BindBuff(out, in2 * _89); // 90个expression
            // auto _91 = Atvoss::BindBuff(temp1, in1 * _90);
            // auto _92 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_91));
            // auto _93 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_92));
            // auto _94 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_93));
            // auto _95 = Atvoss::BindBuff(temp1, _94 * in3);
            // auto _96 = Atvoss::BindBuff(temp1, in3 * _95);
            // auto _97 = Atvoss::BindBuff(temp1, in3 * _96);
            // auto _98 = Atvoss::BindBuff(out, Sqrt(_97));
            // auto _99 = Atvoss::BindBuff(temp1, in1 / _98);
            // auto _100 = Atvoss::BindBuff(out, in2 * _99); // 100个expression
            // auto _101 = Atvoss::BindBuff(temp1, in1 * _100);
            // auto _102 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_101));
            // auto _103 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_102));
            // auto _104 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_103));
            // auto _105 = Atvoss::BindBuff(temp1, _104 * in3);
            // auto _106 = Atvoss::BindBuff(temp1, in3 * _105);
            // auto _107 = Atvoss::BindBuff(temp1, in3 * _106);
            // auto _108 = Atvoss::BindBuff(out, Sqrt(_107));
            // auto _109 = Atvoss::BindBuff(temp1, in1 / _108);
            // auto _110 = Atvoss::BindBuff(out, in2 * _109); // 110个expression
            // auto _111 = Atvoss::BindBuff(temp1, in1 * _110);
            // auto _112 = Atvoss::BindBuff(out, ReduceSum<Atvoss::Pattern::AR>(_111));
            // auto _113 = Atvoss::BindBuff(out, Broadcast<Atvoss::Pattern::AB>(_112));
            // auto _114 = Atvoss::BindBuff(temp1, Divs < WIDTH > (_113));
            // auto _115 = Atvoss::BindBuff(temp1, _114 * in3);
            // auto _116 = Atvoss::BindBuff(temp1, in3 * _115);
            // auto _117 = Atvoss::BindBuff(temp1, in3 * _116);
            // auto _118 = Atvoss::BindBuff(out, Sqrt(_117));
            // auto _119 = Atvoss::BindBuff(temp1, in1 / _118);
            return out = in2 * _59; // 120个expression
        }
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<RmsNormCompute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig>;

    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

int main(int argc, char const* argv[])
{
    uint64_t shapeArray[MAX_DIM] = {0};
    std::vector<uint64_t> shape{32, 32};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<float> t1(nullptr, shapeArray, shape.size());
    Atvoss::Tensor<float> t2(nullptr, shapeArray, shape.size());
    Atvoss::Tensor<float> t3(nullptr, shapeArray, shape.size());
    float a = 1.0f;
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, a, t3).build();

    using DeviceOp = typename RmsNormConfig<float, float, float>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, nullptr);
    return 0;
}
