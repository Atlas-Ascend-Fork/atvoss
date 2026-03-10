
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "atvoss.h"
#include "../../../../include/graph/expr_linearizer.h"
static constexpr int32_t WIDTH = 32;

int main(int argc, char const* argv[])
{
    {
        using DtypeV1 = float;
        using DtypeV2 = float;
        using DtypeV3 = float;
        auto in1 = Atvoss::PlaceHolder<1, AscendC::GlobalTensor<DtypeV1>, Atvoss::ParamUsage::IN>();
        auto in2 = Atvoss::PlaceHolder<2, AscendC::GlobalTensor<DtypeV2>, Atvoss::ParamUsage::IN>();
        auto in3 = Atvoss::PlaceHolder<3, float, Atvoss::ParamUsage::IN>();
        auto out = Atvoss::PlaceHolder<4, AscendC::GlobalTensor<DtypeV3>, Atvoss::ParamUsage::OUT>();
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
        auto _10 = in2 * _9; // 10个expression
        auto _11 = in1 * _10;
        auto _12 = ReduceSum<Atvoss::Pattern::AR>(_11);
        auto _13 = Broadcast<Atvoss::Pattern::AB>(_12);
        auto _14 = Divs<WIDTH>(_13);
        auto _15 = _14 * in3;
        auto _16 = in3 * _15;
        auto _17 = in3 * _16;
        auto _18 = Atvoss::Sqrt(_17);
        auto _19 = in1 / _18;
        auto _20 = in2 * _19; // 20个expression
        auto _21 = in1 * _20;
        auto _22 = ReduceSum<Atvoss::Pattern::AR>(_21);
        auto _23 = Broadcast<Atvoss::Pattern::AB>(_22);
        auto _24 = Divs<WIDTH>(_23);
        auto _25 = _24 * in3;
        auto _26 = in3 * _25 + _15;
        auto _27 = in3 * _26;
        auto _28 = Atvoss::Sqrt(_27);
        auto _29 = in1 / _28;
        auto _30 = in2 * _29; // 30个expression
        auto _31 = in1 * _30;
        auto _32 = ReduceSum<Atvoss::Pattern::AR>(_31);
        auto _33 = Broadcast<Atvoss::Pattern::AB>(_32);
        auto _34 = Divs<WIDTH>(_33);
        auto _35 = _34 * in3;
        auto _36 = in3 * _35;
        auto _37 = in3 * _36;
        auto _38 = Atvoss::Sqrt(_37);
        auto _39 = in1 / _38;
        auto _40 = in2 * _39; // 40个expression
        auto _41 = in1 * _40;
        auto _42 = ReduceSum<Atvoss::Pattern::AR>(_41);
        auto _43 = Broadcast<Atvoss::Pattern::AB>(_42);
        auto _44 = Divs<WIDTH>(_43);
        auto _45 = _44 * in3;
        auto _46 = in3 * _45;
        auto _47 = in3 * _46;
        auto _48 = Atvoss::Sqrt(_47);
        auto _49 = in1 / _48;
        auto _50 = in2 * _49; // 50个expression
        auto _51 = in1 * _50;
        auto _52 = ReduceSum<Atvoss::Pattern::AR>(_51);
        auto _53 = Broadcast<Atvoss::Pattern::AB>(_52);
        auto _54 = Divs<WIDTH>(_53);
        auto _55 = _54 * in3;
        auto _56 = in3 * _55;
        auto _57 = in3 * _56;
        auto _58 = Atvoss::Sqrt(_57);
        auto _59 = in1 / _58;
        auto _60 = in2 * _59; // 60个expression
        auto _61 = in1 * _60;
        auto _62 = ReduceSum<Atvoss::Pattern::AR>(_61);
        auto _63 = Broadcast<Atvoss::Pattern::AB>(_62);
        auto _64 = Divs<WIDTH>(_63);
        auto _65 = _64 * in3;
        auto _66 = in3 * _65;
        auto _67 = in3 * _66;
        auto _68 = Atvoss::Sqrt(_67);
        auto _69 = in1 / _68;
        auto _70 = in2 * _69; // 70个expression
        auto _71 = in1 * _70;
        auto _72 = ReduceSum<Atvoss::Pattern::AR>(_71);
        auto _73 = Broadcast<Atvoss::Pattern::AB>(_72);
        auto _74 = Divs<WIDTH>(_73);
        auto _75 = _74 * in3;
        auto _76 = in3 * _75;
        auto _77 = in3 * _76;
        auto _78 = Atvoss::Sqrt(_77);
        auto _79 = in1 / _78;
        auto _80 = in2 * _79; // 80个expression
        auto _81 = in1 * _80;
        auto _82 = ReduceSum<Atvoss::Pattern::AR>(_81);
        auto _83 = Broadcast<Atvoss::Pattern::AB>(_82);
        auto _84 = Divs<WIDTH>(_83);
        auto _85 = _84 * in3;
        auto _86 = in3 * _85;
        auto _87 = in3 * _86;
        auto _88 = Atvoss::Sqrt(_87);
        auto _89 = in1 / _88;
        auto _90 = in2 * _89; // 90个expression
        auto _91 = in1 * _90;
        auto _92 = ReduceSum<Atvoss::Pattern::AR>(_91);
        auto _93 = Broadcast<Atvoss::Pattern::AB>(_92);
        auto _94 = Divs<WIDTH>(_93);
        auto _95 = _94 * in3;
        auto _96 = in3 * _95;
        auto _97 = in3 * _96;
        auto _98 = Atvoss::Sqrt(_97);
        auto _99 = in1 / _98;
        auto _100 = in2 * _99; // 100个expression
        auto _101 = in1 * _100;
        auto _102 = ReduceSum<Atvoss::Pattern::AR>(_101);
        auto _103 = Broadcast<Atvoss::Pattern::AB>(_102);
        auto _104 = Divs<WIDTH>(_103);
        auto _105 = _104 * in3;
        auto _106 = in3 * _105;
        auto _107 = in3 * _106;
        auto _108 = Atvoss::Sqrt(_107);
        auto _109 = in1 / _108;
        auto _110 = in2 * _109; // 110个expression
        auto _111 = in1 * _110;
        auto _112 = ReduceSum<Atvoss::Pattern::AR>(_111);
        auto _113 = Broadcast<Atvoss::Pattern::AB>(_112);
        auto _114 = Divs<WIDTH>(_113);
        auto _115 = _114 * in3;
        auto _116 = in3 * _115;
        auto _117 = in3 * _116;
        auto _118 = Atvoss::Sqrt(_117);
        auto _119 = in1 / _118;
        auto _120 = in2 * _119; // 120个expression
        auto xx = out = _120;
        static_assert(!std::is_same_v<decltype(Atvoss::ToLinearizerExpr(xx)), int>);
    }
    return 0;
}
