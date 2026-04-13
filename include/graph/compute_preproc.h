/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPUTE_PREPROCESS_H
#define COMPUTE_PREPROCESS_H

#include "utils/layout/layout.h"
#include "utils/patterns.h"
#include "utils/utility.h"

namespace Atvoss::Graph {
enum class AutoOpOnDemand : uint8_t
{
    no,
    yes
};

using Atvoss::Util::Get_t;
using Atvoss::Util::Set_t;
using Atvoss::Util::Size_v;

struct AllocInserter {
    template <typename T, typename ExprList>
    __host_aicore__ constexpr auto operator()(T, ExprList) const
    {
        using ParamUse = typename T::Type;
        using Param = Get_t<ParamUse, 0>;
        if constexpr (!std::is_scalar_v<typename Param::Type>) {
            constexpr auto firstUse = Get_t<ParamUse, 1>::value;
            using OldItem = Get_t<ExprList, firstUse>;
            using NewItem = OpAndThen<OpAlloc<Param>, OldItem>;
            return Set_t<ExprList, firstUse, NewItem>{};
        } else {
            return ExprList{};
        }
    }
};

struct FreeInserter {
    template <typename T, typename ExprList>
    __host_aicore__ constexpr auto operator()(T, ExprList) const
    {
        using ParamUse = typename T::Type;
        using Param = Get_t<ParamUse, 0>;
        if constexpr (!std::is_scalar_v<typename Param::Type>) {
            constexpr auto lastUse = Get_t<ParamUse, 2>::value;
            using OldItem = Get_t<ExprList, lastUse>;
            using NewItem = OpAndThen<OldItem, OpFree<Param>>;
            return Set_t<ExprList, lastUse, NewItem>{};
        } else {
            return ExprList{};
        }
    }
};
} // namespace Atvoss::Graph
#endif // COMPUTE_PREPROCESS_H