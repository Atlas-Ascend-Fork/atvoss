/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_TRANSCENDENTAL_H
#define Atvoss_DEV_TRANSCENDENTAL_H

#include "common.h"

namespace Atvoss::ExprTmpl {
    template<Atvoss::Pattern pattern, typename T>
    struct OpReduceSum : UnaryOp<T> {
        OpReduceSum() = default;

        constexpr OpReduceSum(T t) : UnaryOp<T>(t) {}
    };

    template<Atvoss::Pattern pattern, typename T>
    __host_aicore__ constexpr auto ReduceSum(Expression<T> lhs) {
        return Expression<OpReduceSum<pattern, T>>{{lhs.data}};
    }

    template<Atvoss::Pattern pattern, typename T>
    __host_aicore__ constexpr auto ReduceSum(T &&lhs) {
        return Expression<OpReduceSum<pattern, T>>{{std::forward<T>(lhs)}};
    }

    template<Atvoss::Pattern pattern, typename T>
    struct OpBroadcast : UnaryOp<T> {
        OpBroadcast() = default;

        constexpr OpBroadcast(T t) : UnaryOp<T>(t) {}
    };

    template<Atvoss::Pattern pattern, typename T>
    __host_aicore__ constexpr auto Broadcast(Expression<T> lhs) {
        return Expression<OpBroadcast<pattern, T>>{{lhs.data}};
    }

    template<Atvoss::Pattern pattern, typename T>
    __host_aicore__ constexpr auto Broadcast(T &&lhs) {
        return Expression<OpBroadcast<pattern, T>>{{std::forward<T>(lhs)}};
    }
}
#endif //Atvoss_DEV_TRANSCENDENTAL_H
