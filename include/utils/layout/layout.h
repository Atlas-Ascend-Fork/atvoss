/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef Atvoss_LAYOUT_H
#define Atvoss_LAYOUT_H

namespace Atvoss::Layout {

struct OperationShape {
    uint32_t axis0 = 1;
    uint32_t axis1 = 1;
    uint32_t axis2 = 1;
};

template <size_t TotalCnt, size_t Axis0, size_t Axis1>
struct FixedRankExtents {
    static constexpr auto UNARY_SHAPE = OperationShape{TotalCnt};
    static constexpr auto BINARY_SHAPE = OperationShape{Axis0, Axis1};
};

template <typename Shape = FixedRankExtents<1, 1, 1>, typename Stride = Shape>
class Layout {
public:
    using ShapeType = Shape;
    using StrideType = Stride;
};

template <size_t N>
struct VariableRankExtents {
};

template <typename Shape = VariableRankExtents<1>, typename Stride = Shape>
class TailLayout {
public:
    using ShapeType = Shape;
    using StrideType = Stride;
    __aicore__ inline TailLayout() = default;

    __aicore__ inline TailLayout(uint32_t tailCnt, uint32_t axis0, uint32_t axis1){
        unaryShape_ = OperationShape{tailCnt};
        binaryShape_ = OperationShape{axis0, axis1};
    }

    __aicore__ inline OperationShape GetUnaryShape()
    {
        return unaryShape_;
    };

    __aicore__ inline OperationShape GetBinaryShape()
    {
        return binaryShape_;
    }

private:
    OperationShape unaryShape_;
    OperationShape binaryShape_;
};
}  // namespace Atvoss::Layout
#endif  //Atvoss_LAYOUT_H
