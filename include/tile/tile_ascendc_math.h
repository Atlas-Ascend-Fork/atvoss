/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_DEV_TILE_ASCENDC_MATH_H
#define Atvoss_DEV_TILE_ASCENDC_MATH_H
#include "lib/math/power.h"

using namespace AscendC::Std;
namespace Atvoss::Tile::Eval {

/*!
 * \brief dst[i] = src0[i] + src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void AddAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
{
    AscendC::Add(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src0[i] - src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SubAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
{
    AscendC::Sub(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src0[i] * src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void MulAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
{
    AscendC::Mul(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src0[i] / src1[i]
 * \param[in] src0, Input LocalTensor
 * \param[in] src1, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void DivAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src0,
                                 const AscendC::LocalTensor<T>& src1, OperationShape& operationShape)
{
    AscendC::Div(dst, src0, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = src[i] / scalarValue
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void DivsAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                  OperationShape& operationShape)
{
    T src1 = T{1} / scalarValue;
    AscendC::Muls(dst, src, src1, operationShape.axis0);
}

/*!
 * \brief dst[i] = exp(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void ExpAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                 OperationShape& operationShape)
{
    AscendC::Exp(dst, src, operationShape.axis0);
}

/*!
 * \brief dst[i] = sqrt(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, typename T>
__aicore__ inline void SqrtAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                  OperationShape& operationShape)
{
    AscendC::Sqrt(dst, src, operationShape.axis0);
}

/*!
 * \brief dst[i] = power(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, auto scalarValue, typename T>
__aicore__ inline void PowerAssign(AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                   OperationShape& operationShape)
{
    AscendC::Power(dst, src, T{scalarValue}, operationShape.axis0);
}

/*!
 * \brief dst[i] = cast(src[i])
 * \param[in] src, Input LocalTensor
 * \param[out] dst, Output LocalTensor
 */
template <typename OperationShape, CastMode castMode, typename T1, typename T2>
__aicore__ inline void CastAssign(AscendC::LocalTensor<T1>& dst, const AscendC::LocalTensor<T2>& src,
                                  OperationShape& operationShape)
{
    if constexpr (castMode == CastMode::CAST_NONE) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_RINT) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_FLOOR) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_FLOOR, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_CEIL) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_CEIL, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_ROUND) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ROUND, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_TRUNC) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_TRUNC, operationShape.axis0);
    } else if constexpr (castMode == CastMode::CAST_ODD) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ODD, operationShape.axis0);
    }
}
}  // namespace Atvoss::Tile::Eval
#endif  //Atvoss_DEV_TILE_ASCENDC_MATH_H
