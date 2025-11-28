/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __Atvoss_TILE_OPERATOR_H__
#define __Atvoss_TILE_OPERATOR_H__
#include "tile_evaluator_common.h"

namespace Atvoss::Tile {

/*!
 * \brief Datacopy from src to dst
 * \param[in] src, Input GlobalTensor
 * \param[in] copyCnt, Length of copy data
 * \param[out] dst, Output LocalTensor
 */
template <typename T>
__aicore__ inline void CopyIn(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t copyCnt)
{
    AscendC::DataCopy(dst, src, copyCnt);
}

/*!
 * \brief Datacopy from src to dst
 * \param[in] src, Input LocalTensor
 * \param[in] copyCnt, Length of copy data
 * \param[out] dst, Output GlobalTensor
 */
template <typename T>
__aicore__ inline void CopyIn(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t copyCnt)
{
    AscendC::DataCopy(dst, src, copyCnt);
}
} // namespace Atvoss::Tile
#endif