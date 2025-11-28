/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TILE_ELE_WISE_H
#define TILE_ELE_WISE_H

#include "tile_evaluator.h"

namespace Atvoss::Tile {

/*!
* \brief Perform calculation based on the expression.
* \param[in] args, LocalTensor data of input tensors & output tensor.
* \param[in] localVar, LocalTensor data of temp tensors.
* \param[in] arguments, Other parameters
*/
template<typename Expr, typename ArgTup, typename LocalTuple, typename... Arguments>
__aicore__  inline void Evaluate(ArgTup &args, LocalTuple &localVar, Arguments&... arguments) {
   Eval::Evaluator<Expr>{}(Expr{}, args, localVar, arguments...);
}

/*!
* \brief Calculation layout's total length.
* \param[in] eleCntInTensor, Init-val.
* \param[in] defaultSize, Init-val.
* \return int32_t, Shape's total
*/
template<size_t N, typename T>
static constexpr __aicore__  inline int32_t GetTotal(uint32_t eleCntInTensor = 1, int defaultSize = 1) {
   using shape = typename T::TileShape;
   return Tile::Eval::GetTotal < N, shape > (eleCntInTensor, defaultSize);
}

} // namespace Atvoss::Tile
#endif