/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_ELEWISE_H
#define TILE_ELEWISE_H
#include "common/type_def.h"
#include "operators/tensor_expression.h"
#include "operators/math_expression.h"
#include "operators/transcendental_expression.h"

#include "operators/tile_shape.h"
#include "graph/compute_preproc.h"
#include "evaluator/eval_base.h"
#include "operators/math_evaluator.h"
#include "operators/tensor_evaluator.h"
#include "operators/transcendental_evaluator.h"

namespace Atvoss::Ele::Tile {

/*!
 * \brief Perform calculation based on the expression.
 * \param[in] args, LocalTensor data of input tensors & output tensor.
 * \param[in] localVar, LocalTensor data of temp tensors.
 * \param[in] arguments, Other parameters
 */
template <typename Expr, typename Context>
__aicore__ inline void Evaluate(Context& context)
{
    Evaluator<Expr>{}(Expr{}, context);
}

} // namespace Atvoss::Ele::Tile
#endif