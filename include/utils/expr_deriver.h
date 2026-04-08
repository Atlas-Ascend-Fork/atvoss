/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file expr_deriver.h
 * @brief 表达式推导公共模块
 */

#ifndef ATVOSS_EXPR_DERIVER_H
#define ATVOSS_EXPR_DERIVER_H

#include "graph/expr_linearizer.h"

namespace Atvoss {

/**
 * @brief 参数类型推导器
 * @tparam ExprMaker 表达式生成器类型（如 Compute 类）
 * @tparam TensorT Tensor 模板类型（如 DeviceTensor）
 */
template <typename ExprMaker, template <typename> class TensorT>
struct ParamsDeriver {
    /**
     * @brief 表达式类型
     * 
     * 从 ExprMaker.Compute<TensorT>() 生成表达式
     */
    using Expr = typename decltype(ToLinearizerExpr(ExprMaker{}.template Compute<TensorT>()))::Type;

    /**
     * @brief 参数类型列表
     * 
     * 从表达式推导出的参数类型列表，格式为 TypeList<Param<1>, Param<2>, ...>
     * 每个 Param 包含 number（参数编号）、Type（数据类型）、usage（IN/OUT/IN_OUT）
     */
    using Params = Atvoss::Params_t<Expr>;
};

} // namespace Atvoss

#endif // ATVOSS_EXPR_DERIVER_H