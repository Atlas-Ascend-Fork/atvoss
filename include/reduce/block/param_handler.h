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
 * @file param_handler.h
 * @brief Reduce Block 参数处理模块: 将 GM 地址构造为 BlockTensor
 */

#ifndef ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H
#define ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H

#include "block_info_tile.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

/**
 * @brief Tag 类型: 用于在模板参数推断中传递 BlockTensor 构造器
 * @tparam BlockTensorT BlockTensor 模板 (如 BlockTensorTile)
 */
template <template <typename> class BlockTensorT>
struct BlockTensorTag {};

/**
 * @brief 构造单个 BlockTensor: 从 argTuple 中取出 GM 地址并绑定
 * @tparam ParamType 参数类型 (Param<N, T, Usage>)
 * @tparam ArgTup 参数 tuple 类型
 * @tparam BlockTensorT BlockTensor 模板
 * @param args argTuple (GM 偏移后的地址)
 * @param tag BlockTensorTag 标签 (用于模板推断)
 * @return BlockTensor<T>{gm} 或原始标量值
 */
template <typename ParamType, typename ArgTup, template <typename> class BlockTensorT>
__aicore__ inline constexpr auto ConstructBlockParam(ArgTup& args, BlockTensorTag<BlockTensorT>)
{
    constexpr auto argNumber = ParamType::inplaceNumber - 1;
    if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
        using DType = typename ParamType::Type::PrimType;
        auto& gm = AscendC::Std::get<argNumber>(args);
        return BlockTensorT<DType>{gm};
    } else {
        return AscendC::Std::get<argNumber>(args);
    }
}

/**
 * @brief PrepareBlockParams 实现 (index_sequence 展开)
 * @tparam ParamsT 参数类型列表
 * @tparam ArgTup 参数 tuple 类型
 * @tparam BlockTensorT BlockTensor 模板
 * @tparam Ints 索引序列
 * @param args argTuple
 * @param tag BlockTensorTag 标签
 * @return tuple<BlockTensor...>
 */
template <typename ParamsT, typename ArgTup, template <typename> class BlockTensorT, std::size_t... Ints>
__aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>,
                                                        BlockTensorTag<BlockTensorT> tag)
{
    return AscendC::Std::make_tuple(ConstructBlockParam<Get_t<ParamsT, Ints>>(args, tag)...);
}

/**
 * @brief 批量构造 BlockTensor: 根据 Params 类型列表从 argTuple 构造
 * @tparam ParamsT 参数类型列表 (TypeList<Param<1>, Param<2>, ...>)
 * @tparam ArgTup 参数 tuple 类型
 * @tparam BlockTensorT BlockTensor 模板
 * @param argTuple argTuple (GM 偏移后的地址)
 * @param tag BlockTensorTag 标签
 * @return tuple<BlockTensor<T1>, BlockTensor<T2>, ...>
 */
template <typename ParamsT, typename ArgTup, template <typename> class BlockTensorT>
__aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple, BlockTensorTag<BlockTensorT> tag)
{
    return PrepareBlockParamsImpl<ParamsT>(argTuple, AscendC::Std::make_index_sequence<Size_v<ParamsT>>{}, tag);
}

/**
 * @brief 构造单个 LocalVar: 创建无 GM 绑定的空 BlockTensor
 * @tparam LocalVarT LocalVar 类型
 * @tparam BlockTensorT BlockTensor 模板
 * @param tag BlockTensorTag 标签
 * @return BlockTensor<T>{} (仅 UB, 无 GM)
 */
template <typename LocalVarT, template <typename> class BlockTensorT>
__aicore__ inline auto ConstructLocalVar(BlockTensorTag<BlockTensorT>)
{
    using DType = typename LocalVarT::Type::PrimType;
    return BlockTensorT<DType>{};
}

/**
 * @brief PrepareLocalVars 实现 (index_sequence 展开)
 * @tparam LocalVarsT LocalVar 类型列表
 * @tparam BlockTensorT BlockTensor 模板
 * @tparam Ints 索引序列
 * @param tag BlockTensorTag 标签
 * @return tuple<BlockTensor...>
 */
template <typename LocalVarsT, template <typename> class BlockTensorT, std::size_t... Ints>
__aicore__ inline auto PrepareLocalVarsImpl(AscendC::Std::index_sequence<Ints...>, BlockTensorTag<BlockTensorT> tag)
{
    return AscendC::Std::make_tuple(ConstructLocalVar<Get_t<LocalVarsT, Ints>>(tag)...);
}

/**
 * @brief 批量构造 LocalVar: 根据 LocalVars 类型列表创建空 BlockTensor
 * @tparam LocalVarsT LocalVar 类型列表
 * @tparam BlockTensorT BlockTensor 模板
 * @param tag BlockTensorTag 标签
 * @return tuple<BlockTensor<T1>, BlockTensor<T2>, ...> (无 GM 绑定)
 */
template <typename LocalVarsT, template <typename> class BlockTensorT>
__aicore__ inline auto PrepareLocalVars(BlockTensorTag<BlockTensorT> tag)
{
    return PrepareLocalVarsImpl<LocalVarsT>(AscendC::Std::make_index_sequence<Size_v<LocalVarsT>>{}, tag);
}

} // namespace Atvoss::Reduce
#endif // ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H
