/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H
#define ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H

#include "block_info_tile.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

template <template <typename> class BlockTensorT>
struct BlockTensorTag {};

// ── BlockTensor 构造: Param + GM地址 → BlockTensor ──
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

template <typename ParamsT, typename ArgTup, template <typename> class BlockTensorT, std::size_t... Ints>
__aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>,
                                                        BlockTensorTag<BlockTensorT> tag)
{
    return AscendC::Std::make_tuple(ConstructBlockParam<Get_t<ParamsT, Ints>>(args, tag)...);
}

template <typename ParamsT, typename ArgTup, template <typename> class BlockTensorT>
__aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple, BlockTensorTag<BlockTensorT> tag)
{
    return PrepareBlockParamsImpl<ParamsT>(argTuple, AscendC::Std::make_index_sequence<Size_v<ParamsT>>{}, tag);
}

template <typename LocalVarT, template <typename> class BlockTensorT>
__aicore__ inline auto ConstructLocalVar(BlockTensorTag<BlockTensorT>)
{
    using DType = typename LocalVarT::Type::PrimType;
    return BlockTensorT<DType>{};
}

template <typename LocalVarsT, template <typename> class BlockTensorT, std::size_t... Ints>
__aicore__ inline auto PrepareLocalVarsImpl(AscendC::Std::index_sequence<Ints...>, BlockTensorTag<BlockTensorT> tag)
{
    return AscendC::Std::make_tuple(ConstructLocalVar<Get_t<LocalVarsT, Ints>>(tag)...);
}

template <typename LocalVarsT, template <typename> class BlockTensorT>
__aicore__ inline auto PrepareLocalVars(BlockTensorTag<BlockTensorT> tag)
{
    return PrepareLocalVarsImpl<LocalVarsT>(AscendC::Std::make_index_sequence<Size_v<LocalVarsT>>{}, tag);
}

} // namespace Atvoss::Reduce
#endif // ATVOSS_REDUCE_BLOCK_PARAM_HANDLER_H
