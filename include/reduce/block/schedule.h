/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
#define ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
#include <functional>
#include <type_traits>
#include "block_info_tile.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "utils/buf_pool/block_buf_pool.h"
#include "reduce/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "common/type_def.h"
#include "graph/buffer.h"
#include "graph/expr_linearizer.h"
#include "graph/compute_preproc.h"
#include "operators/tile_shape.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Find_v;
using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
public:
    using ArchTag = ArchTagCfg;

private:
    static constexpr auto computeRes = ToLinearizerExpr(Compute{}.template Compute<BlockTensorTile>());
    static constexpr auto optimizedCompute = Atvoss::Tile::PreProcessComputeExpr<Policy.memPolicy>(computeRes);
    using ComputeInfoT = decltype(optimizedCompute);
    using ExprTile = typename ComputeInfoT::Expr;        // 表达式类型
    using ReduceDag = typename ComputeInfoT::Dag;        // DAG 类型
    using Params = typename ReduceDag::AllParams;        // 所有参数
    using InParams = typename ReduceDag::InParams;       // 输入参数
    using OutParams = typename ReduceDag::OutParams;     // 输出参数
    using LocalVars = typename ReduceDag::AllLocalVars;  // 临时变量

    template <size_t N, typename Params>
    static constexpr int FindParamsMaxTypeSizeImpl(int defaultSize = 1)
    {
        using ParamType = Get_t<Params, N>;
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DType = typename ParamType::Type::PrimType;
            if (defaultSize < sizeof(DType)) {
                defaultSize = sizeof(DType);
            }
        }

        constexpr int len = Size_v<Params>;
        if constexpr (N < len - 1) {
            return FindParamsMaxTypeSizeImpl<N + 1, Params>(defaultSize);
        }
        return defaultSize;
    }

    template <typename Params>
    static constexpr uint32_t FindParamsMaxTypeSize()
    {
        return FindParamsMaxTypeSizeImpl<0, Params>();
    }

    static constexpr uint32_t MaxSize = FindParamsMaxTypeSize<Params>();

public:
    template <typename Args, typename KernelScheduleCfg>
    static bool MakeScheduleConfig(
        const Args& arguments, const KernelScheduleCfg& kernelConfig, ScheduleCfg& blockConfig)
    {
        blockConfig.patternID = PATTERN_AR;
        blockConfig.basicBlock = 39424;
        blockConfig.resultBlock = 512;
        blockConfig.ubFactorA = 1;
        blockConfig.ubFactorR = 19712;
        blockConfig.tileACntPerCore = 16;
        blockConfig.tileRCntPerCore = 27;
        blockConfig.useNddma = 0;
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseBlockSchedule()
    {
        bufPools_.Init();
    }

    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        uint32_t patternID = cfg.patternID;
        if (patternID == PATTERN_EMPTY) {
            ProcessEmpty(cfg, argTuple);
        } else {
            Process(cfg, argTuple);
        }
    }

private:
    template <typename ArgTup>
    __aicore__ inline void ProcessEmpty(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        return;
    }

    template <typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        // 临时Tensor，无GM地址
        auto blockLocalVars = PrepareParams<LocalVars>();
        // 输入输出Tensor，绑定GM地址
        auto blockTensorsTile = PrepareBlockParams<Params>(argTuple);
        using BufferMaps = typename ReduceDag::BufMap;
        using ContextDataT =
            ContextData<decltype(blockTensorsTile), decltype(blockLocalVars), decltype(bufPools_), BufferMaps>;

        int32_t blockId = AscendC::GetBlockIdx();
        uint64_t loopAStartIndex = blockId * cfg.tileACntPerCore;
        uint64_t loopAEndIndex = loopAStartIndex + cfg.tileACntPerCore;
        if (unlikely(loopAEndIndex > cfg.tileACntPerCore)) {
            loopAEndIndex = cfg.tileACntPerCore;
        }
        // Normal Reduce
        if () {
            for (uint64_t i = loopAStartIndex; i < loopAEndIndex; i++) {
                ContextDataT context{blockTensorsTile, blockLocalVars, bufPools_, i * cfg.basicBlock,
                                     cfg.basicBlock, i & 1};
                Atvoss::Reduce::Tile::Evaluate<ExprTile>(context);
            }
        } else {    // Group Reduce
            return;
        }
    }

    template <typename ParamType>
    __aicore__ inline auto ConstructParam()
    {
        return ConstructParam<ParamType>(HasUsage<ParamType>{});
    }

    template <typename ParamType>
    __aicore__ inline auto ConstructParam(std::false_type)
    {
        using DType = typename ParamType::Type::PrimType;
        return BlockTensorTile<DType>{};
    }

    template <typename Params, size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Get_t<Params, Ints>>()...);
    }

    template <typename Params>
    __aicore__ inline auto PrepareParams()
    {
        return PrepareParamsImpl<Params>(AscendC::Std::make_index_sequence<Size_v<Params>>{});
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline constexpr auto ConstructBlockParam(ArgTup& args)
    {
        constexpr auto argNumber = ParamType::inplaceNumber - 1;
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DType = typename ParamType::Type::PrimType;
            auto& gm = AscendC::Std::get<argNumber>(args);
            return BlockTensorTile<DType>{gm};
        } else {
            return AscendC::Std::get<argNumber>(args);
        }
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructBlockParam<Get_t<Params, Ints>>(args)...);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple)
    {
        return PrepareBlockParamsImpl<Params>(argTuple, AscendC::Std::make_index_sequence<Size_v<Params>>{});
    }

private:
    AscendC::TPipe pipe_;
    // 待替换
    Atvoss::BlockBufferEx<MAX_BUFFER_COUNT, UB_TILE_SIZE> bufPools_;
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_