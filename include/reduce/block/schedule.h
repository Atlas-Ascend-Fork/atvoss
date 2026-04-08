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
#include "reduce/block/policy.h"
#include "reduce/block/block_info_tile.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "utils/buf_pool/block_buf_pool.h"
#include "reduce/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "common/type_def.h"
#include "reduce/graph/dag.h"
#include "graph/expr_linearizer.h"
#include "graph/compute_preproc.h"
#include "operators/tile_shape.h"
#include "reduce/utils/reduce_utils.h"

namespace Atvoss::Reduce {

using Atvoss::LocalVars_t;
using Atvoss::Params_t;
using Atvoss::Util::Concatenate_t;
using Atvoss::Util::Find_v;
using Atvoss::Util::FindLast_v;
using Atvoss::Util::ForEach;
using Atvoss::Util::Get_t;
using Atvoss::Util::Map_t;
using Atvoss::Util::Reverse_t;
using Atvoss::Util::Size_v;
using Atvoss::Util::TypeWrapper;

template <typename ExprList, typename Param>
struct FirstAndLastUse {
    static constexpr std::size_t firstUse = Find_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    static constexpr std::size_t lastUse = FindLast_v<Atvoss::HasParamNChecker<Param>::template Type, ExprList>;
    using Type =
        TypeList<Param, std::integral_constant<std::size_t, firstUse>, std::integral_constant<std::size_t, lastUse>>;
};

template <typename ExprList>
struct FirstAndLastUseFinder {
    template <typename Param>
    using Type = FirstAndLastUse<ExprList, Param>;
};

template <typename Expression, typename D>
struct ComputeInfo {
    using Expr = typename Expression::Type;
    using Dag = D;
};

template <bool IsBinaryAcc = true, typename T>
__host_aicore__ constexpr auto PreProcessComputeExpr(const Expression<T>& expr)
{
    using ExprT = T;
    using OriExprList = typename Atvoss::FlattenAtOpAndThen<ExprT>::Type;
    static_assert(Size_v<OriExprList> > 0, "Compute expression is empty.");

    using DagX = Graph::ReduceAutoDag<OriExprList, IsBinaryAcc>;

    using PreReduceList = typename DagX::PreReduceList;
    using ReducedList = typename DagX::ReducedList;
    using CombinedExprList = Concatenate_t<PreReduceList, ReducedList>;

    using AllParams = Params_t<CombinedExprList>;
    using AllLocalVars = LocalVars_t<CombinedExprList>;

    using ParamUseList = Map_t<FirstAndLastUseFinder<CombinedExprList>::template Type, AllParams>;
    using LocalVarUseList = Map_t<FirstAndLastUseFinder<CombinedExprList>::template Type, AllLocalVars>;

    auto result1 = ForEach(Reverse_t<ParamUseList>{}, Atvoss::Graph::AllocInserter{}, CombinedExprList{});
    auto result2 = ForEach(ParamUseList{}, Atvoss::Graph::FreeInserter{}, result1);
    auto result3 = ForEach(Reverse_t<LocalVarUseList>{}, Atvoss::Graph::AllocInserter{}, result2);
    auto result4 = ForEach(LocalVarUseList{}, Atvoss::Graph::FreeInserter{}, result3);

    using LastExpression = typename BuildExpression<decltype(result4)>::Type;
    return ComputeInfo<LastExpression, DagX>{};
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;    // DefaultBlockConfig
    using ArchTag = ArchTagCfg;

    static constexpr uint32_t MAX_BUFFER_COUNT = 1;
    static constexpr uint32_t MAX_UB_TILE_SIZE = 16 * 1024;
    static constexpr int32_t CACHE_BUF_SIZE = 16 * 1024;
    using BufPoolType = Atvoss::BlockBufferEx<MAX_BUFFER_COUNT, MAX_UB_TILE_SIZE>;

private:
    // Policy相关类型
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using TileShape = typename BlockPolicy::TileShape;

    // basicBlock计算
    static constexpr uint32_t BASIC_BLOCK = Atvoss::Tile::GetTotalElement<0, BlockPolicy>(1, 1);

    template <typename T>
    using BlockTensorTile = Atvoss::Reduce::BlockTensor<
        T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<BASIC_BLOCK, 1, 1>>>;

    // Compute函数生成的原始表达式，例：OpAssign<Param<2>, OpReduceSum<Param<1>>>
    static constexpr auto computeRes = ToLinearizerExpr(Compute{}.template Compute<BlockTensorTile>());

    // 优化表达式，插入Alloc/Free/CopyIn/CopyOut，例：OpAndThen<OpAlloc, OpAndThen<OpCopyIn, ...>>
    static constexpr auto optimizedCompute = PreProcessComputeExpr<BlockPolicy::EnableBinaryAcc>(computeRes);
    using ComputeInfoT = decltype(optimizedCompute);
    using ReduceDag = typename ComputeInfoT::Dag;           // DAG 类型

public:
    template <typename Args, typename KernelScheduleCfg>
    static bool MakeScheduleConfig(
        const Args& arguments, const KernelScheduleCfg& kernelConfig, ScheduleCfg& blockConfig)
    {
        blockConfig.patternID = PATTERN_AR;
        blockConfig.basicBlock = 59136;
        blockConfig.resultBlock = 512;
        blockConfig.ubFactorA = 7;
        blockConfig.ubFactorR = 2040;
        blockConfig.tileACntPerCore = 3;
        blockConfig.tileRCntPerCore = 3;
        blockConfig.useNddma = 0;
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseBlockSchedule() : actualBufSize_(MAX_UB_TILE_SIZE) {}

    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        CalculateActualBufSize(cfg);
        bufPools_.Init(actualBufSize_);

        if constexpr (IsEmpty<SchLoopInfo>()) {
            ProcessEmpty(cfg, argTuple);
        } else {
            Process<SchLoopInfo>(cfg, argTuple);
        }
    }

private:
    __aicore__ constexpr static uint32_t GetPreMte2Num()
    {
        return ReduceDag::template GetMTE2Num<true>() * CONST2;
    }

    __aicore__ constexpr static uint32_t GetPreMte3Num()
    {
        return ReduceDag::template GetMTE3Num<true>() * CONST2;
    }

    __aicore__ constexpr static uint32_t GetPreTempCalcNum()
    {
        return ReduceDag::template GetTempBufNum<true>();
    }

    __aicore__ constexpr static uint32_t GetPostMte2Num()
    {
        return ReduceDag::template GetMTE2Num<false>();
    }

    __aicore__ constexpr static uint32_t GetPostMte3Num()
    {
        return ReduceDag::template GetMTE3Num<false>();
    }

    __aicore__ constexpr static uint32_t GetPostTempCalcNum()
    {
        return ReduceDag::template GetTempBufNum<false>();
    }

    template <auto SchLoopInfo>
    __aicore__ inline constexpr static bool IsEmpty()
    {
        return SchLoopInfo->patternID == 0 && SchLoopInfo->innerPatternID == 0 &&
               SchLoopInfo->loopACount == 0 && SchLoopInfo->loopInnerACount == 0 &&
               SchLoopInfo->loopRCount == 0 && SchLoopInfo->loopInnerRCount == 0;
    }

    __aicore__ inline void CalculateActualBufSize(ScheduleCfg& cfg)
    {
        // 根据 blockConfig 计算实际需要的 tile 大小
        preBufSize_ = cfg.basicBlock * (PRE_MTE2_NUM_TOTAL + PRE_MTE3_NUM_TOTAL + PRE_TEMP_CALC_NUM);
        postBufSize_ = cfg.resultBlock * (POST_MTE2_NUM + POST_TEMP_CALC_NUM + POST_MTE3_NUM);
        resBufSize_ = cfg.resultBlock;

        // AscendC::printf("preBufSize_: %d", preBufSize_);
        // AscendC::printf("resBufSize_: %d", resBufSize_);
        actualBufSize_ = preBufSize_ + postBufSize_ + resBufSize_ + CACHE_BUF_SIZE;
    }

    template <typename ArgTup>
    __aicore__ inline void ProcessEmpty(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        return;
    }

    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {   
        if constexpr (Policy.EnableBinaryAcc) { // 二分Reduce
            return;
        } else {    // 顺序Reduce
            return;
        }
    }

private:
    AscendC::TPipe pipe_;
    BufPoolType bufPools_;
    uint32_t actualBufSize_;
    uint32_t preBufSize_ = 0;
    uint32_t postBufSize_ = 0;
    uint32_t resBufSize_ = 0;

    static constexpr int32_t PRE_MTE2_NUM_TOTAL = GetPreMte2Num();
    static constexpr int32_t PRE_MTE3_NUM_TOTAL = GetPreMte3Num();
    static constexpr int32_t PRE_TEMP_CALC_NUM = GetPreTempCalcNum();
    static constexpr int32_t POST_MTE2_NUM = GetPostMte2Num();
    static constexpr int32_t POST_MTE3_NUM = GetPostMte3Num();
    static constexpr int32_t POST_TEMP_CALC_NUM = GetPostTempCalcNum();
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_