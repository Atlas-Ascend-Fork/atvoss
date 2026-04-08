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
#include <type_traits>
#include "reduce/block/policy.h"
#include "reduce/block/block_info_tile.h"
#include "reduce/block/expr_preproc.h"
#include "reduce/block/buffer_analyzer.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "reduce/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "graph/expr_linearizer.h"
#include "operators/tile_shape.h"

namespace Atvoss::Reduce {

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;    // DefaultBlockConfig
    using ArchTag = ArchTagCfg;

private:
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using TileShape = typename BlockPolicy::TileShape;

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

    // Buffer 分析：DAG查询 + UB大小计算 + buffer pool初始化
    using BufAnalyzer = BlockBufferAnalyzer<ReduceDag>;

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
    __aicore__ inline BaseBlockSchedule() = default;

    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        if constexpr (IsEmpty<SchLoopInfo>()) {
            ProcessEmpty(cfg, argTuple);
        } else {
            bufAnalyzer_.Init(cfg);
            Process<SchLoopInfo>(cfg, argTuple);
        }
    }

private:
    template <auto SchLoopInfo>
    __aicore__ inline constexpr static bool IsEmpty()
    {
        return SchLoopInfo->patternID == 0 && SchLoopInfo->innerPatternID == 0 &&
               SchLoopInfo->loopACount == 0 && SchLoopInfo->loopInnerACount == 0 &&
               SchLoopInfo->loopRCount == 0 && SchLoopInfo->loopInnerRCount == 0;
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
    BufAnalyzer bufAnalyzer_;
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
