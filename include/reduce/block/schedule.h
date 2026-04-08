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
#include "reduce/block/param_handler.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "reduce/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "graph/expr_linearizer.h"
#include "operators/tile_shape.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

template <typename OpList, typename AccList = TypeList<>>
struct WrapWithAllocFree;

template <typename LHS, typename RHS, typename... Rest, typename... Acc>
struct WrapWithAllocFree<TypeList<OpAssign<LHS, RHS>, Rest...>, TypeList<Acc...>> {
    using Type = typename WrapWithAllocFree<
        TypeList<Rest...>,
        TypeList<Acc..., OpAlloc<LHS>, OpAssign<LHS, RHS>>
    >::Type;
};

template <typename... Acc>
struct WrapWithAllocFree<TypeList<>, TypeList<Acc...>> {
    using Type = TypeList<Acc...>;
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;
    using ArchTag = ArchTagCfg;

private:
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using TileShape = typename BlockPolicy::TileShape;

    static constexpr uint32_t BASIC_BLOCK = Atvoss::Tile::GetTotalElement<0, BlockPolicy>(1, 1);

    template <typename T>
    using BlockTensorTile = Atvoss::Reduce::BlockTensor<
        T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<BASIC_BLOCK, 1, 1>>>;

    static constexpr auto computeRes = ToLinearizerExpr(Compute{}.template Compute<BlockTensorTile>());
    static constexpr auto optimizedCompute = PreProcessComputeExpr<BlockPolicy::EnableBinaryAcc>(computeRes);
    using ComputeInfoT = decltype(optimizedCompute);
    using ReduceDag = typename ComputeInfoT::Dag;
    using ExprTile = typename ComputeInfoT::Expr;

    using BufAnalyzer = BlockBufferAnalyzer<ReduceDag>;
    using BufPoolType = typename BufAnalyzer::BufPoolType;

    using CombinedList = Concatenate_t<typename ReduceDag::PreReduceList, typename ReduceDag::ReducedList>;
    using Params = Atvoss::Params_t<CombinedList>;
    using LocalVarsList = Atvoss::LocalVars_t<CombinedList>;

    using BufferMapType = Concatenate_t<
        typename ReduceDag::PreReduceBufGen::BufferMapType,
        typename ReduceDag::ReducedBufGen::BufferMapType>;

    using PreReduceWrapped = typename WrapWithAllocFree<typename ReduceDag::PreReduceList>::Type;
    using PreReduceExpr = typename BuildExpression<PreReduceWrapped>::Type::Type;

    using ReducedRaw = typename ReduceDag::ReducedList;
    using ReduceOpType = Get_t<ReducedRaw, 0>;
    using CopyOutOp = Get_t<ReducedRaw, Size_v<ReducedRaw> - 1>;
    using PostReduceWrapped = typename WrapWithAllocFree<TypeList<CopyOutOp>>::Type;
    using PostReduceExpr = typename BuildExpression<PostReduceWrapped>::Type::Type;

    using ReduceEval = Atvoss::Reduce::Tile::Evaluator<ReduceOpType>;
    using ElemT = typename ReduceEval::ElemT;

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
            bufAnalyzer_.Init(cfg, bufPool_);
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
        if constexpr (Policy.EnableBinaryAcc) {
            auto blockTensors = PrepareBlockParams<Params>(argTuple, BlockTensorTag<BlockTensorTile>{});
            auto localVars = PrepareLocalVars<LocalVarsList>(BlockTensorTag<BlockTensorTile>{});

            if constexpr (SchLoopInfo->loopRCount == 0) {
                NormalReduce<SchLoopInfo>(cfg, blockTensors, localVars);
            } else {
                GroupReduce(cfg, blockTensors, localVars);
            }
        } else {
            return;
        }
    }

    __aicore__ inline uint64_t InputGmOffset(uint64_t aIdx, uint64_t rIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.ubFactorA * cfg.basicBlock + rIdx * cfg.basicBlock;
    }

    __aicore__ inline uint64_t OutputGmOffset(uint64_t aIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.resultBlock;
    }

    __aicore__ inline uint64_t FindNearestPower2(uint64_t value)
    {
        if (value <= 1) return value;
        uint64_t power = 1;
        while (power < value) power <<= 1;
        return (power > value) ? (power >> 1) : power;
    }

    // ── NormalReduce: 单核内二分Reduce ──
    // 例: R块数=25, 二次幂点=16, tail=9
    //   Phase 1 (Tail): 尾块[16,25) 与主块前部[0,9) 配对
    //     Evaluate<PreReduceExpr>(尾块) → 保存ub → Evaluate<PreReduceExpr>(主块)
    //     → CallDoPadding×2 → BinaryAccPreProcess → CallCompute → CallUpdateCache
    //   Phase 2 (Main): 主块剩余[9,16) 逐块
    //     Evaluate<PreReduceExpr>(主块)
    //     → CallDoPadding → CallCompute → CallUpdateCache
    //   Phase 3 (CopyOut): Evaluate<PostReduceExpr>
    //
    template <auto SchLoopInfo, typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void NormalReduce(ScheduleCfg& cfg, BlockTensors& blockTensors, LocalVarsTuple& localVars)
    {
        uint64_t aStart = AscendC::GetBlockIdx() * cfg.tileACntPerCore;
        uint64_t aEnd = aStart + cfg.tileACntPerCore;
        if (unlikely(aEnd > cfg.totalTileACnt)) aEnd = cfg.totalTileACnt;
        uint64_t rCount = cfg.tileRCntPerCore;

        uint64_t segCount = cfg.ubFactorA;
        uint64_t segSize  = cfg.ubFactorR;
        int32_t aSplitAxis = SchLoopInfo->loopAAxis[SchLoopInfo->loopACount - 1];
        uint64_t srcStride = (SchLoopInfo->loopACount > 0) ? cfg.srcStride[aSplitAxis] : 0;

        uint64_t bisectionPos = FindNearestPower2(rCount);
        uint64_t tailCount = rCount - bisectionPos;

        for (uint64_t aIdx = aStart; aIdx < aEnd; aIdx++) {
            uint32_t pingPong = 0;

            // ── Phase 1: Tail ──
            for (uint64_t i = 0; i < tailCount; i++) {
                uint64_t tailOffset = InputGmOffset(aIdx, bisectionPos + i, cfg);
                EvaluatePreReduce(blockTensors, localVars, tailOffset, cfg.basicBlock, pingPong,
                                  segCount, segSize, srcStride, segSize);
                auto tailUb = AscendC::Std::get<0>(blockTensors).GetUbTensor();

                uint64_t mainOffset = InputGmOffset(aIdx, i, cfg);
                EvaluatePreReduce(blockTensors, localVars, mainOffset, cfg.basicBlock, pingPong ^ 1,
                                  segCount, segSize, srcStride, segSize);
                auto& mainUb = AscendC::Std::get<0>(blockTensors).GetUbTensor();

                EvaluateReducePair<SchLoopInfo>(blockTensors, tailUb, mainUb, pingPong);
                pingPong ^= 1;
            }

            // ── Phase 2: Main ──
            for (uint64_t i = tailCount; i < bisectionPos; i++) {
                uint64_t mainOffset = InputGmOffset(aIdx, i, cfg);
                EvaluatePreReduce(blockTensors, localVars, mainOffset, cfg.basicBlock, pingPong,
                                  segCount, segSize, srcStride, segSize);

                EvaluateReduceSingle<SchLoopInfo>(blockTensors, pingPong);
                pingPong ^= 1;
            }

            // ── Phase 3: CopyOut ──
            uint64_t outOffset = OutputGmOffset(aIdx, cfg);
            EvaluatePostReduce(blockTensors, localVars, outOffset, cfg.resultBlock, pingPong,
                               segCount, segSize, srcStride, segSize);
        }
    }

    // ── EvaluatePreReduce: Alloc + CopyIn + NormalOp, 通过 Expr 的 Evaluate 驱动 ──
    template <typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void EvaluatePreReduce(BlockTensors& blockTensors, LocalVarsTuple& localVars,
                                             uint64_t gmOffset, uint64_t elemNum, uint32_t pingPong,
                                             uint64_t segCount, uint64_t segSize,
                                             uint64_t srcStride, uint64_t dstStride)
    {
        using CTX = ContextData<BlockTensors, LocalVarsTuple, BufPoolType, BufferMapType>;
        CTX ctx{blockTensors, localVars, bufPool_, gmOffset, elemNum, pingPong,
                segCount, segSize, srcStride, dstStride, 0, 0};
        Atvoss::Reduce::Tile::Evaluate<PreReduceExpr>(ctx);
    }

    // ── EvaluateReducePair (Phase 1): Padding×2 → BinaryAccPreProcess → Compute → UpdateCache ──
    template <auto SchLoopInfo, typename BlockTensors>
    __aicore__ inline void EvaluateReducePair(BlockTensors& blockTensors,
                                              AscendC::LocalTensor<ElemT>& tailUb,
                                              AscendC::LocalTensor<ElemT>& mainUb,
                                              uint32_t pingPong)
    {
        using Pattern = typename GetPattern<SchLoopInfo->patternID>::T;
        ReduceEval eval{};
        auto& dstUb = AscendC::Std::get<1>(blockTensors).GetUbTensor();
        auto shape = Atvoss::Tile::GetShape<Atvoss::Tile::Operation::Binary>(blockTensors);
        ElemT padValue = static_cast<ElemT>(0);

        eval.template DoPadding<decltype(shape), Pattern>(tailUb, shape, padValue);
        eval.template DoPadding<decltype(shape), Pattern>(mainUb, shape, padValue);
        eval.BinaryAccPreProcess(dstUb, tailUb, mainUb, shape);
        eval.template Compute<decltype(shape), Pattern>(dstUb, dstUb, shape);

        if constexpr (SchLoopInfo->innerPatternID > 0) {
            int64_t idx = static_cast<int64_t>(pingPong);
            eval.template UpdateCache<decltype(shape), Pattern>(dstUb, dstUb, shape, idx);
        }
    }

    // ── EvaluateReduceSingle (Phase 2): Padding → Compute → UpdateCache ──
    template <auto SchLoopInfo, typename BlockTensors>
    __aicore__ inline void EvaluateReduceSingle(BlockTensors& blockTensors, uint32_t pingPong)
    {
        using Pattern = typename GetPattern<SchLoopInfo->patternID>::T;
        ReduceEval eval{};
        auto& srcUb = AscendC::Std::get<0>(blockTensors).GetUbTensor();
        auto& dstUb = AscendC::Std::get<1>(blockTensors).GetUbTensor();
        auto shape = Atvoss::Tile::GetShape<Atvoss::Tile::Operation::Binary>(blockTensors);
        ElemT padValue = static_cast<ElemT>(0);

        eval.template DoPadding<decltype(shape), Pattern>(srcUb, shape, padValue);
        eval.template Compute<decltype(shape), Pattern>(dstUb, srcUb, shape);

        if constexpr (SchLoopInfo->innerPatternID > 0) {
            int64_t idx = static_cast<int64_t>(pingPong);
            eval.template UpdateCache<decltype(shape), Pattern>(dstUb, dstUb, shape, idx);
        }
    }

    // ── EvaluatePostReduce: NormalOp + CopyOut + Free, 通过 Expr 的 Evaluate 驱动 ──
    template <typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void EvaluatePostReduce(BlockTensors& blockTensors, LocalVarsTuple& localVars,
                                              uint64_t gmOffset, uint64_t elemNum, uint32_t pingPong,
                                              uint64_t segCount, uint64_t segSize,
                                              uint64_t srcStride, uint64_t dstStride)
    {
        using CTX = ContextData<BlockTensors, LocalVarsTuple, BufPoolType, BufferMapType>;
        CTX ctx{blockTensors, localVars, bufPool_, gmOffset, elemNum, pingPong,
                segCount, segSize, srcStride, dstStride, 0, 0};
        Atvoss::Reduce::Tile::Evaluate<PostReduceExpr>(ctx);
    }

    template <typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void GroupReduce(ScheduleCfg& cfg, BlockTensors& blockTensors, LocalVarsTuple& localVars)
    {
        return;
    }

private:
    AscendC::TPipe pipe_;
    BufPoolType bufPool_;
    BufAnalyzer bufAnalyzer_;
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
