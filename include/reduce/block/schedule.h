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
 * @file schedule.h
 * @brief Reduce 算子 Block 调度模块
 */

#ifndef ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
#define ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
#include <type_traits>
#include "reduce/block/policy.h"
#include "reduce/block/block_info_tile.h"
#include "reduce/block/expr_preproc.h"
#include "reduce/block/buffer_analyzer.h"
#include "reduce/block/param_handler.h"
#include "reduce/utils/type_dump.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "reduce/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "graph/expr_linearizer.h"
#include "operators/tile_shape.h"

namespace Atvoss::Reduce {

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

/**
 * @brief Block 调度基类: 负责单核内的 tile 级数据搬运和计算调度
 * @tparam Compute 计算表达式类型 (如 ReduceSumCompute)
 * @tparam Policy Block 策略配置 (DefaultBlockPolicy)
 * @tparam ScheduleCfg Tiling 配置类型 (DefaultBlockConfig)
 * @tparam ArchTagCfg 架构标签 (DAV_3510)
 */
template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
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

    using BufAnalyzer = BufferAnalyzer<ReduceDag>;
    using BufPoolType = typename BufAnalyzer::BufPoolType;

    using CombinedList = Concatenate_t<typename ReduceDag::PreReduceList, typename ReduceDag::ReducedList>;
    using ParamsList = Atvoss::Params_t<CombinedList>;
    using LocalVarsList = Atvoss::LocalVars_t<CombinedList>;
    using BufferMapType = Concatenate_t<
        typename ReduceDag::PreReduceBufGen::BufferMapType,
        typename ReduceDag::ReducedBufGen::BufferMapType>;

    using PreReduceExpr = typename BuildExpression<typename ReduceDag::PreReduceList>::Type::Type;
    using ReducedRaw = typename ReduceDag::ReducedList;
    using ReduceOpType = Get_t<ReducedRaw, 0>;
    using PostReduceExpr = typename BuildExpression<Atvoss::Util::Drop_t<ReducedRaw, 1>>::Type::Type;

    using ReduceEval = Atvoss::Reduce::Tile::Evaluator<ReduceOpType>;
    using ElemT = typename ReduceEval::ElemT;

public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;
    using ArchTag = ArchTagCfg;
    using PreReduceParams = Atvoss::Params_t<typename ReduceDag::PreReduceList>;
    using PostReduceParams = Atvoss::Params_t<typename ReduceDag::ReducedList>;


#if !defined(__ATVOSS_HOST_ONLY__)
public:
    __aicore__ inline BaseBlockSchedule() = default;

    /**
     * @brief Block 层 schedule 调度入口
     * @tparam SchLoopInfo 调度循环信息
     * @tparam ArgTup 参数 tuple 类型
     * @param cfg Block 层 Tiling 配置
     * @param argTuple 参数 tuple
     */
    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        if constexpr (IsEmpty<SchLoopInfo>()) {
            ProcessEmpty(cfg, argTuple);
        } else {
            // AscendC::printf("block schedule run Normal: %d\n", 0x3);
            bufAnalyzer_.Init(cfg, bufPool_);
            Process<SchLoopInfo>(cfg, argTuple);
        }
    }

private:
    /**
     * @brief 判断是否空Tensor模板
     * @tparam SchLoopInfo 调度循环信息
     * @return 是否为空
     */
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

    /**
     * @brief 调度分发: 二分Reduce / 顺序Reduce / NormalReduce / GroupReduce
     * @tparam SchLoopInfo 调度循环信息
     * @tparam ArgTup 参数 tuple 类型
     * @param cfg Block 配置
     * @param argTuple 参数 tuple
     */
    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        if constexpr (Policy.EnableBinaryAcc) {
            auto blockTensors = PrepareBlockParams<ParamsList>(argTuple, BlockTensorTag<BlockTensorTile>{});
            auto localVars = PrepareLocalVars<LocalVarsList>(BlockTensorTag<BlockTensorTile>{});

            // DumpReduceTypeInfo<ReduceDag, PreReduceExpr, PostReduceExpr>::Print();

            if constexpr (SchLoopInfo->loopRCount == 0) {
                NormalReduce<SchLoopInfo>(cfg, blockTensors, localVars);
            } else {
                GroupReduce(cfg, blockTensors, localVars);
            }
        } else {
            return;
        }
    }

    /**
     * @brief 计算输入 tile 在 GM 中的偏移
     * @param aIdx A 轴索引
     * @param rIdx R 轴索引
     * @param cfg Block 配置
     * @return GM 偏移 (元素偏移)
     */
    __aicore__ inline uint64_t InputGmOffset(uint64_t aIdx, uint64_t rIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.ubFactorA * cfg.basicBlock + rIdx * cfg.basicBlock;
    }

    /**
     * @brief 计算输出在 GM 中的偏移
     * @param aIdx A 轴索引
     * @param cfg Block 配置
     * @return GM 偏移
     */
    __aicore__ inline uint64_t OutputGmOffset(uint64_t aIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.resultBlock;
    }

    /**
     * @brief 查找不大于 value 的最大 2 的幂次
     * @param value 输入值
     * @return 2 的幂次值
     */
    __aicore__ inline uint64_t FindNearestPower2(uint64_t value)
    {
        if (value <= 1) return value;
        uint64_t power = 1;
        while (power < value) power <<= 1;
        return (power > value) ? (power >> 1) : power;
    }

    /**
     * @brief 单核内二分 Reduce (NormalReduce)
     *
     * 例: R块数=25, 二次幂点=16, tail=9
     *   Phase 1 (Tail): 尾块[16,25) 与主块前部[0,9) 配对处理
     *     Evaluate<PreReduceExpr>(尾) → 保存ub → Evaluate<PreReduceExpr>(主)
     *     → DoPadding×2 → BinaryAccPreProcess → Compute → UpdateCache
     *   Phase 2 (Main): 主块剩余[9,16) 逐块处理
     *     Evaluate<PreReduceExpr>(主) → DoPadding → Compute → UpdateCache
     *   Phase 3 (CopyOut): Evaluate<PostReduceExpr>
     *
     * @tparam SchLoopInfo 调度循环信息
     * @tparam BlockTensors BlockTensor tuple 类型
     * @tparam LocalVarsTuple LocalVar tuple 类型
     * @param cfg Block 配置
     * @param blockTensors BlockTensor 参数列表
     * @param localVars LocalVar 列表
     */
    template <auto SchLoopInfo, typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void NormalReduce(ScheduleCfg& cfg, BlockTensors& blockTensors, LocalVarsTuple& localVars)
    {
        uint64_t aStart = AscendC::GetBlockIdx() * cfg.tileACntPerCore;
        uint64_t aEnd = aStart + cfg.tileACntPerCore;
        if (unlikely(aEnd > cfg.totalTileACnt)) {
            aEnd = cfg.totalTileACnt;
        }
        uint64_t rCount = cfg.tileRCntPerCore;

        // AscendC::printf("block NormalReduce aEnd: %d\n", aEnd);
        // AscendC::printf("block NormalReduce rCount: %d\n", rCount);

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

    /**
     * @brief 执行 PreReduce 表达式: Alloc + CopyIn
     * @tparam BlockTensors BlockTensor tuple 类型
     * @tparam LocalVarsTuple LocalVar tuple 类型
     * @param blockTensors BlockTensor 参数列表
     * @param localVars LocalVar 列表
     * @param gmOffset GM 偏移
     * @param elemNum 元素数 (basicBlock)
     * @param pingPong 乒乓缓冲标识
     * @param segCount A 轴分段数
     * @param segSize 每段 R 轴元素数
     * @param srcStride GM 段间 stride
     * @param dstStride UB 段间 stride
     */
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

    /**
     * @brief Phase 1 归约: Padding×2 → BinaryAccPreProcess → Compute → UpdateCache
     *
     * 尾块和主块分别做 32B 对齐填充后, 通过 BinaryAccPreProcess 完成 UB 间累加,
     * 然后对累加结果执行硬件归约和二分数缓存归并.
     *
     * @tparam SchLoopInfo 调度循环信息 (提供 patternID)
     * @tparam BlockTensors BlockTensor tuple 类型
     * @param blockTensors BlockTensor 参数列表
     * @param tailUb 尾块 LocalTensor
     * @param mainUb 主块 LocalTensor
     * @param pingPong 乒乓缓冲标识
     */
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

    /**
     * @brief Phase 2 归约: Padding → Compute → UpdateCache
     *
     * 单 tile 处理, 无 BinaryAccPreProcess.
     *
     * @tparam SchLoopInfo 调度循环信息
     * @tparam BlockTensors BlockTensor tuple 类型
     * @param blockTensors BlockTensor 参数列表
     * @param pingPong 乒乓缓冲标识
     */
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

    /**
     * @brief 执行 PostReduce 表达式: CopyOut + Free
     * @tparam BlockTensors BlockTensor tuple 类型
     * @tparam LocalVarsTuple LocalVar tuple 类型
     * @param blockTensors BlockTensor 参数列表
     * @param localVars LocalVar 列表
     * @param gmOffset GM 偏移
     * @param elemNum 元素数 (resultBlock)
     * @param pingPong 乒乓缓冲标识
     * @param segCount A 轴分段数
     * @param segSize 每段 R 轴元素数
     * @param srcStride GM 段间 stride
     * @param dstStride UB 段间 stride
     */
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

    /**
     * @brief 跨核 Reduce (GroupReduce), 暂未实现
     * @tparam BlockTensors BlockTensor tuple 类型
     * @tparam LocalVarsTuple LocalVar tuple 类型
     * @param cfg Block 配置
     * @param blockTensors BlockTensor 参数列表
     * @param localVars LocalVar 列表
     */
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

/**
 * @brief 默认 Block 调度类
 * @tparam Compute 计算表达式类型
 * @tparam Policy Block 策略配置
 * @tparam ScheduleCfg Tiling 配置类型
 * @tparam ArchTag 架构标签
 */
template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_
