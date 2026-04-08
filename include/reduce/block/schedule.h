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

using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

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

    // 合并 PreReduce + Reduced 两阶段的 BufferMap
    using BufferMapType = Concatenate_t<
        typename ReduceDag::PreReduceBufGen::BufferMapType,
        typename ReduceDag::ReducedBufGen::BufferMapType>;

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

    // ── Process: NormalReduce / GroupReduce 分发 ──
    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        if constexpr (Policy.EnableBinaryAcc) {
            auto blockTensors = PrepareBlockParams<Params>(argTuple);
            auto localVars = PrepareLocalVars<LocalVarsList>();

            if constexpr (SchLoopInfo->loopRCount == 0) {
                NormalReduce(cfg, blockTensors, localVars);
            } else {
                GroupReduce(cfg, blockTensors, localVars);
            }
        } else {
            // 顺序Reduce: 待实现
        }
    }

    // ── BlockTensor 构造: 将 argTuple 中的 GM 地址绑定到 BlockTensor ──
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

    template <typename ParamsT, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructBlockParam<Get_t<ParamsT, Ints>>(args)...);
    }

    template <typename ParamsT, typename ArgTup>
    __aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple)
    {
        return PrepareBlockParamsImpl<ParamsT>(argTuple, AscendC::Std::make_index_sequence<Size_v<ParamsT>>{});
    }

    // ── LocalVar 构造: 创建 UB 上的临时 Tensor ──
    template <typename LocalVarT>
    __aicore__ inline auto ConstructLocalVar()
    {
        using DType = typename LocalVarT::Type::PrimType;
        return BlockTensorTile<DType>{};
    }

    template <typename LocalVarsT, std::size_t... Ints>
    __aicore__ inline auto PrepareLocalVarsImpl(AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructLocalVar<Get_t<LocalVarsT, Ints>>()...);
    }

    template <typename LocalVarsT>
    __aicore__ inline auto PrepareLocalVars()
    {
        return PrepareLocalVarsImpl<LocalVarsT>(AscendC::Std::make_index_sequence<Size_v<LocalVarsT>>{});
    }

    // ── GM 偏移计算 ──
    // 输入: 每个 tile 在 GM 中的起始位置 = a偏移 + r偏移
    //    a偏移 = aIdx * (ubFactorA维度的stride)  — 非归约维度的偏移
    //    r偏移 = rIdx * basicBlock                  — 归约维度内按 tile 大小步进
    __aicore__ inline uint64_t InputGmOffset(uint64_t aIdx, uint64_t rIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.ubFactorA * cfg.basicBlock + rIdx * cfg.basicBlock;
    }

    // 输出: 每个 A 位置产生一个结果块，大小为 resultBlock
    __aicore__ inline uint64_t OutputGmOffset(uint64_t aIdx, ScheduleCfg& cfg)
    {
        return aIdx * cfg.resultBlock;
    }

    // ── NormalReduce: 单核内二分Reduce ──
    //
    // 算法: 对于每个非归约维度的 A-tile:
    //         流式处理归约维度的 R-tile 序列
    //         每个 R-tile 经历: CopyIn(从GM加载) → Reduce(归约+累加) → (最后一tile时)CopyOut(写回GM)
    //         累加使用 ping-pong 双缓冲: 偶数tile→ping, 奇数tile→pong
    //
    // 为什么叫"二分":
    //   表达式内部的归约算子(如ReduceSum)在UB内部使用二分树归约,
    //   Block层通过ping-pong缓冲管理实现tile间的流式二分累加,
    //   避免单个累加buffer成为数据依赖瓶颈。
    //
    template <typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void NormalReduce(ScheduleCfg& cfg, BlockTensors& blockTensors, LocalVarsTuple& localVars)
    {
        uint64_t aStart = AscendC::GetBlockIdx() * cfg.tileACntPerCore;
        uint64_t aEnd = aStart + cfg.tileACntPerCore;
        if (aEnd > cfg.totalTileACnt) {
            aEnd = cfg.totalTileACnt;
        }
        uint64_t rCount = cfg.tileRCntPerCore;

        for (uint64_t aIdx = aStart; aIdx < aEnd; aIdx++) {
            uint32_t pingPong = 0;

            for (uint64_t rIdx = 0; rIdx < rCount; rIdx++) {
                uint64_t gmOffset = InputGmOffset(aIdx, rIdx, cfg);
                EvaluateTile(blockTensors, localVars, gmOffset, cfg.basicBlock, pingPong);
                pingPong ^= 1;
            }
        }
    }

    // ── 单 Tile 执行 ──
    //
    // Reduce 管线 (由 ExprTile 驱动, 通过 Evaluator 逐步执行):
    //   1. CopyIn  (ARAR→AARR转置): GM交织布局转置为UB分块布局
    //   2. Padding (32B对齐填充):   对齐尾部填入安全值 (sum→0, prod→1, min→+∞, max→-∞)
    //   3. Reduce  (AR/RA pattern): 调用 AscendC::ReduceSum 硬件归约
    //   4. Cache   (二分缓存归并):   VReg→Cache 多级二分树最终归并成单值
    //   5. CopyOut (写回GM):        最终结果写回 GM 输出地址
    //
    // Context 含义:
    //   gmOffset  - CopyIn/CopyOut 的 GM 地址偏移 (InputGmOffset / OutputGmOffset)
    //   elemNum   - 当前 tile 的有效元素数 (basicBlock)
    //   pingPong  - 累加缓冲选择 (0→ping组, 1→pong组)
    //   bufPools  - UB buffer 池, 供 Alloc/Free 管理
    //   BuffMaps  - 编译期 BufferId→node 映射表, GetBufferId 运行时查询
    //
    template <typename BlockTensors, typename LocalVarsTuple>
    __aicore__ inline void EvaluateTile(BlockTensors& blockTensors, LocalVarsTuple& localVars,
                                        uint64_t gmOffset, uint64_t elemNum, uint32_t pingPong)
    {
        using ContextDataT = ContextData<BlockTensors, LocalVarsTuple, BufPoolType, BufferMapType>;
        ContextDataT context{blockTensors, localVars, bufPool_, gmOffset, elemNum, pingPong};
        Atvoss::Reduce::Tile::Evaluate<ExprTile>(context);
    }

    // ── GroupReduce: 跨核Reduce ──
    // 在 Kernel 层由两次 BlockOp::Run + SyncAll 实现:
    //   第一次 Run: 各核独立完成核内Reduce (本函数)
    //   SyncAll: 核间同步
    //   第二次 Run: 各核处理核间Reduce结果 (本函数, 此时 GM 偏移指向核间中间结果)
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
