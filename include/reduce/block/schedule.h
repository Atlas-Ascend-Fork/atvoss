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
    
    // 使用固定模板参数
    using BufPoolType = Atvoss::BlockBufferEx<MAX_BUFFER_COUNT, MAX_UB_TILE_SIZE>;

private:
    // Policy相关类型
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using TileShape = typename BlockPolicy::TileShape;

    // basicBlock计算
    static constexpr uint32_t BASIC_BLOCK = Atvoss::Tile::GetTotalElement<0, BlockPolicy>(1, 1);

//     // Shape相关
//     using ShapeT = Atvoss::Tile::Shape_t<BlockPolicy>;
//     using ShapeSize = Atvoss::Tile::ShapeSize<ShapeT>;

    template <typename T>
    using BlockTensorTile = Atvoss::Reduce::BlockTensor<
        T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<BASIC_BLOCK, 1, 1>>>;

    // Compute函数生成的原始表达式，例：OpAssign<Param<2>, OpReduceSum<Param<1>>>
    static constexpr auto computeRes = ToLinearizerExpr(Compute{}.template Compute<BlockTensorTile>());

    // 优化表达式，插入Alloc/Free/CopyIn/CopyOut，例：OpAndThen<OpAlloc, OpAndThen<OpCopyIn, ...>>
    static constexpr auto optimizedCompute = PreProcessComputeExpr<BlockPolicy::EnableBinaryAcc>(computeRes);
    using ComputeInfoT = decltype(optimizedCompute);
//     using ExprTile = typename ComputeInfoT::Expr;        // 表达式类型，OpAndThen<...> 类型链
    using ReduceDag = typename ComputeInfoT::Dag;        // DAG 类型
//     using Params = typename ReduceDag::AllParams;        // 所有参数列表，例：TypeList<Param<1, IN>, Param<2, OUT>>
//     using InParams = typename ReduceDag::InParams;       // 输入参数列表，例：TypeList<Param<1, IN>>
//     using OutParams = typename ReduceDag::OutParams;     // 输出参数列表，例：TypeList<Param<1, OUT>>
//     using LocalVars = typename ReduceDag::AllLocalVars;  // 临时变量列表，例：TypeList<LocalVar<1>>

//     static constexpr uint32_t IN_PARAMS_COUNT = Size_v<InParams> * 2;
//     static constexpr uint32_t OUT_PARAMS_COUNT = Size_v<OutParams> * 2;
//     static constexpr uint32_t LOCAL_VAR_COUNT = Size_v<LocalVars>;
//     static constexpr uint32_t MAX_BUFFER_COUNT = IN_PARAMS_COUNT + OUT_PARAMS_COUNT + LOCAL_VAR_COUNT;
//     static constexpr uint32_t UB_TILE_SIZE = ArchTag::UB_SIZE / MAX_BUFFER_COUNT / 1024 * 1024;

//     using BlockLocalVarsType = decltype(PrepareParams<LocalVars>());
//     using BlockTensorsTileType = decltype(PrepareBlockParams<Params>
//                                           (std::declval<AscendC::Std::tuple<__gm__ uint8_t*>>()));
//     using BufferMaps = typename ReduceDag::BufMap;
//     using ContextDataT = ContextData<BlockTensorsTileType, BlockLocalVarsType,
//                                      Atvoss::BlockBufferEx<MAX_BUFFER_COUNT, UB_TILE_SIZE>, BufferMaps>;

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
    __aicore__ inline BaseBlockSchedule() : actualTileSize_(MAX_UB_TILE_SIZE) {}

    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        // uint32_t patternID = cfg.patternID;
        CalculateActualTileSize(cfg);
        bufPools_.Init(actualTileSize_);


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

    __aicore__ inline void CalculateActualTileSize(ScheduleCfg& cfg)
    {
        // 根据 blockConfig 计算实际需要的 tile 大小
        preBufSize_ = cfg.basicBlock * (PRE_MTE2_NUM_TOTAL + PRE_MTE3_NUM_TOTAL + PRE_TEMP_CALC_NUM);
        postBufSize_ = cfg.resultBlock * (POST_MTE2_NUM + POST_TEMP_CALC_NUM + POST_MTE3_NUM);
        resBufSize_ = cfg.resultBlock;

        
        actualTileSize_ = preBufSize_ + postBufSize_ + resBufSize_ + CACHE_BUF_SIZE;
    }

//     template <typename ArgTup>
//     __aicore__ inline void Initialize(ScheduleCfg& cfg, ArgTup& argTuple)
//     {   
//         // 临时Tensor，无GM地址
//         blockLocalVars_ = PrepareParams<LocalVars>();
//         // 输入输出Tensor，绑定GM地址
//         blockTensorsTile_ = PrepareBlockParams<Params>(argTuple);

//         ubFactorA_ = cfg.ubFactorA;
//         ubFactorR_ = cfg.ubFactorR;
//     }

    // __aicore__ inline void SetLoopRange(ScheduleCfg& cfg)
    // {
    //     int32_t blockId = AscendC::GetBlockIdx();
    //     loopAStartIndex_ = blockId * cfg.tileACntPerCore;
    //     loopAEndIndex_ = loopAStartIndex_ + cfg.tileACntPerCore;
    //     if (unlikely(loopAEndIndex_ > cfg.tileACntPerCore)) {
    //         loopAEndIndex_ = cfg.tileACntPerCore;
    //     }
    //     // TODO: loopACount、loopAAxis、loopInnerRCount、loopInnerRAxis
    //     int32_t aAxisIdx = loopACount - 1;
    //     int32_t aAxis = loopAAxis[aAxisIdx];
    //     loopAAxisStep_ = CeilDiv(cfg.shape[aAxis], ubFactorA_);
    //     if (loopInnerRCount > 0) {
    //         int32_t rAxisIdx = loopInnerRCount - 1;
    //         int32_t rAxis = loopInnerRAxis[rAxisIdx];
    //         loopRAxisStep_ = CeilDiv(cfg.shape[rAxis], ubFactorR_);
    //     }
    // }

    // __aicore__ inline uint64_t FindNearestPower2(const uint64_t value)
    // {
    //     if (value == 0) {
    //         return 0;
    //     } else if (value <= CONST2) {
    //         return CONST1;
    //     } else if (value <= CONST4) {
    //         return CONST2;
    //     } else {
    //         const uint64_t num = value - CONST1;
    //         const uint64_t pow = CONST_SIXTY_THREE - AscendC::ScalarCountLeadingZero(num);
    //         return (CONST1 << pow);
    //     }
    // }

    // __aicore__ inline uint64_t CalLog2(uint64_t value)
    // {
    //     uint64_t res = 0;
    //     while (value > 1) {
    //         value = value >> 1;
    //         res++;
    //     }
    //     return res;
    // }

    // __aicore__ inline void SetIterRRange(ScheduleCfg& cfg)
    // {
    //     uint64_t rCount = 0;
    //     if constexpr (LoopInfo->loopRCount == 0) {
    //         rCount = cfg.tileRCntPerCore;
    //     } else {
    //         rCount = loopREndIndex_ - loopRStartIndex_;
    //     }
    //     bisectionPos_ = FindNearestPower2(rCount);
    //     cacheCount_ = CalLog2(bisectionPos_) + 1;
    //     bisectionTail_ = rCount - bisectionPos_;
    // }

    // __aicore__ inline void LinearCompute(int32_t aIndex, int32_t& startIdx)
    // {
    //     int32_t idx = startIdx;
    //     for (uint64_t i = 0; i < bisectionTail_; i++) {
    //         ContextDataT context{blockTensorsTile_,
    //                              blockLocalVars_,
    //                              bufPools_,
    //                              idx * cfg.basicBlock,
    //                              cfg.basicBlock,
    //                              idx & 1};
    //         Atvoss::Reduce::Tile::Evaluate<ExprTile>(context);
    //         idx = (idx + 1) & 1;
    //     }
    //     for (uint64_t i = bisectionTail_; i < bisectionPos_; i++) {
    //         ContextDataT context{blockTensorsTile_,
    //                              blockLocalVars_,
    //                              bufPools_,
    //                              idx * cfg.basicBlock,
    //                              cfg.basicBlock,
    //                              idx & 1};
    //         Atvoss::Reduce::Tile::Evaluate<ExprTile>(context);
    //     }
    //     startIdx = idx;
    // }

    template <typename ArgTup>
    __aicore__ inline void ProcessEmpty(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        return;
    }

    template <auto SchLoopInfo, typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {   
        if constexpr (Policy.EnableBinaryAcc) { // 二分Reduce



        //     // 初始化
        //     Initialize(cfg, argTuple);
        //     // 核内循环计算
        //     SetLoopRange(cfg);
        //     // 核内二次幂点计算
        //     SetIterRRange(cfg);
        //     if (1 == 1) { // Normal Reduce TODO: 判断条件修改
        //         for (uint64_t i = loopAStartIndex_; i < loopAEndIndex_; i++) {
        //             // 核内A/R轴迭代地址计算
        //             // 二分计算
        //             int32_t startIdx = 0;
        //             LinearCompute(i, startIdx);
        //         }
        //     } else {    // Group Reduce
        //         return;
        //     }
            return;
        } else {    // 顺序Reduce
            return;
        }
    }

    // template <typename ParamType>
    // __aicore__ inline auto ConstructParam()
    // {
    //     return ConstructParam<ParamType>(HasUsage<ParamType>{});
    // }

    // template <typename ParamType>
    // __aicore__ inline auto ConstructParam(std::false_type)
    // {
    //     using DType = typename ParamType::Type::PrimType;
    //     return BlockTensorTile<DType>{};
    // }

    // template <typename Params, size_t... Ints>
    // __aicore__ inline auto PrepareParamsImpl(AscendC::Std::index_sequence<Ints...>)
    // {
    //     return AscendC::Std::make_tuple(ConstructParam<Get_t<Params, Ints>>()...);
    // }

    // template <typename Params>
    // __aicore__ inline auto PrepareParams()
    // {
    //     return PrepareParamsImpl<Params>(AscendC::Std::make_index_sequence<Size_v<Params>>{});
    // }

    // template <typename ParamType, typename ArgTup>
    // __aicore__ inline constexpr auto ConstructBlockParam(ArgTup& args)
    // {
    //     constexpr auto argNumber = ParamType::inplaceNumber - 1;
    //     if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
    //         using DType = typename ParamType::Type::PrimType;
    //         auto& gm = AscendC::Std::get<argNumber>(args);
    //         return BlockTensorTile<DType>{gm};
    //     } else {
    //         return AscendC::Std::get<argNumber>(args);
    //     }
    // }

    // template <typename Params, typename ArgTup, std::size_t... Ints>
    // __aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    // {
    //     return AscendC::Std::make_tuple(ConstructBlockParam<Get_t<Params, Ints>>(args)...);
    // }

    // template <typename Params, typename ArgTup>
    // __aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple)
    // {
    //     return PrepareBlockParamsImpl<Params>(argTuple, AscendC::Std::make_index_sequence<Size_v<Params>>{});
    // }

private:
    AscendC::TPipe pipe_;
    BufPoolType bufPools_;
    uint32_t actualTileSize_;
    uint32_t preBufSize_ = 0;
    uint32_t postBufSize_ = 0;
    uint32_t resBufSize_ = 0;

    static constexpr int32_t PRE_MTE2_NUM_TOTAL = GetPreMte2Num();
    static constexpr int32_t PRE_MTE3_NUM_TOTAL = GetPreMte3Num();
    static constexpr int32_t PRE_TEMP_CALC_NUM = GetPreTempCalcNum();
    static constexpr int32_t POST_MTE2_NUM = GetPostMte2Num();
    static constexpr int32_t POST_MTE3_NUM = GetPostMte3Num();
    static constexpr int32_t POST_TEMP_CALC_NUM = GetPostTempCalcNum();

    // BlockLocalVarsType blockLocalVars_;
    // BlockTensorsTileType blockTensorsTile_;

    uint64_t loopAStartIndex_ = 0;
    uint64_t loopAEndIndex_ = 0;
    uint64_t loopAAxisStep_ = 0;
    uint64_t ubFactorA_ = 0;

    uint64_t loopRStartIndex_ = 0;
    uint64_t loopREndIndex_ = 0;
    uint64_t loopRAxisStep_ = 0;
    uint64_t ubFactorR_ = 0;

    // 二分算法的二分点
    uint64_t bisectionPos_ = 0;
    uint64_t cacheCount_ = 0;
    uint64_t bisectionTail_ = 0;
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_SCHEDULE_H_