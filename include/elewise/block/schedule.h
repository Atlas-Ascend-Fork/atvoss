/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_BLOCK_SCHEDULE_H_
#define ATVOSS_INCLUDE_BLOCK_SCHEDULE_H_
#include <functional>
#include <type_traits>
#include "block_info_tile.h"
#if !defined(__ATVOSS_HOST_ONLY__)
#include "utils/buf_pool/block_buf_pool.h"
#include "elewise/tile/tile_evaluate.h"
#endif

#include "utils/layout/layout.h"
#include "common/type_def.h"
#include "graph/buffer.h"
#include "graph/expr_linearizer.h"
#include "graph/compute_preproc.h"
#include "operators/tile_shape.h"

namespace Atvoss::Ele {

using Atvoss::Util::Find_v;
using Atvoss::Util::Get_t;
using Atvoss::Util::Size_v;

template <uint32_t UB_TILE_SIZE, uint32_t USER_TILE_SIZE>
struct TileCheckAssert {
    static_assert(USER_TILE_SIZE <= UB_TILE_SIZE, "user's tile size can not be bigger than Ub tile size.");
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg = void>
class BaseBlockSchedule {
public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using ArchTag = ArchTagCfg;
    using TileShape = typename BlockPolicy::TileShape;
    // number of elements in Tile calculation
    static constexpr uint32_t BASIC_BLOCK = Atvoss::Ele::Tile::GetTotalElement<0, BlockPolicy>(1, 1);

private:
    // alignment of constants
    static constexpr uint32_t ALIGNMENT = 32;

    /* ----------------Prepare param ---------------- */
    using ShapeT = Atvoss::Ele::Tile::Shape_t<typename std::remove_reference<decltype(Policy)>::type>;
    using ShapeSize = Atvoss::Ele::Tile::ShapeSize<ShapeT>;

    static constexpr uint32_t GetLayoutAxis0()
    {
        static_assert(ShapeSize::value <= 2, "[ERROR]: [Atvoss][Block] Tile shape can not be greater than 2!");
        if constexpr (ShapeSize::value == 2) {
            using DstShape0Type = typename ShapeT::template get_type<0>;
            static_assert((DstShape0Type::value > 0), "[ERROR]: [Atvoss][Block] Shape axis0 must not be zero");
            return DstShape0Type::value;
        }
        return BASIC_BLOCK;
    }
    static constexpr uint32_t GetLayoutAxis1()
    {
        static_assert(ShapeSize::value <= 2, "[ERROR]: [Atvoss][Block] Tile shape can not be greater than 2!");
        if constexpr (ShapeSize::value == 2) {
            using DstShape1Type = typename ShapeT::template get_type<1>;
            static_assert((DstShape1Type::value > 0), "[ERROR]: [Atvoss][Block] Shape axis1 must not be zero");
            return DstShape1Type::value;
        }
        return 1;
    }

    template <typename T>
    using BlockTensorTile = Atvoss::Ele::BlockTensor<
        T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<BASIC_BLOCK, GetLayoutAxis0(), GetLayoutAxis1()>>>;
    static constexpr auto computeRes = ToLinearizerExpr(Compute{}.template Compute<BlockTensorTile>());
    static constexpr auto optimizedCompute = Tile::PreProcessComputeExpr<Policy.memPolicy>(computeRes);
    using ComputeInfoT = decltype(optimizedCompute);
    using ExprTile = typename ComputeInfoT::Expr;
    using EleWiseDag = typename ComputeInfoT::Dag;
    // Remember-1: Size_v<Params> <= Size_v<InParams> + Size_v<OutParams>
    // Remember-2: Params comes from `Optimized` Dag rather than the original one.
    using Params = typename EleWiseDag::AllParams;
    using InParams = typename EleWiseDag::InParams;
    using OutParams = typename EleWiseDag::OutParams;
    using LocalVars = typename EleWiseDag::AllLocalVars;

    static constexpr uint32_t IN_PARAMS_COUNT = Size_v<InParams> * 2;
    static constexpr uint32_t OUT_PARAMS_COUNT = Size_v<OutParams> * 2;
    static constexpr uint32_t LOCAL_VAR_COUNT = Size_v<LocalVars>;

    static constexpr uint32_t MAX_BUFFER_COUNT = IN_PARAMS_COUNT + OUT_PARAMS_COUNT + LOCAL_VAR_COUNT;
    static constexpr uint32_t UB_TILE_SIZE = ArchTag::UB_SIZE / MAX_BUFFER_COUNT / 1024 * 1024;
    static constexpr uint64_t UB_ADDR_IN = 0;
    static constexpr uint64_t UB_ADDR_OUT = UB_TILE_SIZE * IN_PARAMS_COUNT;
    static constexpr uint64_t UB_ADDR_CALC = UB_ADDR_OUT + UB_TILE_SIZE * OUT_PARAMS_COUNT;

    /* ----------------Calc element count in a tensor start---------------- */
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

    /*!
     * \brief Get the size of UB space occupied by a single node during compilation.
     * \return uint32_t. Maximum number of elements allocated in a single node.
     */
    static constexpr uint32_t GetEleCntInTensor()
    {
        if constexpr (ShapeSize::value == 2) {
            using DstShape1Type = typename ShapeT::template get_type<1>;
            return ((UB_TILE_SIZE / MaxSize) + DstShape1Type::value - 1) / DstShape1Type::value;
        }
        return ((UB_TILE_SIZE / MaxSize) + ALIGNMENT - 1) / ALIGNMENT;
    }

    static constexpr uint32_t ELEMENT_COUNT_IN_TENSOR = GetEleCntInTensor();
    /* ----------------Calc element count in a tensor end---------------- */

public:
    /*!
     * \brief Configure Block Param during compilation.
     * \param[in] blockParam, Block ScheduleCfg(Config).
     * \return bool. Return true to indicate calculation success.
     */
    template <typename Args, typename KernelScheduleCfg>
    static bool MakeScheduleConfig(
        const Args& arguments, const KernelScheduleCfg& kernelConfig, ScheduleCfg& blockConfig)
    {
        printf("[DEBUG]: [Atvoss][BlockSchedule] MakeScheduleConfig for block!\n");
        printf("[DEBUG]: [Atvoss][BlockSchedule] MemPolicy is %d.\n", static_cast<int>(Policy.memPolicy));
        return true;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    /*!
     * \brief Default constructor
     */
    __aicore__ inline BaseBlockSchedule()
    {
        if constexpr (UB_TILE_SIZE < (BASIC_BLOCK * MaxSize + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT) {
            TileCheckAssert<UB_TILE_SIZE, (BASIC_BLOCK * MaxSize + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT> dummy;
        }

        bufPools_.Init();
    }

    /*!
     * \brief Block layer execution function.
     * \param[in] cfg, Segmentation of tile in block.
     * \param[in] argTuple, Input and output data in GM.
     */
    template <typename ArgTup>
    __aicore__ inline void Run(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        cfg.wholeLoop = cfg.totalElemCnt / BASIC_BLOCK;
        cfg.tileCnt = cfg.totalElemCnt % BASIC_BLOCK;
        Process(cfg, argTuple);
    }

private:
    template <typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto blockLocalVars = PrepareParams<LocalVars>();
        auto blockTensorsTile = PrepareBlockParams<Params>(argTuple);
        using BufferMaps = typename EleWiseDag::BufMap;
        using ContextDataT =
            ContextData<decltype(blockTensorsTile), decltype(blockLocalVars), decltype(bufPools_), BufferMaps>;

        uint32_t i = 0;
        for (; i < cfg.wholeLoop; i++) {
            ContextDataT context{blockTensorsTile, blockLocalVars, bufPools_, i * BASIC_BLOCK, BASIC_BLOCK, i & 1};
            Atvoss::Ele::Tile::Evaluate<ExprTile>(context);
        }
        if (cfg.tileCnt > 0) {
            ContextDataT context{blockTensorsTile, blockLocalVars, bufPools_, i * BASIC_BLOCK, cfg.tileCnt, i & 1};
            Atvoss::Ele::Tile::Evaluate<ExprTile>(context);
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
        // We use `inplaceNumber` to adapter IN_OUT params optimization in `AUTO` Dag
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
    Atvoss::BlockBufferEx<MAX_BUFFER_COUNT, UB_TILE_SIZE> bufPools_;
#endif
};

template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTag>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, ArchTag> {};

} // namespace Atvoss::Ele
#endif // ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
