/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ELE_BASE_SCHEDULE_H
#define BLOCK_ELE_BASE_SCHEDULE_H

#include <functional>
#include <type_traits>
#include "block/block_tensor.h"
#include "block/block_tail_tensor.h"
#include "common/tuple_tool.h"
#include "utils/buf_pool/loopbuf.h"
#include "utils/layout/layout.h"
#include "utils/layout/shape.h"

namespace Atvoss::Block {

template <typename Compute, const auto& Policy, typename ScheduleCfg>
class BaseBlockSchedule {
public:
    using ExpressMaker = Compute;
    using ParamStruct = ScheduleCfg;
    using BlockPolicy = typename std::remove_reference<decltype(Policy)>::type;
    using TileShape = typename BlockPolicy::TileShape;
    template <typename T>
    using blockTensorFake = Atvoss::Block::Tensor<T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<1, 1, 1>>>;
    static constexpr auto EXPRESSION_FAKE = Compute{}.template Compute<blockTensorFake>();
    using ExprFake = typename decltype(EXPRESSION_FAKE)::Type;
    using Params = Atvoss::ExprTmpl::Params_t<ExprFake>;
    using InParams = Atvoss::ExprTmpl::InParams_t<ExprFake>;
    using OutParams = Atvoss::ExprTmpl::OutParams_t<ExprFake>;
    using LocalVars = Atvoss::ExprTmpl::LocalVars_t<ExprFake>;
    static constexpr uint32_t IN_PARAMS_COUNT = Util::TMP::Size_v<InParams> + 1;
    static constexpr uint32_t OUT_PARAMS_COUNT = Util::TMP::Size_v<OutParams> + 1;
    static constexpr uint32_t LOCAL_VAR_COUNT = Util::TMP::Size_v<LocalVars>;

    static constexpr uint32_t MAX_BUFFER_COUNT = IN_PARAMS_COUNT + OUT_PARAMS_COUNT + LOCAL_VAR_COUNT;
    static constexpr uint32_t UB_TILE_SIZE = Policy.ubSizeMax / MAX_BUFFER_COUNT / 1024 * 1024;
    static constexpr uint64_t UB_ADDR_IN = 0;
    static constexpr uint64_t UB_ADDR_OUT = UB_TILE_SIZE * IN_PARAMS_COUNT;
    static constexpr uint64_t UB_ADDR_CALC = UB_ADDR_OUT + UB_TILE_SIZE * OUT_PARAMS_COUNT;

    // alignment of constants
    static constexpr uint32_t ALIGNMENT = 32;

    template <size_t N, typename Params>
    static constexpr int FindParamsMaxTypeSizeImpl(int defaultSize = 1)
    {
        using ParamType = Util::TMP::Get_t<Params, N>;
        using DType = typename ParamType::Type::PrimType;
        if (defaultSize < sizeof(DType)) {
            defaultSize = sizeof(DType);
        }
        constexpr int len = Util::TMP::Size_v<Params>;
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

    // number of elements in Tile calculation
    static constexpr uint32_t BASIC_BLOCK =
        Atvoss::Tile::GetTotal<0, typename std::remove_reference<decltype(Policy)>::type>(1, 1);

    /*!
     * \brief Default constructor
     */
    __aicore__ inline BaseBlockSchedule()
    {
        bufPoolIn_.template Init<UB_ADDR_IN>();
        bufPoolOut_.template Init<UB_ADDR_OUT>();
        bufPoolCalc_.template Init<UB_ADDR_CALC>();
    }

    /*!
     * \brief Configure Block Param during compilation.
     * \param[in] blockParam, Block ScheduleCfg(Config).
     * \return bool. Return true to indicate calculation success.
     */
    static bool MakeBlockParam(ScheduleCfg& blockParam)
    {
        if (ELEMENT_COUNT_IN_TENSOR == 0) {
            printf("[ERROR]: [Atvoss][Block] Element count can not be zero\n");
            return false;
        }
        blockParam.wholeLoop = ELEMENT_COUNT_IN_TENSOR / BASIC_BLOCK;
        blockParam.tileCnt = ELEMENT_COUNT_IN_TENSOR % BASIC_BLOCK;
        blockParam.basicNum = ELEMENT_COUNT_IN_TENSOR;
        blockParam.ubAssign = {IN_PARAMS_COUNT, OUT_PARAMS_COUNT, LOCAL_VAR_COUNT, ELEMENT_COUNT_IN_TENSOR};
        return true;
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
        bufPoolIn_.DeInit();
        bufPoolOut_.DeInit();
        if constexpr (LOCAL_VAR_COUNT > 0) {
            bufPoolCalc_.DeInit();
        }
    }

    using ShapeT = Atvoss::Tile::Eval::Shape_t<typename std::remove_reference<decltype(Policy)>::type>;
    using ShapeSize = Atvoss::Tile::Eval::ShapeSize<ShapeT>;

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

    template <typename T> // Tile Tensor
    using BlockTensorTile = Atvoss::Block::Tensor<
        T, Atvoss::Layout::Layout<Atvoss::Layout::FixedRankExtents<BASIC_BLOCK, GetLayoutAxis0(), GetLayoutAxis1()>>>;
    static constexpr auto EXPRESSION_TILE = Compute{}.template Compute<BlockTensorTile>();
    using ExprTile = typename decltype(EXPRESSION_TILE)::Type;
    using ParamsTile = Atvoss::ExprTmpl::Params_t<ExprTile>;
    using LocalVarsTile = Atvoss::ExprTmpl::LocalVars_t<ExprTile>;

    template <typename T> // Tail Tensor
    using BlockTensorTail =
        Atvoss::Block::TailTensor<T, Atvoss::Layout::TailLayout<Atvoss::Layout::VariableRankExtents<1>>>;
    static constexpr auto EXPRESSION_TAIL = Compute{}.template Compute<BlockTensorTail>();
    using ExprTail = typename decltype(EXPRESSION_TAIL)::Type;
    using ParamsTail = Atvoss::ExprTmpl::Params_t<ExprTail>;
    using LocalVarsTail = Atvoss::ExprTmpl::LocalVars_t<ExprTail>;

private:
    template <typename ArgTup>
    __aicore__ inline void Process(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto blockLocalVars = PrepareParams<LocalVarsTile>();
        auto blockTensorsParamsTile = PrepareBlockParams<ParamsTile>(argTuple);
        auto blockTensorsTile = ConvertArgs<ParamsTile>(blockTensorsParamsTile, argTuple);

        int i = 0;
        for (; i < cfg.wholeLoop; i++) {
            blockTensorsTile = CopyBlockIn<ParamsTile, true>(blockTensorsTile, i * BASIC_BLOCK, BASIC_BLOCK);
            blockTensorsTile = CopyBlockIn<ParamsTile, false>(blockTensorsTile, i * BASIC_BLOCK, BASIC_BLOCK);
            bufPoolIn_.Sync();
            Tile::Evaluate<ExprTile>(blockTensorsTile, blockLocalVars);
            bufPoolOut_.Sync();
            FreeBlockTensors<ParamsTile, true>(blockTensorsTile);
            CopyBlockOut<ParamsTile>(blockTensorsTile, i * BASIC_BLOCK, BASIC_BLOCK);
            FreeBlockTensors<ParamsTile, false>(blockTensorsTile);
        }
        if constexpr (LOCAL_VAR_COUNT > 0) {
            FreeCalcTensors(blockLocalVars,
                            AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<decltype(blockLocalVars)>>{});
        }
        if (cfg.tileCnt > 0) {
            int minTypeSize = FindParamsMinTypeSize<ParamsTail>();
            cfg.tileCnt = (cfg.tileCnt * minTypeSize + 31) / 32 * 32 / minTypeSize;

            auto blockLocalVarsTail = PrepareParams<LocalVarsTail>();
            auto blockTensorsParamsTail = PrepareBlockParams<ParamsTail>(argTuple, cfg.tileCnt);
            auto blockTensorsTail = ConvertArgs<ParamsTail>(blockTensorsParamsTail, argTuple);

            blockTensorsTail = CopyBlockIn<ParamsTail, true>(blockTensorsTail, i * BASIC_BLOCK, cfg.tileCnt);
            blockTensorsTail = CopyBlockIn<ParamsTail, false>(blockTensorsTail, i * BASIC_BLOCK, cfg.tileCnt);
            bufPoolIn_.Sync();
            if constexpr (ShapeSize::value == 2) {
                using DstShape1Type = typename ShapeT::template get_type<1>;
                Tile::Evaluate<ExprTile>(blockTensorsTail, blockLocalVarsTail, cfg.tileCnt, DstShape1Type::value);
            } else {
                Tile::Evaluate<ExprTile>(blockTensorsTail, blockLocalVarsTail, cfg.tileCnt);
            }
            bufPoolOut_.Sync();
            FreeBlockTensors<ParamsTail, true>(blockTensorsTail);
            CopyBlockOut<ParamsTail>(blockTensorsTail, i * BASIC_BLOCK, cfg.tileCnt);
            FreeBlockTensors<ParamsTail, false>(blockTensorsTail);
            if constexpr (LOCAL_VAR_COUNT > 0) {
                FreeCalcTensors(
                    blockLocalVarsTail,
                    AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<decltype(blockLocalVarsTail)>>{});
            }
        }
    }

    template <typename T, typename = void>
    struct HasUsage : std::false_type {};

    template <typename T>
    struct HasUsage<T, std::void_t<decltype(T::usage)>> : std::true_type {};

    template <typename ParamType>
    __aicore__ inline auto ConstructParam()
    {
        return ConstructParam<ParamType>(HasUsage<ParamType>{});
    }

    template <typename ParamType>
    __aicore__ inline auto ConstructParam(std::false_type)
    {
        using DType = typename ParamType::Type::PrimType;
        AscendC::LocalTensor<DType> tensor;
        bufPoolCalc_.AllocTensor(tensor);
        if constexpr (std::is_same_v<ParamType, Atvoss::Layout::VariableRankExtents<1>>) {
            Tensor<DType> blockTensorTail{ParamType::number - 1};
            blockTensorTail.SetUbTensor(tensor);
            return blockTensorTail;
        } else {
            Tensor<DType> blockTensorTile{ParamType::number - 1};
            blockTensorTile.SetUbTensor(tensor);
            return blockTensorTile;
        }
    }

    template <typename Params, size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Util::TMP::Get_t<Params, Ints>>()...);
    }

    template <typename Params>
    __aicore__ inline auto PrepareParams()
    {
        return PrepareParamsImpl<Params>(AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
    __aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos = Util::TMP::Find_v<Atvoss::ExprTmpl::CheckVarNum<Index + 1>::template Checker, Params>;
        if constexpr (pos < Util::TMP::Size_v<Params>) {
            return AscendC::Std::get<pos>(params);
        } else {
            return AscendC::Std::get<Index>(args);
        }
    }

    template <typename Params, typename ParamTup, typename ArgTup, size_t... Ints>
    __aicore__ inline constexpr auto ConvertArgsImpl(ParamTup& params, ArgTup& args,
                                                     AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConvertOneArg<Params, Ints>(params, args)...);
    }

    template <typename Params, typename ParamTup, typename ArgTup>
    __aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args)
    {
        return ConvertArgsImpl<Params>(params, args,
                                       AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTup>>{});
    }

    __aicore__ uint32_t GetDynamicTailLayoutAxis0(uint32_t tailCnt)
    {
        static_assert(ShapeSize::value <= 2, "[ERROR]: [Atvoss][Block] Tile shape can not be greater than 2!");
        if constexpr (ShapeSize::value == 2) {
            using DstShape1Type = typename ShapeT::template get_type<1>;
            static_assert((DstShape1Type::value > 0), "[ERROR]: [Atvoss][Block] Shape axis1 must not be zero");
            return tailCnt / DstShape1Type::value;
        }
        return tailCnt;
    }

    __aicore__ constexpr uint32_t GetDynamicTailLayoutAxis1()
    {
        if constexpr (ShapeSize::value == 2) {
            using DstShape1Type = typename ShapeT::template get_type<1>;
            static_assert((DstShape1Type::value > 0), "[ERROR]: [Atvoss][Block] Shape axis1 must not be zero");
            return DstShape1Type::value;
        }
        return 1;
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline constexpr auto ConstructBlockParam(ArgTup& args, uint32_t tileCnt)
    {
        using DType = typename ParamType::Type::PrimType;
        auto gm = AscendC::Std::get<ParamType::number - 1>(args);
        if constexpr (std::is_same_v<ParamType, Atvoss::Layout::VariableRankExtents<1>>) {
            return BlockTensorTail<DType>{gm,
                                          Atvoss::Layout::TailLayout{tileCnt, GetDynamicTailLayoutAxis0(tileCnt),
                                                                     GetDynamicTailLayoutAxis1(tileCnt)},
                                          ParamType::usage, ParamType::number - 1};
        } else {
            return BlockTensorTile<DType>{gm, ParamType::usage, ParamType::number - 1};
        }
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto PrepareBlockParamsImpl(ArgTup& args, uint32_t tileCnt,
                                                            AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructBlockParam<Util::TMP::Get_t<Params, Ints>>(args, tileCnt)...);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline constexpr auto PrepareBlockParams(ArgTup& argTuple, uint32_t tileCnt = 0)
    {
        return PrepareBlockParamsImpl<Params>(argTuple, tileCnt,
                                              AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename ParamType, bool isCopyInput, typename ArgTup>
    __aicore__ inline constexpr auto ConstructCopyBlockIn(ArgTup& args, int pos, int copyLen)
    {
        using DType = typename ParamType::Type::PrimType;
        auto blockTensor = AscendC::Std::get<ParamType::number - 1>(args);
        if (isCopyInput && (blockTensor.GetParamUsage() == Atvoss::ParamUsage::in ||
                            blockTensor.GetParamUsage() == Atvoss::ParamUsage::in_out)) {
            AscendC::LocalTensor<DType> tensor;
            bufPoolIn_.AllocTensor(tensor);
            blockTensor.SetUbTensor(tensor);
            blockTensor.CopyIn(pos, copyLen);

        } else if (!isCopyInput && (blockTensor.GetParamUsage() == Atvoss::ParamUsage::out)) {
            AscendC::LocalTensor<DType> tensor;
            bufPoolOut_.AllocTensor(tensor);
            blockTensor.SetUbTensor(tensor);
        }

        return blockTensor;
    }

    template <typename Params, bool isCopyInput, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto CopyBlockInImpl(ArgTup& args, int pos, int copyLen,
                                                     AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(
            ConstructCopyBlockIn<
                Util::TMP::FindUnique_t<Atvoss::ExprTmpl::CheckVarNum<Ints + 1>::template Checker, Params>,
                isCopyInput>(args, pos, copyLen)...);
    }

    template <typename Params, bool isCopyInput, typename ArgTup>
    __aicore__ inline constexpr auto CopyBlockIn(ArgTup& argTuple, int pos, int copyLen)
    {
        return CopyBlockInImpl<Params, isCopyInput>(argTuple, pos, copyLen,
                                                    AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline constexpr auto ConstructCopyBlockOut(ArgTup& args, int pos, int copyLen)
    {
        auto blockTensor = AscendC::Std::get<ParamType::number - 1>(args);
        if ((blockTensor.GetParamUsage() == Atvoss::ParamUsage::out ||
             blockTensor.GetParamUsage() == Atvoss::ParamUsage::in_out)) {
            blockTensor.CopyOut(pos, copyLen);
        }
        return blockTensor;
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto CopyBlockOutImpl(ArgTup& args, int pos, int copyLen,
                                                      AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(
            ConstructCopyBlockOut<
                Util::TMP::FindUnique_t<Atvoss::ExprTmpl::CheckVarNum<Ints + 1>::template Checker, Params>>(
                args, pos, copyLen)...);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline constexpr auto CopyBlockOut(ArgTup& argTuple, int pos, int copyLen)
    {
        return CopyBlockOutImpl<Params>(argTuple, pos, copyLen,
                                        AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline constexpr auto ConstructMakeBlockTensor2LocalTensors(ArgTup& args)
    {
        auto blockTensor = AscendC::Std::get<ParamType::number - 1>(args);
        return blockTensor.GetUbTensor();
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto MakeBlockTensor2LocalTensorsImpl(ArgTup& args,
                                                                      AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(
            ConstructMakeBlockTensor2LocalTensors<
                Util::TMP::FindUnique_t<Atvoss::ExprTmpl::CheckVarNum<Ints + 1>::template Checker, Params>>(args)...);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline constexpr auto MakeBlockTensor2LocalTensors(ArgTup& argTuple)
    {
        return MakeBlockTensor2LocalTensorsImpl<Params>(argTuple,
                                                        AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename ParamType, bool isInput, typename ArgTup>
    __aicore__ inline constexpr auto ConstructFreeBlockTensors(ArgTup& args)
    {
        using DType = typename ParamType::Type::PrimType;

        auto blockTensor = AscendC::Std::get<ParamType::number - 1>(args);
        auto localTensor = blockTensor.GetUbTensor();

        if (isInput && (blockTensor.GetParamUsage() == Atvoss::ParamUsage::in)) {
            bufPoolIn_.FreeTensor(localTensor);
        } else if (!isInput && (blockTensor.GetParamUsage() == Atvoss::ParamUsage::out ||
                                blockTensor.GetParamUsage() == Atvoss::ParamUsage::in_out)) {
            bufPoolOut_.FreeTensor(localTensor);
        }
        return localTensor;
    }

    template <typename Params, bool isInput, typename ArgTup, std::size_t... Ints>
    __aicore__ inline constexpr auto FreeBlockTensorsImpl(ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructFreeBlockTensors<Util::TMP::Get_t<Params, Ints>, isInput>(args)...);
    }

    template <typename Params, bool isInput, typename ArgTup>
    __aicore__ inline constexpr auto FreeBlockTensors(ArgTup& argTuple)
    {
        return FreeBlockTensorsImpl<Params, isInput>(argTuple,
                                                     AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
    }

    template <typename ArgTup, size_t... Ints>
    __aicore__ inline void FreeCalcTensors(ArgTup& tensors, AscendC::Std::index_sequence<Ints...>)
    {
        bufPoolCalc_.FreeTensor(AscendC::Std::get<Ints>(tensors).GetUbTensor()...);
    }

    template <size_t N, typename Params>
    static constexpr __aicore__ inline int FindParamsMinTypeSizeImpl(int defaultSize = 32)
    {
        using ParamType = Util::TMP::Get_t<Params, N>;
        using DType = typename ParamType::Type::PrimType;
        if (defaultSize > sizeof(DType)) {
            defaultSize = sizeof(DType);
        }
        constexpr int len = Util::TMP::Size_v<Params>;
        if constexpr (N < len - 1) {
            return FindParamsMinTypeSizeImpl<N + 1, Params>(defaultSize);
        }
        return defaultSize;
    }

    template <typename Params>
    static constexpr __aicore__ inline int FindParamsMinTypeSize()
    {
        return FindParamsMinTypeSizeImpl<0, Params>();
    }

private:
    Atvoss::LoopBuffer<AscendC::TPosition::VECIN, IN_PARAMS_COUNT, UB_TILE_SIZE> bufPoolIn_;
    Atvoss::LoopBuffer<AscendC::TPosition::VECOUT, OUT_PARAMS_COUNT, UB_TILE_SIZE> bufPoolOut_;
    Atvoss::LoopBuffer<AscendC::TPosition::VECCALC, LOCAL_VAR_COUNT, UB_TILE_SIZE> bufPoolCalc_;
};

} // namespace Atvoss::Block

#endif