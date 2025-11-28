/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_SCHEDULE_H
#define BASE_SCHEDULE_H
#include <functional>
#include "common/compile_info.h"
namespace Atvoss::Kernel {

/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process based on the block ID,
 * and pass it to the block to complete the computation.
*/
template <typename BlockOp, const auto &Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker;
    using ParamStruct = ScheduleCfg;
    using BlockTemplate = BlockOp;
    static constexpr auto EXPRESSION = ExprMaker{}.template Compute<AscendC::GlobalTensor>();
    using Expr = typename decltype(EXPRESSION)::Type;
    using Params = Atvoss::ExprTmpl::Params_t<Expr>;
     using TileShape = typename BlockTemplate::BlockTileShape;
    static constexpr uint32_t TILE_SHAPE_SIZE = TileShape::size::value;
    static constexpr uint32_t BASIC_BLOCK = BlockOp::ScheduleClz::BASIC_BLOCK;
    /*!
     * \brief The constructor interface of KernelBuilder class.
     */
    __aicore__ inline BaseKernelSchedule()
    {
        blockId_ = AscendC::GetBlockIdx();
    };
    /*!
     * \brief Kernel layer execution function.
     * \param[in] shapeInfo, Shape information of the user-input tensor.
     * \param[in] kernelParam, Tiling information in kernel.
     * \return bool. Return true to indicate param calculation success.
     */
    static bool MakeKernelParam(std::vector<uint32_t> &shapeInfo, ScheduleCfg &kernelParam)
    {
        uint32_t  totalEleNum = 1;
        for(int i = 0; i < shapeInfo.size(); i++) {
            totalEleNum = totalEleNum * shapeInfo[i];
        }
        if(shapeInfo.size() == 0 || totalEleNum == 0){
            printf("[ERROR]: [Atvoss][Kernel] Shape info error \n");
            return false;
        }
        if (TILE_SHAPE_SIZE != 1 && TILE_SHAPE_SIZE != 2) {
            printf("[ERROR]: [Atvoss][Kernel] Tile shape size only support 1 or 2\n");
            return false;
        }
        uint32_t actualNAssign = TILE_SHAPE_SIZE == 1 ? 32 : TileShape::template get_type<TILE_SHAPE_SIZE - 1>::value;
        
        // Initial core tiling baseline
        uint32_t basicCoreEleNum = (BASIC_BLOCK + actualNAssign - 1) / actualNAssign * actualNAssign;
        if (basicCoreEleNum < BASIC_BLOCK) {
            printf("[ERROR]: [Atvoss][Kernel] basicCoreEleNum is too small \n");
            return false;
        }
        kernelParam.unitNum = actualNAssign;
        if (totalEleNum <= basicCoreEleNum) {
            kernelParam.blockNum = 1;
            kernelParam.tailNum = totalEleNum;
            kernelParam.unitNumPerCore = 0;
            kernelParam.moreUnitCoreNum = 0;
        } else {
            uint32_t basicCoreUnitNum = basicCoreEleNum / actualNAssign; // Number of aligned units processable by a single core and basicCoreEleNum >= actualNAssign
            uint32_t totalUnitCnt = totalEleNum / actualNAssign; // Total number of full units
            kernelParam.blockNum = (totalUnitCnt + basicCoreUnitNum -1 ) / basicCoreUnitNum;
            if (kernelParam.blockNum > Policy.blockDimMax) {
                kernelParam.blockNum = Policy.blockDimMax;
            }
            kernelParam.unitNumPerCore = totalUnitCnt / kernelParam.blockNum;
            kernelParam.moreUnitCoreNum = totalUnitCnt % kernelParam.blockNum; // Number of remaining full units
            kernelParam.tailNum = totalEleNum % actualNAssign; // Remaining total elements
        }
        return true;
    }

    /*!
     * \brief Kernel layer execution function.
     * \param[in] cfg, Tiling information in kernel.
     * \param[in] argTuple, Input and output GM address.
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        typename BlockOp::ScheduleClz::ParamStruct configBlock = cfg.blockParam;
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        BlockOp blockOp{};
        uint32_t actualNum = CalCurCoreEleCnt(cfg.kernelParam);
        totalEleNumCurCore_ = actualNum;
        configBlock.totalElemCnt = actualNum;
        auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
        auto convertArgs = ConvertArgs<Params>(params, argTuple);
        blockOp.Run(configBlock, convertArgs);
    }

private:
    template <typename Params, typename ArgTup>
    __aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto params = PrepareParamsImpl<Params>(cfg, argTuple, AscendC::Std::make_index_sequence<Util::TMP::Size_v<Params>>{});
        return params;
    }

    __aicore__ inline auto CalCurCoreEleCnt(ScheduleCfg& cfg)
    {
        uint32_t blockNum = cfg.blockNum;
        uint32_t actualNum = cfg.unitNum * cfg.unitNumPerCore;
        if(blockId_ < cfg.moreUnitCoreNum){
            actualNum += cfg.unitNum;
        }
        if(blockId_ == blockNum - 1){
            actualNum += cfg.tailNum;
        }
        return actualNum;
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(ScheduleCfg& config , ArgTup &args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(
            ConstructParam<Util::TMP::Get_t<Params, Ints>>(config, args)...);
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline auto ConstructParam(ScheduleCfg& config, ArgTup& args)
    {
        using DTypeTmp = typename ParamType::Type::PrimType;
        uint64_t offset = CalGMOffset(config);
        auto ptr = AscendC::Std::get<ParamType::number - 1>(args);
        auto gm_ptr = reinterpret_cast<__gm__ uint8_t*>(ptr);
        gm_ptr = gm_ptr + sizeof(DTypeTmp) * offset;
        AscendC::GlobalTensor<DTypeTmp> curCoreTensor;
        curCoreTensor.SetGlobalBuffer(reinterpret_cast<__gm__ DTypeTmp*>(gm_ptr),totalEleNumCurCore_);
        return curCoreTensor;
    }

    template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
    __aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos =
            Util::TMP::Find_v<Atvoss::ExprTmpl::CheckVarNum<Index + 1>::template Checker, Params>;
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
        return ConvertArgsImpl<Params>(
            params, args, AscendC::Std::make_index_sequence<AscendC::Std::tuple_size_v<ArgTup>>{});
    }

    __aicore__ inline auto CalGMOffset(ScheduleCfg& config)
    {
        if(blockId_ < config.moreUnitCoreNum){
            return blockId_ * (config.unitNumPerCore * config.unitNum + config.unitNum);
        }
        return config.unitNumPerCore * config.unitNum * blockId_ + config.moreUnitCoreNum * config.unitNum;
    }
    uint32_t blockId_;
    uint32_t totalEleNumCurCore_;
};

} // namespace Atvoss::Kernel
#endif