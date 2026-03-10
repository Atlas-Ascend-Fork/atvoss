/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
#define ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
#include <functional>
#include "common/arch.h"

#if !defined(__ATVOSS_HOST_ONLY__)
template <typename T>
using KernelTensor = AscendC::GlobalTensor<T>;
#else
template <typename T>
struct KernelTensor {
    using Type = T;
};
#endif

namespace Atvoss::Ele {

/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process
 * based on the block ID, and pass it to the block to complete the computation.
 */
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class BaseKernelSchedule {
public:
    using ExprMaker = typename BlockOp::ScheduleClz::ExpressMaker;
    using ParamStruct = ScheduleCfg;
    using BlockTemplate = BlockOp;
    static constexpr auto EXPRESSION = ToLinearizerExpr(ExprMaker{}.template Compute<KernelTensor>());
    using Expr = typename decltype(EXPRESSION)::Type;
    using Params = Atvoss::Params_t<Expr>;
    using TileShape = typename BlockTemplate::BlockTileShape;
    using ArchTag = typename BlockOp::ScheduleClz::ArchTag;
    static constexpr uint64_t TILE_SHAPE_SIZE = TileShape::size::value;
    static constexpr uint64_t BASIC_BLOCK = BlockOp::ScheduleClz::BASIC_BLOCK;
    static constexpr uint64_t ALIGN_TILE_SHAPE_SIZE = 2;
    static constexpr uint64_t ACTUAL_N_ASSIGN =
        TILE_SHAPE_SIZE == 1 ? 32 : TileShape::template get_type<TILE_SHAPE_SIZE - 1>::value;
    static constexpr uint64_t BASIC_CORE_ELE_NUM =
        (BASIC_BLOCK + ACTUAL_N_ASSIGN - 1) / ACTUAL_N_ASSIGN * ACTUAL_N_ASSIGN;

    /*!
     * \brief Kernel layer execution function.
     * \param[in] arguments, information of the user inputs.
     * \param[in] kernelParam, Tiling information in kernel.
     * \return bool. Return true to indicate param calculation success.
     */
    template <typename Args>
    static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
    {
        // Extract input shape infomation from arguments
        using InputTuple = std::decay_t<decltype(std::get<0>(arguments))>;
        static_assert(std::tuple_size_v<InputTuple> > 0, "[ERROR]: [Atvoss][Kernel] Get input shape error \n");
        std::vector<uint64_t> shapeInfo = std::get<0>(std::get<0>(arguments)).shape_vector();
        uint64_t totalEleNum = CalculateTotalElements(shapeInfo);

        if (shapeInfo.size() == 0 || totalEleNum == 0) {
            printf("[ERROR]: [Atvoss][Kernel] Shape info error \n");
            return false;
        }

        kernelParam.unitNum = ACTUAL_N_ASSIGN;
        if (totalEleNum <= BASIC_CORE_ELE_NUM) {
            kernelParam.blockNum = 1;
            kernelParam.tailNum = totalEleNum;
            kernelParam.unitNumPerCore = 0;
            kernelParam.moreUnitCoreNum = 0;
            return true;
        }

        uint64_t basicCoreUnitNum =
            BASIC_CORE_ELE_NUM / ACTUAL_N_ASSIGN; // Number of aligned units processable by a single core and
                                                  // basicCoreEleNum >= actualNAssign
        uint64_t totalUnitCnt = totalEleNum / ACTUAL_N_ASSIGN; // Total number of full units
        kernelParam.blockNum = (totalUnitCnt + basicCoreUnitNum - 1) / basicCoreUnitNum;
        if (kernelParam.blockNum > ArchTag::CORE_NUM) {
            kernelParam.blockNum = ArchTag::CORE_NUM;
        }
        kernelParam.unitNumPerCore = totalUnitCnt / kernelParam.blockNum;
        kernelParam.moreUnitCoreNum = totalUnitCnt % kernelParam.blockNum; // Number of remaining full units
        kernelParam.tailNum = totalEleNum % ACTUAL_N_ASSIGN;               // Remaining total elements

        return true;
    }

private:
    static inline uint64_t CalculateTotalElements(const std::vector<uint64_t>& shapeInfo)
    {
        uint64_t totalEleNum = 1;
        for (auto dim : shapeInfo) {
            totalEleNum *= dim;
        }
        return totalEleNum;
    }

#if !defined(__ATVOSS_HOST_ONLY__)
public:
    /*!
     * \brief The constructor interface of KernelBuilder class.
     */
    __aicore__ inline BaseKernelSchedule() = default;

    /*!
     * \brief Kernel layer execution function.
     * \param[in] cfg, Tiling information in kernel.
     * \param[in] argTuple, Input and output GM address.
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        // Configure block scheduling parameters
        typename BlockOp::ScheduleClz::ParamStruct configBlock = cfg.blockParam;
        configBlock.totalElemCnt = CalCurCoreEleCnt(cfg.kernelParam);
        // Pack and prepare arguments for kernel execution
        auto argTuple = AscendC::Std::forward_as_tuple(AscendC::Std::forward<Args>(args)...);
        auto params = PrepareParams<Params>(cfg.kernelParam, argTuple);
        auto convertArgs = ConvertArgs<Params>(params, argTuple);

        BlockOp blockOp{};
        blockOp.Run(configBlock, convertArgs);
    }

    template <typename Params, typename ArgTup>
    __aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
    {
        auto params =
            PrepareParamsImpl<Params>(cfg, argTuple, AscendC::Std::make_index_sequence<Atvoss::Util::Size_v<Params>>{});
        return params;
    }

    __aicore__ inline auto CalCurCoreEleCnt(ScheduleCfg& cfg)
    {
        uint64_t blockNum = cfg.blockNum;
        uint64_t actualNum = cfg.unitNum * cfg.unitNumPerCore;
        if (AscendC::GetBlockIdx() < cfg.moreUnitCoreNum) {
            actualNum += cfg.unitNum;
        }
        if (AscendC::GetBlockIdx() == blockNum - 1) {
            actualNum += cfg.tailNum;
        }
        return actualNum;
    }

    template <typename Params, typename ArgTup, std::size_t... Ints>
    __aicore__ inline auto PrepareParamsImpl(ScheduleCfg& config, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
    {
        return AscendC::Std::make_tuple(ConstructParam<Atvoss::Util::Get_t<Params, Ints>>(config, args)...);
    }

    template <typename ParamType, typename ArgTup>
    __aicore__ inline auto ConstructParam(ScheduleCfg& config, ArgTup& args)
    {
        auto arg = AscendC::Std::get<ParamType::number - 1>(args);
        if constexpr (!std::is_scalar_v<typename ParamType::Type>) {
            using DTypeTmp = typename ParamType::Type::PrimType;
            uint64_t offset = CalGMOffset(config);
            auto ptr = (uint64_t)(arg) + sizeof(DTypeTmp) * offset;
            return reinterpret_cast<__gm__ uint8_t*>(ptr);
        } else {
            using argType = decltype(arg);
            if constexpr (std::is_pointer_v<argType>) {
                return *arg;
            } else {
                return arg;
            }
        }
    }

    template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
    __aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
    {
        constexpr auto pos = Atvoss::Util::Find_v<Atvoss::CheckVarNum<Index + 1>::template Checker, Params>;
        if constexpr (pos < Atvoss::Util::Size_v<Params>) {
            return AscendC::Std::get<pos>(params);
        } else {
            return AscendC::Std::get<Index>(args);
        }
    }

    template <typename Params, typename ParamTup, typename ArgTup, size_t... Ints>
    __aicore__ inline constexpr auto ConvertArgsImpl(
        ParamTup& params, ArgTup& args, AscendC::Std::index_sequence<Ints...>)
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
        if (AscendC::GetBlockIdx() < config.moreUnitCoreNum) {
            return AscendC::GetBlockIdx() * (config.unitNumPerCore * config.unitNum + config.unitNum);
        }
        return config.unitNumPerCore * config.unitNum * AscendC::GetBlockIdx() +
               config.moreUnitCoreNum * config.unitNum;
    }
#endif
};

template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class DefaultKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {};

} // namespace Atvoss::Ele
#endif // ATVOSS_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
