/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_ELEWISE_H
#define KERNEL_ELEWISE_H
#include "schedule.h"
namespace Atvoss::Ele {

template <int32_t Dim>
struct DefaultKernelConfig {      // Kernel layer tiling information
    uint32_t patternID;
    uint32_t coreNum;  // Number of cores
    uint64_t ubFactorA = 0;
    uint64_t ubFactorR = 0;
    uint64_t totalTileACnt = 0; // Number of TileA count
    uint64_t totalTileRCnt = 0; // Number of TileR count
    uint64_t tileACntPerCore = 0; // Number of TileA count processed per core
    uint64_t tileRCntPerCore = 0; // Number of TileR count processed per core
    uint64_t groupR = 1;  // Number of groups (for multi-axis reduce)
    uint64_t shape[Dim];     // shape of tensor
    uint64_t srcStride[Dim];  // input stride of tensor
    uint64_t outStride[Dim];  // output stride of tensor
};

enum class DefaultSegmentPolicy
{
    UniformSegment, // Uniform tiling
};

struct DefaultKernelPolicy {
    DefaultSegmentPolicy segmentPolicy; // Multi-core tiling strategy
};

static constexpr DefaultKernelPolicy defaultKernelPolicy{DefaultSegmentPolicy::UniformSegment};

/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process
 * based on the block ID, and pass it to the block to complete the computation.
 */
template <
    typename BlockOp, const auto& Policy = defaultKernelPolicy, typename ScheduleCfg = DefaultKernelConfig,
    template <typename, const auto&, typename> class Schedule = DefaultKernelSchedule>
class KernelBuilder {
public:
    struct OpParam {
        ScheduleCfg kernelParam;
        typename BlockOp::ScheduleCfgClz blockParam;
    };
    using ScheduleClz = Schedule<BlockOp, Policy, ScheduleCfg>;
    using ScheduleCfgClz = OpParam;

#if !defined(__ATVOSS_HOST_ONLY__)

    /*!
     * \brief Kernel layer execution function.
     * \param[in] cfg, Tiling information in kernel.
     * \param[in] argTuple, Input and output GM address.
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
        static_assert(
            (... && (std::is_scalar_v<Args> || Util::IsTensor_v<Args>)),
            "KernelBuilder::Run only accepts scalar or Tensor types");
        ScheduleClz schedule;
        schedule.Run(cfg, args...);
    }
#endif
};

} // namespace Atvoss::Ele
#endif