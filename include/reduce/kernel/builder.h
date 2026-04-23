/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_REDUCE_H
#define KERNEL_REDUCE_H
#include "schedule.h"
#include "device/tilingdata.h"
namespace Atvoss::Reduce {

enum class DefaultSegmentPolicy
{
    MemoryFirst = 0U,   // For bandwidth bound operators, prioritize fully utilizing the UB
    ComputeFirst,       // For compute bound operators, prioritize fully utilizing all cores
};

struct DefaultKernelPolicy {
    DefaultSegmentPolicy segmentPolicy; // Multi-core tiling strategy
};

// TODO: 默认的kernelPolicy
static constexpr DefaultKernelPolicy defaultKernelPolicy{DefaultSegmentPolicy::MemoryFirst};

/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process
 * based on the block ID, and pass it to the block to complete the computation.
 */
template <
    typename BlockOp, const auto& Policy = defaultKernelPolicy, typename ScheduleCfg = ReduceTilingData,
    template <typename, const auto&, typename> class Schedule = DefaultKernelSchedule>
class KernelBuilder {
public:
    struct OpParam {
        ScheduleCfg TilingData;
    };
    using ScheduleClz = Schedule<BlockOp, Policy, ScheduleCfg>;
    using ScheduleCfgClz = OpParam;
    using TilingData = ScheduleCfg;

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

} // namespace Atvoss::Reduce
#endif