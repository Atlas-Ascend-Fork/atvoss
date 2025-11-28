/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_ELE_WISE_H
#define KERNEL_ELE_WISE_H
#include <functional>
#include "common/compile_info.h"
#include "kernel_schedule.h"
namespace Atvoss::EleWise {

struct KernelConfig {                     // Kernel layer tiling information
    uint32_t blockNum = 1;          // Number of cores started
    uint32_t unitNumPerCore = 0;    // Average number of unit processed per core
    uint32_t moreUnitCoreNum = 0;   // Number of cores that need to process an additional full unit
    uint32_t tailNum = 0;           // Number of tail elements to be processed by the last core
    uint32_t unitNum = 1;           // Number of elements per unit block
};

enum class KernelPolicySegment {
    Auto = 0U,       // Automatic tiling
    UniformSegment,  // Uniform tiling
    FullAddTail      // Full block and tail block tiling
};

struct KernelPolicy {
    uint32_t blockDimMax;           // Maximum blockDim used
    KernelPolicySegment segmentPolicy;    // Multi-core tiling strategy
};


static constexpr Atvoss::EleWise::KernelPolicy kernelPolicyDefault{48, KernelPolicySegment::UniformSegment};


/*!
 * KernelBuilder: Calculate the tiling information, then determine the GM data that the current core needs to process based on the block ID,
 * and pass it to the block to complete the computation.
*/
template <typename BlockOp, const auto &Policy = kernelPolicyDefault,   typename ScheduleCfg = KernelConfig,
    class Schedule = Kernel::DefaultSchedule<BlockOp, Policy, ScheduleCfg>>
class KernelBuilder {
public:
    using ScheduleClz = Schedule;
    /*!
     * \brief Kernel layer execution function.
     * \param[in] cfg, Tiling information in kernel.
     * \param[in] argTuple, Input and output GM address.
     */
    template <typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args)
    {
       Schedule schedule;
       schedule.Run(cfg, args...);
    }
};

} // namespace Atvoss::Kernel
#endif