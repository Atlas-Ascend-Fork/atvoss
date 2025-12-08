/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_INCLUDE_BLOCK_SCHEDULE_H_
#define Atvoss_INCLUDE_BLOCK_SCHEDULE_H_
#include "base_schedule.h"
namespace Atvoss::EleWise {

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
template <typename Compute, const auto& Policy, typename ScheduleCfg>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, true> {};
#else
template <typename Compute, const auto& Policy, typename ScheduleCfg>
class DefaultBlockSchedule : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, false> {};
#endif

template <typename Compute, const auto& Policy, typename ScheduleCfg>
class BlockScheduleWithTPipe : public BaseBlockSchedule<Compute, Policy, ScheduleCfg, true> {};

} // namespace Atvoss::EleWise
#endif // Atvoss_INCLUDE_KERNEL_KERNEL_SCHEDULE_H_
