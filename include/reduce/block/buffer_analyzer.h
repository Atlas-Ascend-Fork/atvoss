/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOSS_INCLUDE_REDUCE_BLOCK_BUFFER_ANALYZER_H_
#define ATVOSS_INCLUDE_REDUCE_BLOCK_BUFFER_ANALYZER_H_

#include <cstdint>
#include "utils/buf_pool/block_buf_pool.h"

namespace Atvoss::Reduce {

template <typename ReduceDag,
          uint32_t MaxBufCount   = 1,
          uint32_t MaxUBTileSize = 16 * 1024,
          int32_t  CacheBufSize  = 16 * 1024>
class BlockBufferAnalyzer {
    // 编译期：从 DAG 查询各阶段 buffer 需求数量
    static constexpr uint32_t PRE_MTE2_NUM      = ReduceDag::template GetMTE2Num<true>() * CONST2;
    static constexpr uint32_t PRE_MTE3_NUM      = ReduceDag::template GetMTE3Num<true>() * CONST2;
    static constexpr uint32_t PRE_TEMP_CALC_NUM = ReduceDag::template GetTempBufNum<true>();
    static constexpr uint32_t POST_MTE2_NUM     = ReduceDag::template GetMTE2Num<false>();
    static constexpr uint32_t POST_MTE3_NUM     = ReduceDag::template GetMTE3Num<false>();
    static constexpr uint32_t POST_TEMP_CALC_NUM = ReduceDag::template GetTempBufNum<false>();

public:
    using BufPoolType = Atvoss::BlockBufferEx<MaxBufCount, MaxUBTileSize>;

    uint32_t preBufSize = 0;
    uint32_t postBufSize = 0;
    uint32_t resBufSize = 0;
    uint32_t actualBufSize = 0;

#if !defined(__ATVOSS_HOST_ONLY__)
    template <typename ScheduleCfg>
    __aicore__ void Init(ScheduleCfg& cfg, BufPoolType& bufPool)
    {
        preBufSize  = cfg.basicBlock  * (PRE_MTE2_NUM + PRE_MTE3_NUM + PRE_TEMP_CALC_NUM);
        postBufSize = cfg.resultBlock * (POST_MTE2_NUM + POST_TEMP_CALC_NUM + POST_MTE3_NUM);
        resBufSize  = cfg.resultBlock;
        actualBufSize = preBufSize + postBufSize + resBufSize + CacheBufSize;
        bufPool.Init(actualBufSize);
    }
#endif
};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BLOCK_BUFFER_ANALYZER_H_
