/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file buffer_analyzer.h
 * @brief Reduce Block 层 Buffer 分析模块: DAG查询 + UB大小计算 + buffer pool初始化
 */

#ifndef ATVOSS_INCLUDE_REDUCE_BUFFER_ANALYZER_H_
#define ATVOSS_INCLUDE_REDUCE_BUFFER_ANALYZER_H_

#include <cstdint>
#include "utils/buf_pool/block_buf_pool.h"

namespace Atvoss::Reduce {

/**
 * @brief Block 层 Buffer 分析器
 * @tparam ReduceDag DAG 分析结果类型 (ReduceAutoDag)
 * @tparam MaxBufCount UB buffer 最大数量
 * @tparam MaxUBTileSize UB tile 最大字节数
 * @tparam CacheBufSize Cache buffer 字节数
 */
template <typename ReduceDag,
          uint32_t MaxBufCount   = 1,
          uint32_t MaxUBTileSize = 16 * 1024,
          int32_t  CacheBufSize  = 16 * 1024>
class BufferAnalyzer {
    static constexpr uint32_t PRE_MTE2_NUM       = ReduceDag::template GetMTE2Num<true>() * CONST2;
    static constexpr uint32_t PRE_MTE3_NUM       = ReduceDag::template GetMTE3Num<true>() * CONST2;
    static constexpr uint32_t PRE_TEMP_CALC_NUM  = ReduceDag::template GetTempBufNum<true>();
    static constexpr uint32_t POST_MTE2_NUM      = ReduceDag::template GetMTE2Num<false>();
    static constexpr uint32_t POST_MTE3_NUM      = ReduceDag::template GetMTE3Num<false>();
    static constexpr uint32_t POST_TEMP_CALC_NUM = ReduceDag::template GetTempBufNum<false>();

public:
    using BufPoolType = Atvoss::BlockBufferEx<MaxBufCount, MaxUBTileSize>;

    uint32_t preBufSize = 0;        // PreReduce 阶段 UB 大小
    uint32_t postBufSize = 0;       // PostReduce 阶段 UB 大小
    uint32_t resBufSize = 0;        // 结果 buffer 大小
    uint32_t actualBufSize = 0;     // 实际 UB 总需求

#if !defined(__ATVOSS_HOST_ONLY__)
    /**
     * @brief 根据 blockConfig 计算 UB 大小并初始化 buffer pool
     * @tparam ScheduleCfg Block 配置类型
     * @param cfg Block 配置
     * @param bufPool 待初始化的 buffer pool
     */
    template <typename ScheduleCfg>
    __aicore__ void Init(ScheduleCfg& cfg, BufPoolType& bufPool)
    {
        preBufSize  = cfg.basicBlock  * (PRE_MTE2_NUM + PRE_MTE3_NUM + PRE_TEMP_CALC_NUM);
        postBufSize = cfg.resultBlock * (POST_MTE2_NUM + POST_TEMP_CALC_NUM + POST_MTE3_NUM);
        resBufSize  = cfg.resultBlock;
        actualBufSize = preBufSize + postBufSize + resBufSize + CacheBufSize;
        // AscendC::printf("buffer analyzer preBufSize: %d\n", preBufSize);
        // AscendC::printf("kernel analyzer postBufSize: %d\n", postBufSize);
        // AscendC::printf("kernel analyzer resBufSize: %d\n", resBufSize);
        // AscendC::printf("kernel analyzer actualBufSize: %d\n", actualBufSize);
        bufPool.Init(actualBufSize, preBufSize, postBufSize, resBufSize);
    }
#endif
};

} // namespace Atvoss::Reduce
#endif // ATVOSS_INCLUDE_REDUCE_BUFFER_ANALYZER_H_
