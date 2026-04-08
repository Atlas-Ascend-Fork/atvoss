/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATVOSS_TYPE_DEF_H
#define ATVOSS_TYPE_DEF_H

namespace Atvoss {

template <typename OriginalArgs, typename LocalVars, typename BufPools, typename BuffIdMap = void>
struct ContextData {
    OriginalArgs argsTensors;
    LocalVars tmpTensors;
    BufPools& bufPools;
    uint64_t gmOffset;
    uint64_t elementNum;
    uint32_t pingPong;
    // CopyIn ARAR→AARR 转置参数
    uint64_t segCount = 1;   // A维度分段数 (TileA=ubFactorA)
    uint64_t segSize = 0;    // 每段元素数   (TileR=ubFactorR)
    uint64_t srcStride = 0;  // GM中段间stride (cfg.srcStride[A_split_axis])
    uint64_t dstStride = 0;  // UB中段间stride (segSize, AARR连续布局)
    // Reduce pattern
    int32_t patternID = 0;      // ReduceSchLoopInfo::patternID
    int32_t innerPatternID = 0; // ReduceSchLoopInfo::innerPatternID

    using BuffMaps = BuffIdMap;
};

template <typename OriginalArgs, typename LocalVars, typename BufPools, typename BuffIdMap = void>
ContextData(OriginalArgs, LocalVars, BufPools&) -> ContextData<OriginalArgs, LocalVars, BufPools, BuffIdMap>;

} // namespace Atvoss
#endif