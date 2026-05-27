/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tilingdata.h
 * \brief Reduce算子统一的Tiling数据结构定义
 *
 * 本文件定义了Reduce算子在Kernel层和Block层共享的Tiling数据结构：
 * - ReduceTilingData: 统一的Tiling数据容器，包含所有切分参数
 * - ReduceTilingUnit: 单个切分轴的参数
 * - CacheLineBlock: CacheLine切分参数
 *
 * Kernel层在MakeScheduleConfig中一次性计算所有Tiling数据，
 * Block层按需从中提取需要的参数。
 */

#ifndef ATVOSS_REDUCE_TILINGDATA_H
#define ATVOSS_REDUCE_TILINGDATA_H

#include <cstdint>

namespace Atvoss::Reduce {

constexpr size_t MAX_DIM = 8; // 最大维度数

/*!
 * \brief 统一的Reduce Tiling数据结构
 *
 * 包含Kernel层和Block层所有需要的Tiling参数，
 * 在Kernel层一次性计算完成，Block层按需提取。
 *
 * 数据分组：
 * 1. 基础信息：patternID、coreNum、dtype大小
 * 2. Shape信息：变换后的shape、stride
 * 3. Tiling单元：unitA、unitR、cacheLineBlock
 * 4. Block参数：basicBlock、resultBlock、ubFactor
 * 5. 多核切分：tileCnt、groupR
 */
struct ReduceTilingData {
    // ========== 基础信息 ==========
    uint32_t patternID = 0;      // Pattern ID（A/AR/ARA等）
    uint32_t coreNum = 0;        // 实际使用的核心数
    uint32_t useNddma = 0;       // 是否使用NDDMA搬运

    // ========== Shape信息（变换后） ==========
    uint64_t shape[MAX_DIM] = {0};     // 变换后的Shape
    uint64_t srcStride[MAX_DIM] = {0}; // 输入Stride
    uint64_t outStride[MAX_DIM] = {0}; // 输出Stride

    // ========== Block层参数 ==========
    uint64_t basicBlock = 0;  // UB搬入的buffer大小（字节）
    uint64_t resultBlock = 0; // Reduce计算后的buffer大小（字节）
    uint64_t ubFactorA = 0;   // A轴UB切分因子
    uint64_t ubFactorR = 0;   // R轴UB切分因子
    uint64_t tileACntPerCore = 0; // 每个Core处理的A轴Tile数
    uint64_t tileRCntPerCore = 0; // 每个Core处理的R轴Tile数

    // ========== 多核切分参数 ==========
    uint64_t totalTileACnt = 0;   // A轴总Tile数
    uint64_t totalTileRCnt = 0;   // R轴总Tile数
    int16_t groupR = 1;          // R轴跨核Reduce组数

    int8_t loopACount = 1;
    int8_t loopRCount = 1;

};

} // namespace Atvoss::Reduce

#endif // ATVOSS_REDUCE_TILINGDATA_H