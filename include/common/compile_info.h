/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef Atvoss_COMPILE_INFO_H
#define Atvoss_COMPILE_INFO_H
#include "tiling/platform/platform_ascendc.h"
#include <map>
namespace Atvoss {
struct OpCompileInfo {
    uint64_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t cacheLineSize = 0;
    uint64_t ubBlockSize = 0;
    OpCompileInfo(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
    {
        this->vectorCoreNum = a;
        this->ubBlockSize = d;
        this->ubSize = b;
        this->cacheLineSize = c;
    }
};

inline OpCompileInfo GetOpCompileInfo()
{
    const auto& platformInfoMgr = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platformInfoMgr == nullptr) {
        printf("[ERROR]: [Atvoss][Common] Failed to retrieve platform infomation.\n");
        return {0, 0, 0, 0};
    }
    auto soc = platformInfoMgr->GetSocVersion();
    static const std::map<platform_ascendc::SocVersion, OpCompileInfo> compileInfoMap = {
        {platform_ascendc::SocVersion::ASCEND910B, {48, 196352, 256, 32}},
        {platform_ascendc::SocVersion::ASCEND310B, {1, 262144, 256, 32}},
        {platform_ascendc::SocVersion::ASCEND310P, {8, 262144, 256, 32}},
    };

    auto findRes = compileInfoMap.find(soc);
    if (findRes != compileInfoMap.cend()) {
        OpCompileInfo compileInfo = findRes->second;
        if (soc == platform_ascendc::SocVersion::ASCEND910B) {
            compileInfo.vectorCoreNum = platformInfoMgr->GetCoreNumAiv();
            uint64_t ubSize = compileInfo.ubSize;
            platformInfoMgr->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
            compileInfo.ubSize = ubSize;
        }
        return compileInfo;
    }
    printf("[ERROR]: [Atvoss][Common] Current framework does not support this current device. Please check chip version.\n");
    return {0, 0, 0, 0};
}
} // namespace Atvoss
#endif
