/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

    using BuffMaps = BuffIdMap;
};

template <typename OriginalArgs, typename LocalVars, typename BufPools, typename BuffIdMap = void>
ContextData(OriginalArgs, LocalVars, BufPools&) -> ContextData<OriginalArgs, LocalVars, BufPools, BuffIdMap>;

} // namespace Atvoss
#endif