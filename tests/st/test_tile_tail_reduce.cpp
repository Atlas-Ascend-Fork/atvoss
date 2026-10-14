/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "atvoss.h"
#include "kernel_operator.h"

namespace {
constexpr int32_t TILE_LEN = 511;
constexpr size_t TILE_COUNT = 56 * 2;
constexpr size_t ELEMENT_COUNT = TILE_COUNT * TILE_LEN + 1;

template <Atvoss::Pattern ReducePattern, Atvoss::Pattern BroadcastPattern, int32_t Height, int32_t Width>
struct TailReduceCompute {
    using TileShape = Atvoss::Shape<Height, Width>;

    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in = Atvoss::PlaceHolder<1, Tensor<float>, Atvoss::ParamUsage::IN>();
        auto scale = Atvoss::PlaceHolder<2, Tensor<float>, Atvoss::ParamUsage::IN>();
        auto out = Atvoss::PlaceHolder<3, Tensor<float>, Atvoss::ParamUsage::OUT>();
        auto sum = Atvoss::ReduceSum<ReducePattern>(in * in);
        auto mean = Atvoss::Divs<TILE_LEN>(Atvoss::Broadcast<BroadcastPattern>(sum));
        return out = scale * (in / Atvoss::Sqrt(mean));
    }
};

template <Atvoss::Pattern ReducePattern, Atvoss::Pattern BroadcastPattern, int32_t Height, int32_t Width>
struct TailReduceConfig {
    using TileShape = typename TailReduceCompute<ReducePattern, BroadcastPattern, Height, Width>::TileShape;
    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};
    using ArchTag = Atvoss::Arch::DAV_3510;
    using Compute = TailReduceCompute<ReducePattern, BroadcastPattern, Height, Width>;
    using BlockOp = Atvoss::Ele::BlockBuilder<Compute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig>;
    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;
    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

template <Atvoss::Pattern ReducePattern, Atvoss::Pattern BroadcastPattern, int32_t Height, int32_t Width>
int RunTailCase()
{
    if (aclInit(nullptr) != ACL_ERROR_NONE) {
        return 1;
    }
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void* rawIn = nullptr;
    void* rawScale = nullptr;
    void* rawOut = nullptr;
    int rc = 1;
    do {
        if (aclrtSetDevice(0) != ACL_ERROR_NONE || aclrtCreateContext(&context, 0) != ACL_ERROR_NONE ||
            aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
            break;
        }
        const size_t bytes = ELEMENT_COUNT * sizeof(float);
        if (aclrtMalloc(&rawIn, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE ||
            aclrtMalloc(&rawScale, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE ||
            aclrtMalloc(&rawOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
            break;
        }
        constexpr size_t tailIndex = ELEMENT_COUNT - 1;
        std::vector<float> in(ELEMENT_COUNT, 1.0F);
        in[tailIndex] = 7.0F;
        std::vector<float> scale(ELEMENT_COUNT, 1.0F);
        std::vector<float> out(ELEMENT_COUNT, 0.0F);
        if (aclrtMemcpy(rawIn, bytes, in.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
            aclrtMemcpy(rawScale, bytes, scale.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
            break;
        }
        uint64_t shape[] = {ELEMENT_COUNT};
        Atvoss::Tensor<float> inTensor(static_cast<float*>(rawIn), shape, 1);
        Atvoss::Tensor<float> scaleTensor(static_cast<float*>(rawScale), shape, 1);
        Atvoss::Tensor<float> outTensor(static_cast<float*>(rawOut), shape, 1);
        auto args = Atvoss::ArgumentsBuilder{}.inputOutput(inTensor, scaleTensor, outTensor).build();
        typename TailReduceConfig<ReducePattern, BroadcastPattern, Height, Width>::DeviceOp op;
        if (op.Run(args, stream) < 0 || aclrtSynchronizeStream(stream) != ACL_ERROR_NONE ||
            aclrtMemcpy(out.data(), bytes, rawOut, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            break;
        }
        const float expectedTail = std::sqrt(static_cast<float>(TILE_LEN));
        size_t bad = 0;
        for (size_t index = 0; index < out.size(); ++index) {
            const float expected = index == tailIndex ? expectedTail : 1.0F;
            const float value = out[index];
            if (!std::isfinite(value) || std::abs(value - expected) > 1e-3F) {
                ++bad;
            }
        }
        std::cout << (ReducePattern == Atvoss::Pattern::AR ? "AR" : "RA") << ": bad=" << bad
                  << " tail=" << out[tailIndex] << " expected=" << expectedTail << std::endl;
        rc = bad == 0 ? 0 : 2;
    } while (false);
    if (rawOut != nullptr) {
        aclrtFree(rawOut);
    }
    if (rawScale != nullptr) {
        aclrtFree(rawScale);
    }
    if (rawIn != nullptr) {
        aclrtFree(rawIn);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}
} // namespace

int main(int argc, char* argv[])
{
    return RunTailCase<Atvoss::Pattern::AR, Atvoss::Pattern::AB, 1, TILE_LEN>();
}
