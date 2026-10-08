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
 * \\file test_op_divs_int32.cpp
 * \\brief Test integer division by a compile-time scalar.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"

static constexpr uint64_t kCount = 4;
static constexpr int32_t kDeviceId = 0;
static int exitCode = 1;

template <typename T>
struct DivsInt32Config {
    using TileShape = Atvoss::Shape<4>;
    struct DivsCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<T>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<T>, Atvoss::ParamUsage::OUT>();
            return out = Atvoss::Divs<2>(in);
        }
    };
    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<DivsCompute, ArchTag, blockPolicy>;
    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp>;
    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

static void Run()
{
    const std::vector<int32_t> input{4, 6, -4, 100};
    const std::vector<int32_t> expected{2, 3, -2, 50};
    CHECK_ACL_RET(aclInit(nullptr));
    auto finalize = ReleaseSource([]() { aclFinalize(); });
    CHECK_ACL_RET(aclrtSetDevice(kDeviceId));
    auto reset = ReleaseSource([]() { aclrtResetDevice(kDeviceId); });
    aclrtContext context = nullptr;
    CHECK_ACL_RET(aclrtCreateContext(&context, kDeviceId));
    auto destroyContext = ReleaseSource([context]() { aclrtDestroyContext(context); });
    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));
    auto destroyStream = ReleaseSource([stream]() { aclrtDestroyStream(stream); });

    void* inputDevice = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&inputDevice, kCount * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    auto freeInput = ReleaseSource([inputDevice]() { aclrtFree(inputDevice); });
    void* outputDevice = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&outputDevice, kCount * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    auto freeOutput = ReleaseSource([outputDevice]() { aclrtFree(outputDevice); });
    CHECK_ACL_RET(aclrtMemcpy(
        inputDevice, kCount * sizeof(int32_t), input.data(), kCount * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t shape[kCount] = {kCount};
    Atvoss::Tensor<int32_t> in(static_cast<int32_t*>(inputDevice), shape, 1);
    Atvoss::Tensor<int32_t> out(static_cast<int32_t*>(outputDevice), shape, 1);
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in, out).build();
    typename DivsInt32Config<int32_t>::DeviceOp op;
    op.Run(arguments, stream);
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));

    std::vector<int32_t> actual(kCount);
    CHECK_ACL_RET(aclrtMemcpy(
        actual.data(), kCount * sizeof(int32_t), outputDevice, kCount * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    bool passed = actual == expected;
    std::cout << "input=";
    for (auto value : input) {
        std::cout << value << ',';
    }
    std::cout << " expected=";
    for (auto value : expected) {
        std::cout << value << ',';
    }
    std::cout << " actual=";
    for (auto value : actual) {
        std::cout << value << ',';
    }
    std::cout << '\n' << (passed ? "PASS" : "FAIL") << '\n';
    exitCode = passed ? 0 : 1;
}

int main()
{
    Run();
    return exitCode;
}
