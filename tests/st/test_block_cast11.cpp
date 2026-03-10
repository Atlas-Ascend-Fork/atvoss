/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory>
#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "command_line.h"

static constexpr int32_t MAX_DIM = 8;

template <typename T1, typename T2, int32_t tileShapeLen>
struct CastConfig {
    using DtypeV1 = T1;
    using DtypeV2 = T2;

    using TileShape = Atvoss::Shape<tileShapeLen>;
    struct CastCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::IN>();
            auto out1 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::OUT>();
            return (out1 = Atvoss::Cast<Atvoss::CastMode::CAST_NONE, DtypeV2>(in1));
        };
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}, Atvoss::MemMngPolicy::MANUAL};

    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};

    using ArchTag = Atvoss::Arch::DAV_3510;

    using BlockOp = Atvoss::Ele::BlockBuilder<CastCompute, ArchTag, blockPolicy>;

    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

struct Options {
    // 存储解析后的值
    std::vector<int> shape;
    std::string typeInput;
    std::string typeOutput;
    bool help;

    // 默认构造：不解析，只设默认值
    Options() : shape({}), typeInput("f16"), typeOutput("f32"), help(false)
    {}
};

template <typename T1, typename T2, int32_t tileShapeLen>
static void Run(const Options& options)
{
    // --- Step 1: ACL 初始化 ---
    CHECK_ACL_RET(aclInit(nullptr));
    auto finalizeGuard = ReleaseSource([]() { aclFinalize(); });
    // --- Step 2: 设置 device ---
    const int32_t deviceId = 0;
    CHECK_ACL_RET(aclrtSetDevice(deviceId));
    auto deviceResetGuard = ReleaseSource([deviceId]() { aclrtResetDevice(deviceId); });

    // --- Step 3: 创建 Context ---
    aclrtContext context = nullptr;
    CHECK_ACL_RET(aclrtCreateContext(&context, deviceId));
    auto contextDestroyGuard = ReleaseSource([context]() { aclrtDestroyContext(context); });

    // --- Step 4: 创建 Stream ---
    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));
    auto streamDestroyGuard = ReleaseSource([stream]() { aclrtDestroyStream(stream); });

    // --- Step 5: 准备数据 ---
    const auto& shape = options.shape;
    const size_t shapeSize = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
    const size_t inputSize = shapeSize * sizeof(T1);
    const size_t outputSize = shapeSize * sizeof(T2);

    // --- Step 6: 分配设备内存 ---
    void* rawInput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard = ReleaseSource([rawInput]() { aclrtFree(rawInput); });
    T1* deviceInput = static_cast<T1*>(rawInput);

    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    T2* deviceOutput = static_cast<T2*>(rawOutput);

    // --- Step 7: Host -> Device 数据拷贝 ---
    std::vector<T1> hostInput(shapeSize, static_cast<T1>(1.5F));
    CHECK_ACL_RET(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 8: 构造 Tensor 并执行算子 ---
    uint64_t shapeArray[MAX_DIM] = {0};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<T1> t1(deviceInput, shapeArray, shape.size());
    Atvoss::Tensor<T2> t2(deviceOutput, shapeArray, shape.size());
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2).build();

    using DeviceOp = typename CastConfig<T1, T2, tileShapeLen>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T2> hostOutput(shapeSize);
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T2> golden(shapeSize, 1.5f);
    if (!VerifyResults(golden, hostOutput)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    Options options;
    std::cout << "case11 test tilelen" << std::endl;
    std::cout << "Start cast data from f16 to f32" << std::endl;
    options.shape = std::vector<int>{64, 8192};
    Run<half, float, 17>(options);

    return 0;
}
