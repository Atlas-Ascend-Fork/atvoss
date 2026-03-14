/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "command_line.h"
#include "utils/layout/shape.h"

static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;
static constexpr int32_t MAX_DIM = 2;

template <typename T1, typename T2, typename T3>
struct RmsNormConfig {
    using DtypeV1 = T1;
    using DtypeV2 = T2;
    using DtypeV3 = T3;
    using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;

    struct RmsNormCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV3>, Atvoss::ParamUsage::OUT>();
            auto _1 = Atvoss::ReduceSum<Atvoss::Pattern::AR>(in1 * in1);
            auto _2 = Atvoss::Broadcast<Atvoss::Pattern::AB>(_1);
            auto _3 = in1 / Atvoss::Sqrt(Atvoss::Divs<WIDTH>(_2));
            return out = in2 * _3;
        }
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy {
        TileShape{}
    };
    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy {
        Atvoss::Ele::DefaultSegmentPolicy::UniformSegment
    };

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<
        RmsNormCompute,
        ArchTag,
        blockPolicy,
        Atvoss::Ele::DefaultBlockConfig>;

    using KernelOp = Atvoss::Ele::KernelBuilder<
        BlockOp,
        kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};


struct Options {
    // 存储解析后的值
    std::vector<int> shape;
    bool help;

    // 默认构造：不解析，只设默认值
    Options() : shape({}), help(false)
    {
    }

    // 解析函数
    void parse(int argc, char const* argv[])
    {
        CommandLine cmd(argc, argv);

        help = cmd.Present("help") || cmd.Present("h");
        shape = cmd.Get("shape", std::vector<int>{});

        validate(argv[0]);
    }

    // 打印使用说明
    void PrintUsage(const char* progName = nullptr) const
    {
        if (!progName)
            progName = "program";

        std::cout << "Usage: " << progName << " [options]\n"
                  << "\n"
                  << "Options:\n"
                  << "  --help                      Print this message\n"
                  << "  --shape=M,N,...           Tensor dimensions (e.g., --shape=512,32)\n"
                  << "\n"
                  << "Example:\n"
                  << "  " << progName << " --shape=512,32 \n";
    }

    // 打印当前配置
    void print() const
    {
        std::cout << "Configuration:\n";
        std::cout << "  Tensor Shape [" << shape.size() << "D]: (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i != shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << ")\n";
    }

private:
    static std::string ToLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    }

    void validate(const char* progName)
    {
        if (help) {
            return;  // 帮助不需要验证
        }

        if (shape.empty()) {
            std::cerr << "[ERROR] Missing required argument: --shape\n";
            PrintUsage(progName);
            exit(1);
        }
        if (shape.size() > MAX_DIM) {
            std::cerr << "[ERROR] Input shape max dim is 2, current shape dim is: " << shape.size() << "\n";
            PrintUsage(progName);
            exit(1);
        }
        for (int d : shape) {
            if (d < 0) {
                std::cerr << "[ERROR] Invalid dimension size: " << d << "\n";
                exit(1);
            }
        }
    }
};

template <typename T1, typename T2, typename T3>
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
    const size_t outputSize = shapeSize * sizeof(T3);

    // --- Step 6: 分配设备内存 ---
    void* rawInput1 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput1, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard1 = ReleaseSource([rawInput1]() { aclrtFree(rawInput1); });
    T1* deviceInput1 = static_cast<T1*>(rawInput1);

    void* rawInput2 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput2, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard2 = ReleaseSource([rawInput2]() { aclrtFree(rawInput2); });
    T2* deviceInput2 = static_cast<T2*>(rawInput2);

    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    T3* deviceOutput = static_cast<T3*>(rawOutput);

    // --- Step 7: Host -> Device 数据拷贝 ---
    std::vector<T1> hostInput1(shapeSize, static_cast<T1>(1.0F));
    CHECK_ACL_RET(aclrtMemcpy(deviceInput1, inputSize, hostInput1.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<T2> hostInput2(shapeSize, static_cast<T2>(2.0F));
    CHECK_ACL_RET(aclrtMemcpy(deviceInput2, inputSize, hostInput2.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 8: 构造 Tensor 并执行算子 ---
    uint64_t shapeArray[MAX_DIM] = {0};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<T1> t1(deviceInput1, shapeArray, shape.size());
    Atvoss::Tensor<T2> t2(deviceInput2, shapeArray, shape.size());
    Atvoss::Tensor<T3> t3(deviceOutput, shapeArray, shape.size());
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, t3).build();

    using DeviceOp = typename RmsNormConfig<T1, T2, T3>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T3> hostOutput(shapeSize);
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T3> golden(shapeSize, 2.0f);
    if (!VerifyResults(golden, hostOutput)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    Options options;
    options.parse(argc, argv);

    if (options.help) {
        options.PrintUsage(argv[0]);
        return 0;
    }

    std::cout << "Start rms_norm" << std::endl;
    Run<float, float, float>(options);
    return 0;
}
