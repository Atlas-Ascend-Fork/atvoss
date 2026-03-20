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
 * \file muls.cpp
 * \brief 通过muls算子示例，展示用户不同输入信息，不同Compute表达的运算过程，并展示表达式中使用scalar的方式
 */

#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "command_line.h"

static constexpr int32_t WIDTH = 32;
static constexpr int32_t MAX_DIM = 8;

template <typename TensorDtype, typename ScalarDtype>
struct MulsConfig {
    using TileShape = Atvoss::Shape<WIDTH>;

    struct MulsCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<TensorDtype>, Atvoss::ParamUsage::IN>();
            auto scalar = Atvoss::PlaceHolder<2, ScalarDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<TensorDtype>, Atvoss::ParamUsage::OUT>();

            return (out = in * scalar);
        };
    };

    struct MulsComputePromtIn {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<TensorDtype>, Atvoss::ParamUsage::IN>();
            auto scalar = Atvoss::PlaceHolder<2, ScalarDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<ScalarDtype>, Atvoss::ParamUsage::OUT>();
            auto inTmp = Atvoss::PlaceHolderTmpLike<1, Tensor<ScalarDtype>>(in);

            return (inTmp = Atvoss::Cast<Atvoss::CastMode::CAST_NONE, ScalarDtype>(in), out = inTmp * scalar);
        };
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};
    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<
        MulsCompute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig, Atvoss::Ele::DefaultBlockSchedule>;

    using BlockOpPromtIn = Atvoss::Ele::BlockBuilder<
        MulsComputePromtIn, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig, Atvoss::Ele::DefaultBlockSchedule>;

    using KernelOp = Atvoss::Ele::KernelBuilder<
        BlockOp, kernelPolicy, Atvoss::Ele::DefaultKernelConfig, Atvoss::Ele::DefaultKernelSchedule>;
    using KernelOpPromtIn = Atvoss::Ele::KernelBuilder<
        BlockOpPromtIn, kernelPolicy, Atvoss::Ele::DefaultKernelConfig, Atvoss::Ele::DefaultKernelSchedule>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
    using DeviceOpPromtIn = Atvoss::DeviceAdapter<KernelOpPromtIn>;
};

struct Options {
    // 存储解析后的值
    std::vector<int> shape;
    bool help;

    // 默认构造：不解析，只设默认值
    Options() : shape({}), help(false)
    {}

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
                  << "  --shape=M,N,O,...           Tensor dimensions (e.g., --shape=4,3,224,224)\n"
                  << "\n"
                  << "Example:\n"
                  << "  " << progName << " --shape=4,3,224,224 \n";
    }

private:
    void validate(const char* progName)
    {
        if (help) {
            return; // 帮助不需要验证
        }

        if (shape.empty()) {
            std::cerr << "[ERROR] Missing required argument: --shape\n";
            PrintUsage(progName);
            exit(1);
        }
        if (shape.size() > MAX_DIM) {
            std::cerr << "[ERROR] Input shape max dim is 8, current shape dim is: " << shape.size() << "\n";
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

template <typename TensorDtype, typename ScalarDtype>
static void Run(const Options& options)
{
    // --- Step 1: ACL 初始化 ---
    CHECK_ACL_RET(aclInit(nullptr));
    auto finalizeGuard = ReleaseSource([]() { aclFinalize(); });

    // --- Step 2: 设置 device ID ---
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

    // --- Step 5: 分配输入/输出的Device内存 ---
    const auto& shape = options.shape;
    const size_t shapeSize = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
    const size_t inputSize = shapeSize * sizeof(TensorDtype);
    const size_t outputSize = shapeSize * sizeof(ScalarDtype);
    void* rawInput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard = ReleaseSource([rawInput]() { aclrtFree(rawInput); });
    TensorDtype* deviceInput = static_cast<TensorDtype*>(rawInput);
    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    ScalarDtype* deviceOutput = static_cast<ScalarDtype*>(rawOutput);

    // --- Step 6: Host -> Device 数据拷贝 ---
    std::vector<TensorDtype> hostInput(shapeSize, static_cast<TensorDtype>(3.0f));
    CHECK_ACL_RET(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 7: 构造参数信息 ---
    uint64_t shapeArray[MAX_DIM] = {0};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<TensorDtype> in(deviceInput, shapeArray, shape.size());
    Atvoss::Tensor<ScalarDtype> out(deviceOutput, shapeArray, shape.size());
    float scalar = 3.0f;
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in, scalar, out).build();

    // --- Step 8: 执行算子 ---
    if constexpr (std::is_same_v<TensorDtype, float>) {
        using DeviceOp = typename MulsConfig<TensorDtype, ScalarDtype>::DeviceOp;
        DeviceOp deviceOp;
        deviceOp.Run(arguments, stream);
    } else if constexpr (std::is_same_v<TensorDtype, int32_t>) {
        using DeviceOp = typename MulsConfig<TensorDtype, ScalarDtype>::DeviceOpPromtIn;
        DeviceOp deviceOp;
        deviceOp.Run(arguments, stream);
    }

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<ScalarDtype> hostOutput(shapeSize);
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<ScalarDtype> golden(shapeSize, 9.0f);
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

    // 继续执行计算逻辑
    std::cout << "Start muls int32_t and float" << std::endl;
    Run<int32_t, float>(options);

    return 0;
}
