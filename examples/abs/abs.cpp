/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "command_line.h"

static constexpr int32_t TILE_SIZE = 32;
static constexpr int32_t MAX_DIM = 8;

template <typename T>
struct AbsConfig {
    using Dtype = T;

    using TileShape = Atvoss::Shape<TILE_SIZE>;
    struct AbsCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<Dtype>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<Dtype>, Atvoss::ParamUsage::OUT>();
            return (out = Abs(in));
        };
    };

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};

    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};

    using ArchTag = Atvoss::Arch::DAV_3510;

    using BlockOp = Atvoss::Ele::BlockBuilder<AbsCompute, ArchTag, blockPolicy, Atvoss::Ele::DefaultBlockConfig>;

    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp, kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

struct Options {
    // 存储解析后的值
    std::vector<int> shape;
    std::string typeInput;
    bool help;

    // 默认构造：不解析，只设默认值
    Options() : shape({}), typeInput("f32"), help(false)
    {}

    // 解析函数
    void parse(int argc, char const* argv[])
    {
        CommandLine cmd(argc, argv);

        help = cmd.Present("help") || cmd.Present("h");
        shape = cmd.Get("shape", std::vector<int>{});
        typeInput = ToLower(cmd.Get("type_input", std::string("f32")));

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
                  << "  --type_input=TYPE           Input data type (f32; default: f32)\n"
                  << "\n"
                  << "Example:\n"
                  << "  " << progName << " --shape=512,256,128 --type_input=f32 \n";
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

        std::vector<std::string> validTypes = {"f32"};
        auto checkType = [&](const std::string& t, const char* name) {
            if (std::find(validTypes.begin(), validTypes.end(), t) == validTypes.end()) {
                std::cerr << "[ERROR] Invalid " << name << " type '" << t << "'\n";
                PrintUsage(progName);
                exit(1);
            }
        };

        checkType(typeInput, "input");
    }
};

template <typename T>
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
    const size_t inputSize = shapeSize * sizeof(T);
    const size_t outputSize = shapeSize * sizeof(T);
    void* rawInput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard = ReleaseSource([rawInput]() { aclrtFree(rawInput); });
    T* deviceInput = static_cast<T*>(rawInput);

    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    T* deviceOutput = static_cast<T*>(rawOutput);

    // --- Step 6: Host -> Device 数据拷贝 ---
    std::vector<T> hostInput(shapeSize, static_cast<T>(-1.5F));
    CHECK_ACL_RET(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 7: 构造参数信息 ---
    uint64_t shapeArray[MAX_DIM] = {0};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<T> t1(deviceInput, shapeArray, shape.size());
    Atvoss::Tensor<T> t2(deviceOutput, shapeArray, shape.size());
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2).build();

    // --- Step 8: 执行算子 ---
    using DeviceOp = typename AbsConfig<T>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T> hostOutput(shapeSize);
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T> golden(shapeSize, 1.5f);
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

    // 继续执行计算逻辑...
    if (options.typeInput == "f32") {
        std::cout << "Start abs fp32 data" << std::endl;
        Run<float>(options);
    }

    return 0;
}
