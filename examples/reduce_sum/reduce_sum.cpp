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
 * \file reduce_sum.cpp
 * \brief 通过reduce_sum算子示例，展示如何使用Reduce模板实现reduceSum操作
 */

#include "kernel_operator.h"
#include "atvoss.h"
#include "example_common.h"
#include "command_line.h"

static constexpr int32_t TILE_A = 32;
static constexpr int32_t TILE_R = 8;
static constexpr int32_t MAX_DIM = 8;

template <typename TensorDtype>
struct ReduceSumConfig {
    using TileShape = Atvoss::Shape<TILE_A, TILE_R>;

    struct ReduceSumCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<TensorDtype>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<TensorDtype>, Atvoss::ParamUsage::OUT>();
            return (out = Atvoss::ReduceSum(in));
        }
    };

    static constexpr auto blockPolicy = 
        Atvoss::Reduce::DefaultBlockPolicy<TileShape>{TileShape{}};

    static constexpr auto kernelPolicy = 
        Atvoss::Reduce::DefaultSegmentPolicy::MemoryFirst;

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Reduce::BlockBuilder<
        ReduceSumCompute,
        ArchTag,
        blockPolicy,
        Atvoss::Reduce::DefaultBlockConfig>;

    using KernelOp = Atvoss::Reduce::KernelBuilder<
        BlockOp,
        kernelPolicy>;

    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

struct Options {
    std::vector<int> shape;
    std::vector<int> axes;
    bool help;

    Options() : shape({}), axes({}), help(false) {}

    void parse(int argc, char const* argv[])
    {
        CommandLine cmd(argc, argv);

        help = cmd.Present("help") || cmd.Present("h");
        shape = cmd.Get("shape", std::vector<int>{});
        axes = cmd.Get("axes", std::vector<int>{});

        validate(argv[0]);
    }

    void PrintUsage(const char* progName = nullptr) const
    {
        if (!progName)
            progName = "program";

        std::cout << "Usage: " << progName << " [options]\n"
                  << "\n"
                  << "Options:\n"
                  << "  --help                      Print this message\n"
                  << "  --shape=M,N,O,...           Tensor dimensions (e.g., --shape=4,3,224,224)\n"
                  << "  --axes=I,J,K,...            Axes to reduce (e.g., --axes=0,2)\n"
                  << "\n"
                  << "Example:\n"
                  << "  " << progName << " --shape=4,3,224,224 --axes=0,2 \n";
    }

private:
    void validate(const char* progName)
    {
        if (help) {
            return;
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

        bool hasZeroDim = false;
        for (int d : shape) {
            if (d == 0) {
                hasZeroDim = true;
                break;
            }
        }
        
        if (hasZeroDim) {
            std::cout << "[INFO] Input tensor has zero dimension, will handle empty tensor case\n";
        }
        
        if (axes.empty()) {
            std::cerr << "[ERROR] Missing required argument: --axes\n";
            PrintUsage(progName);
            exit(1);
        }
        
        std::vector<int> normalizedAxes;
        for (int axis : axes) {
            int normalizedAxis = axis;
            if (axis < 0) {
                normalizedAxis = static_cast<int>(shape.size()) + axis;
            }
            if (normalizedAxis < -static_cast<int>(shape.size()) || normalizedAxis >= static_cast<int>(shape.size())) {
                std::cerr << "[ERROR] Invalid axis: " << axis << ", must be in range [" 
                          << -static_cast<int>(shape.size()) << ", " << shape.size() - 1 << "]\n";
                exit(1);
            }
            normalizedAxes.push_back(normalizedAxis);
        }
        
        std::set<int> uniqueAxes(normalizedAxes.begin(), normalizedAxes.end());
        if (uniqueAxes.size() != normalizedAxes.size()) {
            std::cerr << "[ERROR] Duplicate axes found\n";
            exit(1);
        }
        
        const_cast<std::vector<int>&>(axes) = normalizedAxes;
    }
};

template <typename TensorDtype>
static void Run(const Options& options)
{
    CHECK_ACL_RET(aclInit(nullptr));
    auto finalizeGuard = ReleaseSource([]() { aclFinalize(); });

    const int32_t deviceId = 0;
    CHECK_ACL_RET(aclrtSetDevice(deviceId));
    auto deviceResetGuard = ReleaseSource([deviceId]() { aclrtResetDevice(deviceId); });

    aclrtContext context = nullptr;
    CHECK_ACL_RET(aclrtCreateContext(&context, deviceId));
    auto contextDestroyGuard = ReleaseSource([context]() { aclrtDestroyContext(context); });

    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));
    auto streamDestroyGuard = ReleaseSource([stream]() { aclrtDestroyStream(stream); });

    const auto& shape = options.shape;
    const auto& axes = options.axes;
    
    std::set<int> axesSet(axes.begin(), axes.end());
    std::vector<int> outShapeVec;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (axesSet.find(static_cast<int>(i)) == axesSet.end()) {
            outShapeVec.push_back(shape[i]);
        }
    }
    
    uint64_t outShapeArray[MAX_DIM] = {0};
    std::copy(outShapeVec.begin(), outShapeVec.end(), outShapeArray);
    size_t outShapeSize = outShapeVec.size();
    
    bool hasZeroReduceDim = false;
    bool hasZeroNonReduceDim = false;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            if (axesSet.find(static_cast<int>(i)) != axesSet.end()) {
                hasZeroReduceDim = true;
            } else {
                hasZeroNonReduceDim = true;
            }
        }
    }
    
    size_t outputSize = std::accumulate(outShapeVec.begin(), outShapeVec.end(), size_t{1}, std::multiplies<>{}) * sizeof(TensorDtype);
    
    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    TensorDtype* deviceOutput = static_cast<TensorDtype*>(rawOutput);
    
    if (hasZeroNonReduceDim) {
        std::cout << "Empty tensor case (non-reduce axis is 0), output is empty" << std::endl;
    } else if (hasZeroReduceDim) {
        std::cout << "Empty tensor case (only reduce axis is 0), output is all zeros" << std::endl;
        size_t outputElementCount = outputSize / sizeof(TensorDtype);
        std::vector<TensorDtype> hostOutputOnes(outputElementCount, static_cast<TensorDtype>(0));
        CHECK_ACL_RET(aclrtMemcpy(deviceOutput, outputSize, hostOutputOnes.data(), outputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        const size_t shapeSize = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
        const size_t inputSize = shapeSize * sizeof(TensorDtype);
        
        void* rawInput = nullptr;
        CHECK_ACL_RET(aclrtMalloc(&rawInput, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
        auto inputFreeGuard = ReleaseSource([rawInput]() { aclrtFree(rawInput); });
        TensorDtype* deviceInput = static_cast<TensorDtype*>(rawInput);
        
        std::vector<TensorDtype> hostInput(shapeSize);
        for (size_t i = 0; i < shapeSize; ++i) {
            hostInput[i] = static_cast<TensorDtype>(i % 10);
        }
        CHECK_ACL_RET(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint64_t shapeArray[MAX_DIM] = {0};
        std::copy(shape.begin(), shape.end(), shapeArray);
        Atvoss::Tensor<TensorDtype> in(deviceInput, shapeArray, shape.size());
        
        Atvoss::Tensor<TensorDtype> out(deviceOutput, outShapeArray, outShapeSize);
        auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in, out).build();

        using DeviceOp = typename ReduceSumConfig<TensorDtype>::DeviceOp;
        DeviceOp deviceOp;
        deviceOp.Run(arguments, stream);

        CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    }
    
    std::vector<TensorDtype> hostOutput(outputSize / sizeof(TensorDtype));
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "ReduceSum computation completed." << std::endl;
    std::cout << "Input shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Axes: [";
    for (size_t i = 0; i < axes.size(); ++i) {
        std::cout << axes[i];
        if (i < axes.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Output shape: [";
    for (size_t i = 0; i < outShapeSize; ++i) {
        std::cout << outShapeArray[i];
        if (i < outShapeSize - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "First 10 output values: ";
    for (size_t i = 0; i < std::min(size_t(10), hostOutput.size()); ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char const* argv[])
{
    Options options;
    options.parse(argc, argv);

    if (options.help) {
        options.PrintUsage(argv[0]);
        return 0;
    }

    std::cout << "Start reduce_sum float" << std::endl;
    Run<float>(options);

    return 0;
}