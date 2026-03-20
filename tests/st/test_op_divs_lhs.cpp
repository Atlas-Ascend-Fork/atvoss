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
 * \file test_op_divs_lhs.cpp
 * \brief 测试/运算符的功能，左操作数是scalar
 */

#include "atvoss.h"
#include "example_common.h"
#include "utils/layout/shape.h"

static constexpr int32_t MAX_DIM = 8;

template <typename T1, typename T2>
struct DivsLhsConfig {
    struct DivsLhsCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<T1>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<T2>, Atvoss::ParamUsage::OUT>();
            auto other = Atvoss::PlaceHolder<3, T1, Atvoss::ParamUsage::IN>();

            return (out = other / in);
        };
    };

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<DivsLhsCompute, ArchTag>;
    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp>;
    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

template <typename T1, typename T2>
static void Run()
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

    // --- Step 5: 分配设备内存 ---
    void* in = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&in, 8 * sizeof(T1), ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard = ReleaseSource([in]() { aclrtFree(in); });
    T1* deviceIn = static_cast<T1*>(in);
    void* out = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&out, 8 * sizeof(T2), ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([out]() { aclrtFree(out); });
    T2* deviceOutput = static_cast<T2*>(out);

    // --- Step 6: Host -> Device 数据拷贝 ---
    std::vector<T1> hostInput(8 * sizeof(T1), static_cast<T1>(5.0));
    CHECK_ACL_RET(aclrtMemcpy(deviceIn, 8 * sizeof(T1), hostInput.data(), 8 * sizeof(T1), ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 7: 构造 Args ---
    uint64_t shapeArray[MAX_DIM] = {8, 0, 0, 0, 0, 0, 0, 0};
    Atvoss::Tensor<T1> t1(deviceIn, shapeArray, 1);
    Atvoss::Tensor<T2> t2(deviceOutput, shapeArray, 1);
    float other = 15.0;
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, other).build();

    // --- Step 8: 执行算子 ---
    using DeviceOp = typename DivsLhsConfig<T1, T2>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T2> hostOutput(8);
    CHECK_ACL_RET(
        aclrtMemcpy(hostOutput.data(), 8 * sizeof(T2), deviceOutput, 8 * sizeof(T2), ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T2> golden(8, 3.0);
    if (!VerifyResults(golden, hostOutput)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    Run<float, float>();

    return 0;
}
