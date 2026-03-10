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
 * \file test_scalar_param_order_case1.cpp
 * \brief 测试Scalar场景：inputOutput传参顺序
 * case1：order: input, output, scalar
 */

#include "atvoss.h"
#include "example_common.h"
#include "utils/layout/shape.h"

static constexpr int32_t MAX_DIM = 8;

template <typename T1, typename T2>
struct ScalarParamOrderConfig {
    struct ScalarParamOrderCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<T1>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<T2>, Atvoss::ParamUsage::OUT>();
            auto scalar = Atvoss::PlaceHolder<3, T1, Atvoss::ParamUsage::IN>();

            return (out = scalar * in);
        };
    };

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<ScalarParamOrderCompute, ArchTag>;
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
    void* in = nullptr; // 输入1
    CHECK_ACL_RET(aclrtMalloc(&in, 9 * sizeof(T1), ACL_MEM_MALLOC_HUGE_FIRST));
    auto input1FreeGuard = ReleaseSource([in]() { aclrtFree(in); });
    T1* deviceIn = static_cast<T1*>(in);

    void* out = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&out, 9 * sizeof(T2), ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([out]() { aclrtFree(out); });
    T2* deviceOutput = static_cast<T2*>(out);

    // --- Step 6: Host -> Device 数据拷贝 ---
    std::vector<T1> hostInput = {512, 256, 128, 128, 256, 512, 256, 128, 512};
    CHECK_ACL_RET(aclrtMemcpy(deviceIn, 9 * sizeof(T1), hostInput.data(), 9 * sizeof(T1), ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 7: 构造 Args ---
    uint64_t shapeArray[MAX_DIM] = {3, 3, 0, 0, 0, 0, 0, 0};
    Atvoss::Tensor<T1> t1(deviceIn, shapeArray, 2);

    Atvoss::Tensor<T1> t2(deviceOutput, shapeArray, 2);

    float scalar = 2.2;
    // order: t1, t2, scalar
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, scalar).build();
    using DeviceOp = typename ScalarParamOrderConfig<T1, T2>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T2> hostOutput(9);
    CHECK_ACL_RET(
        aclrtMemcpy(hostOutput.data(), 9 * sizeof(T2), deviceOutput, 9 * sizeof(T2), ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T2> golden = {scalar * 512, scalar * 256, scalar * 128, scalar * 128, scalar * 256,
                              scalar * 512, scalar * 256, scalar * 128, scalar * 512};
    if (!VerifyResults(golden, hostOutput)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    std::cout << "----开始测试Scalar场景: inputOutput传参顺序----" << std::endl;
    Run<float, float>();
    std::cout << "-------------------测试完毕!------------------" << std::endl;
    return 0;
}
