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
 * \file test_compute_assign_rhs_is_param.cpp
 * \brief 测试compute表达：
 * param = param
 * param = localVar
 * localVar = param
 * localVar = localVar
 */

#include "atvoss.h"
#include "example_common.h"
#include "utils/layout/shape.h"

static constexpr int32_t MAX_DIM = 8;

template <typename T1, typename T2>
struct ComputeTestConfig {
    struct ComputeTestCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<T1>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<T1>, Atvoss::ParamUsage::IN>();
            auto out1 = Atvoss::PlaceHolder<3, Tensor<T2>, Atvoss::ParamUsage::OUT>();
            auto out2 = Atvoss::PlaceHolder<4, Tensor<T2>, Atvoss::ParamUsage::OUT>();

            auto _1 = in1;
            auto _2 = _1;

            return (out1 = _2, out2 = in2);
        };
    };

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<ComputeTestCompute, ArchTag>;
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
    void* in1 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&in1, 8 * sizeof(T1), ACL_MEM_MALLOC_HUGE_FIRST));
    auto input1FreeGuard = ReleaseSource([in1]() { aclrtFree(in1); });
    T1* deviceIn1 = static_cast<T1*>(in1);
    void* in2 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&in2, 8 * sizeof(T1), ACL_MEM_MALLOC_HUGE_FIRST));
    auto input2FreeGuard = ReleaseSource([in2]() { aclrtFree(in2); });
    T1* deviceIn2 = static_cast<T1*>(in2);
    void* out1 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&out1, 8 * sizeof(T2), ACL_MEM_MALLOC_HUGE_FIRST));
    auto output1FreeGuard = ReleaseSource([out1]() { aclrtFree(out1); });
    T2* deviceOutput1 = static_cast<T2*>(out1);
    void* out2 = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&out2, 8 * sizeof(T2), ACL_MEM_MALLOC_HUGE_FIRST));
    auto output2FreeGuard = ReleaseSource([out2]() { aclrtFree(out2); });
    T2* deviceOutput2 = static_cast<T2*>(out2);

    // --- Step 6: Host -> Device 数据拷贝 ---
    std::vector<T1> hostInput1(8 * sizeof(T1), static_cast<T1>(3));
    CHECK_ACL_RET(aclrtMemcpy(deviceIn1, 8 * sizeof(T1), hostInput1.data(), 8 * sizeof(T1), ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<T1> hostInput2(8 * sizeof(T1), static_cast<T1>(2));
    CHECK_ACL_RET(aclrtMemcpy(deviceIn2, 8 * sizeof(T1), hostInput2.data(), 8 * sizeof(T1), ACL_MEMCPY_HOST_TO_DEVICE));

    // --- Step 7: 构造 Args ---
    uint64_t shapeArray[MAX_DIM] = {8, 0, 0, 0, 0, 0, 0, 0};
    Atvoss::Tensor<T1> t1(deviceIn1, shapeArray, 1);
    Atvoss::Tensor<T1> t2(deviceIn2, shapeArray, 1);
    Atvoss::Tensor<T2> t3(deviceOutput1, shapeArray, 1);
    Atvoss::Tensor<T2> t4(deviceOutput2, shapeArray, 1);
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(t1, t2, t3, t4).build();

    // --- Step 8: 执行算子 ---
    using DeviceOp = typename ComputeTestConfig<T1, T2>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);

    // --- Step 9: 同步并拷回结果 ---
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    std::vector<T2> hostOutput1(8);
    CHECK_ACL_RET(
        aclrtMemcpy(hostOutput1.data(), 8 * sizeof(T2), deviceOutput1, 8 * sizeof(T2), ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<T2> hostOutput2(8);
    CHECK_ACL_RET(
        aclrtMemcpy(hostOutput2.data(), 8 * sizeof(T2), deviceOutput2, 8 * sizeof(T2), ACL_MEMCPY_DEVICE_TO_HOST));

    // --- Step 10: 验证结果 ---
    std::vector<T2> golden1(8, 3);
    if (!VerifyResults(golden1, hostOutput1)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
    std::vector<T2> golden2(8, 2);
    if (!VerifyResults(golden2, hostOutput2)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    Run<int32_t, int32_t>();

    return 0;
}
