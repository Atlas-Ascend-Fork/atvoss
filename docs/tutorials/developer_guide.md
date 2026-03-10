# 算子开发指南

本指南以 `muls`（矩阵标量乘法）算子为例，详细介绍在 Atvoss 框架中开发自定义算子的完整流程。通过该示例，您将学会如何实现一个二元运算算子。

## 1. 算子需求

### 1.1 功能描述

`muls` 算子实现**矩阵标量乘法**功能，将输入矩阵的每个元素与一个标量值相乘。

数学表达式：
$$
Y = X \times scalar
$$

其中：
- $X$：输入矩阵，任意维度
- $scalar$：标量乘数
- $Y$：输出矩阵，与输入矩阵形状相同

### 1.2 支持的数据类型

| x输入类型 | scalar输入类型| 输出类型 | 
|---------|---------|------|
| float | float| float | 
| half | half | half | 

## 2. 设计规格

### 2.1 核心组件

| 组件 | 类型 | 说明 |
|------|------|------|
| `MulsCompute` | Struct | 计算逻辑 |
| `TileShape` | Shape<32> | 分块大小 |
| `blockPolicy` | DefaultBlockPolicy | 默认分块策略 |
| `kernelPolicy` | DefaultKernelPolicy | 默认分核策略 |

### 2.2 表达式定义

```cpp
out = in * scalar
```

## 3. 编写核心代码逻辑

### 3.1 导入头文件
```cpp
#include "kernel_operator.h"    // 必须，AscendC 底层 API 定义
#include "atvoss.h"             // 必须，Atvoss 框架头文件
#include "example_common.h"     // 可选，示例通用工具，ACL 初始化、内存管理等通用代码
```

### 3.2 配置结构体定义

建议所有算子定义相关的配置都定义在一个Struct中，便于后续实例化使用。
```cpp
// 定义TileShape参数配置
static constexpr int32_t WIDTH = 32;

// 算子配置结构体
template <typename TensorDtype, typename ScalarDtype>
struct MulsConfig {
    using TileShape = Atvoss::Shape<WIDTH>;
    
    // ...
};
```

#### 3.2.1 Compute 结构体
在MulsConfig中增加Compute定义
`Compute()` 接口必须符合以下固定格式：
```cpp
struct MulsCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        // 1. 定义输入输出参数描述
        auto in = Atvoss::PlaceHolder<1, Tensor<TensorDtype>, Atvoss::ParamUsage::IN>();
        auto scalar = Atvoss::PlaceHolder<2, ScalarDtype, Atvoss::ParamUsage::IN>();
        auto out = Atvoss::PlaceHolder<3, Tensor<TensorDtype>, Atvoss::ParamUsage::OUT>();
        // 2. 返回计算表达式
        return (out = in * scalar);
    };
};
```

**接口说明**：
- 先定义输入输出参数
- 再定义基于输入输出的计算表达式
- 使用逗号分隔多个计算步骤

更多运算API接口参考[API](../api/README.md#table_Operator)

### 3.2.2 策略配置
在MulsConfig中增加策略配置
```cpp
// 块级策略：指定分块形状
static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
// 核策略：使用均匀分段
static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};
// 指定目标架构
using ArchTag = Atvoss::Arch::DAV_3510;
```

### 3.2.3 算子Op定义
在MulsConfig中增加算子Op定义
```cpp
// 定义BlockOp
using BlockOp = Atvoss::Ele::BlockBuilder<
    MulsCompute,
    ArchTag,
    blockPolicy,
    Atvoss::Ele::DefaultBlockConfig,
    Atvoss::Ele::DefaultBlockSchedule>;
// 定义KernelOp
using KernelOp = Atvoss::Ele::KernelBuilder<
    BlockOp,
    kernelPolicy,
    Atvoss::Ele::DefaultKernelConfig,
    Atvoss::Ele::DefaultKernelSchedule>;
// 定义DeviceOp
using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
```

### 3.3 运行时执行逻辑

```cpp
template <typename TensorDtype, typename ScalarDtype>
static void ProcessMuls(std::vector<TensorDtype>& hostInput, ScalarDtype scalar, std::vector<TensorDtype>& hostOutput, const std::vector<int>& shape)
{
    // Step 1: ACL 初始化
    CHECK_ACL_RET(aclInit(nullptr));
    auto finalizeGuard = ReleaseSource([]() { aclFinalize(); });

    // Step 2: 设置 device
    const int32_t deviceId = 0;
    CHECK_ACL_RET(aclrtSetDevice(deviceId));
    auto deviceResetGuard = ReleaseSource([deviceId]() { aclrtResetDevice(deviceId); });

    // Step 3: 创建 Context
    aclrtContext context = nullptr;
    CHECK_ACL_RET(aclrtCreateContext(&context, deviceId));
    auto contextDestroyGuard = ReleaseSource([context]() { aclrtDestroyContext(context); });

    // Step 4: 创建 Stream
    aclrtStream stream = nullptr;
    CHECK_ACL_RET(aclrtCreateStream(&stream));
    auto streamDestroyGuard = ReleaseSource([stream]() { aclrtDestroyStream(stream); });

    // Step 5: 准备数据
    const size_t shapeSize = std::accumulate(shape.begin(), shape.end(), 
                                             size_t{1}, std::multiplies<>{});
    const size_t inputSize = shapeSize * sizeof(TensorDtype);
    const size_t outputSize = inputSize;

    // Step 6: 分配设备内存
    void* rawInput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawInput, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto inputFreeGuard = ReleaseSource([rawInput]() { aclrtFree(rawInput); });
    TensorDtype* deviceInput = static_cast<TensorDtype*>(rawInput);

    void* rawOutput = nullptr;
    CHECK_ACL_RET(aclrtMalloc(&rawOutput, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto outputFreeGuard = ReleaseSource([rawOutput]() { aclrtFree(rawOutput); });
    ScalarDtype* deviceOutput = static_cast<ScalarDtype*>(rawOutput);
    
    // Step 7: Host → Device 数据拷贝
    CHECK_ACL_RET(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Step 8: 构造 Tensor 并执行算子
    uint32_t shapeArray[8] = {0};
    std::copy(shape.begin(), shape.end(), shapeArray);
    Atvoss::Tensor<TensorDtype> in(deviceInput, shapeArray, shape.size());
    Atvoss::Tensor<ScalarDtype> out(deviceOutput, shapeArray, shape.size());
    
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in, scalar, out).build();

    // 根据输入类型选择算子
    if constexpr (std::is_same_v<TensorDtype, ScalarDtype>) {
        using DeviceOp = typename MulsConfig<TensorDtype, ScalarDtype>::DeviceOp;
        DeviceOp deviceOp;
        deviceOp.Run(arguments, stream);
    } else {
        static_assert("The data types of x and scalar must be the same.");
    }

    // Step 9: 同步并拷回结果
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));
    CHECK_ACL_RET(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
}
```

### 3.4 主函数

```cpp
int main(int argc, char const* argv[])
{
    std::cout << "Start muls" << std::endl;
    
    // 定义输入数据
    const std::vector<int> shape = {32, 32};
    std::vector<float> hostInput(shape[0] * shape[1], 3.0f);
    std::vector<float> hostOutput(shape[0] * shape[1]);
    float scalar = 3.0f;
    
    // 调用模板函数
    ProcessMuls<float, float>(hostInput, scalar, hostOutput, shape);

    // 验证结果
    std::vector<float> golden(shape[0] * shape[1], 9.0f);
    if (!VerifyResults(golden, hostOutput)) {
        std::cout << "Accuracy verification failed." << std::endl;
    } else {
        std::cout << "Accuracy verification passed." << std::endl;
    }
    return 0;
}
```

## 4. 编写 CMake 文件

**文件位置**：`CMakeLists.txt`

```cmake

cmake_minimum_required(VERSION 3.16)
project(test_muls)
# 这部分需要根据用户自己环境的路径进行配置
set(ATVOSS_INCLUDE_DIRS /mnt/workspace/gitCode/cann/atvoss/include)
set(ASCEND_DIR /home/developer/Ascend/cann-9.0.0)
set(ASCEND_INCLUDE_DIRS ${ASCEND_DIR}/include)
set(ATVOSS_EXAMPLES_COMMON_SOURCE_DIR /mnt/workspace/gitCode/cann/atvoss/examples/common)

set(NPU_ARCH dav-3510)

set(INCLUDE_DIRECTORIES
    ${ASCEND_INCLUDE_DIRS}
    ${ATVOSS_INCLUDE_DIRS}
    ${ATVOSS_EXAMPLES_COMMON_SOURCE_DIR}
)

set(LINK_DIRECTORIES
    ${ASCEND_DIR}/lib64
)

set(LINK_LIBRARIES
    ascendcl
    platform
    register
    tiling_api
    runtime
)

set(COMPILE_OPTIONS
    -O3
    -fdiagnostics-color=always
    -w
)

set(BISHENG ${ASCEND_DIR}/bin/bisheng)

unset(CMAKE_CXX_FLAGS)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

set(COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc ")

set_source_files_properties(
    test_muls.cpp PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "${COMPILE_FLAGS}"
)

add_executable(test_muls test_muls.cpp)

set_target_properties(test_muls PROPERTIES
    LINK_DEPENDS_NO_SHARED ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(test_muls PRIVATE ${COMPILE_OPTIONS})
target_include_directories(test_muls PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(test_muls PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(test_muls PRIVATE ${LINK_LIBRARIES})
```

## 5. 编译与执行

### 5.1 环境准备

确保以下环境变量已设置：

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
source ${install_path}/cann/set_env.sh
```

### 5.2 编译命令

```bash
# 编译 muls 算子
mkdir build && cd build && cmake .. && make
```

### 5.3 编译输出

成功编译后，可执行文件位于：
```
build/test_muls
```

### 5.4 执行命令

```bash
# 查看帮助
./build/test_muls
```

## 6. 查看输出结果

### 6.1 正常输出

```
Start muls
Accuracy verification passed.
```

## 7. 参考文档
ATVOSS接口参考[API](../api/README.md#table_ATVOSS)
运算API接口参考[API](../api/README.md#table_Operator)
