# Reduce Device 模块设计文档
## 一、模块及依赖关系概览
### 1.1 模块概览
```shell
# 核心模块
include/reduce/
├── device/
│   ├── device_adapter.h         # 协调模块
│   ├── param_handler.h          # 参数处理模块
│   └── kernel_launcher.h        # Kernel 启动模块
├── utils/
    └── reduce_tiling_key.h      # TilingKey 模块

# 公共模块
include/utils/
├── tensor/
│   └── device_tensor.h      # Device端 Tensor 封装（公共基础模块）
├── expr_deriver.h           # 表达式推导（公共推导模块）
└── tiling.h                 # Tiling 参数计算（公共配置模块）
```
### 1.2 模块关系图
![[ATVOSS 2.0 Reduce Device 模块设计文档 2026-04-29 14.58.53.excalidraw]]
**依赖说明：**
- `device_adapter.h` → `utils/expr_deriver.h`, `utils/tiling.h`, `param_handler.h`, `kernel_launcher.h`
- `param_handler.h` → `utils/tensor/device_tensor.h`
- `kernel_launcher.h` → `param_handler.h`, `reduce/utils/reduce_tiling_key.h`
- `utils/expr_deriver.h` → `graph/expr_linearizer.h`
- `utils/tiling.h` → 无其他模块依赖
- `utils/device_tensor.h` → 无其他模块依赖
---
## 二、各模块详细说明
### 2.1 模块职责总览

| 模块                    | 路径               | Namespace        | 职责          | 关键特征                       |
| --------------------- | ---------------- | ---------------- | ----------- | -------------------------- |
| `device_tensor.h`     | `utils/tensor/`  | `Atvoss`         | 数据封装        | 公共模块，轻量级指针包装，类型标记          |
| `expr_deriver.h`      | `utils/`         | `Atvoss`         | 表达式推导       | 公共模块，编译时类型推导               |
| `tiling.h`            | `utils/`         | `Atvoss`         | 配置计算        | 公共模块，调用 MakeScheduleConfig |
| `param_handler.h`     | `reduce/device/` | `Atvoss::Reduce` | 参数处理        | 构造、转换、合并参数                 |
| `kernel_launcher.h`   | `reduce/device/` | `Atvoss::Reduce` | Kernel启动    | Host 入口，AI Core 执行         |
| `reduce_tiling_key.h` | `reduce/utils/`  | `Atvoss::Reduce` | TilingKey定义 | TilingKey定义、SchLoopInfo定义  |
| `device_adapter.h`    | `reduce/device/` | `Atvoss::Reduce` | 流程协调        | 串联各模块，对外统一接口               |

### 2.2 utils/tensor/device_tensor.h (基础模块)
**模块职责：**
提供 Device 端 Tensor 的轻量级封装，用于 Host 端参数转换和 Kernel 启动前的数据准备。
该模块为公共模块，elewise 和 reduce 共享使用。
**输入输出：**

| 接口           | 输入                   | 输出                | 说明               |
| ------------ | -------------------- | ----------------- | ---------------- |
| 构造函数         | `Atvoss::Tensor<T>&` | `DeviceTensor<T>` | 从 Host Tensor 构造 |
| `GetPtr()`   | 无                    | `T*`              | 获取原始数据指针         |
| `operator[]` | `std::size_t pos`    | `T&`              | 元素访问             |
| `Clear()`    | 无                    | 无                 | 释放内存             |

**主要方法：**
```cpp
namespace Atvoss {

template <typename T>
class DeviceTensor {
    // 构造：从 Atvoss::Tensor 提取数据指针
    explicit DeviceTensor(Atvoss::Tensor<T>& src);
    
    // 获取指针：供 Kernel 启动时传递 GM 地址
    auto GetPtr() const;
    
    // 类型标记：用于模板元编程判断是否为 Tensor 类型
    using IsTensor = void;
};

} // namespace Atvoss
```
---
### 2.3 utils/tiling.h (配置模块)
**模块职责：**
计算 Kernel 和 Block 层级的 Tiling 配置参数，确定多核划分和数据分块策略。
该模块为公共模块，elewise 和 reduce 共享使用。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `CalculateTiling` | `Args` (用户参数), `OpParam&` | `bool` | 填充 `kernelParam` 和 `blockParam` |

**主要方法：**
```cpp
namespace Atvoss {

template <typename KernelOp, typename Args>
bool CalculateTiling(const Args& args, typename KernelOp::ScheduleCfgClz& cfg);

} // namespace Atvoss
```
**执行流程：**
```
CalculateTiling(args, cfg)
    │
    ├─► KernelOp::ScheduleClz::MakeScheduleConfig(args, cfg.kernelParam)
    │       计算核数、TileA/TileR 数量等
    │
    └─► BlockOp::ScheduleClz::MakeScheduleConfig(args, cfg.kernelParam, cfg.blockParam)
            计算 UB 分块因子、basicBlock 等
```
---
### 2.4 param_handler.h (参数模块)
**模块职责：**
处理 Reduce 算子参数的完整生命周期，包括参数构造、类型转换和参数合并。
**输入输出：**

| 接口                   | 输入                       | 输出                    | 说明     |
| -------------------- | ------------------------ | --------------------- | ------ |
| `TransformArgToPtr`  | `T` (scalar 或 Tensor)    | 指针或原值                 | 单参数转换  |
| `TransformArgsToPtr` | `ArgTuple`               | `AscendC::Std::tuple` | 参数组转换  |
| `ConstructParam`     | `ArgTuple&`, `ParamType` | `DeviceTensor` 或标量    | 构造单个参数 |
| `PrepareParams`      | `ArgTuple&`, `Params`    | `std::tuple`          | 构造所有参数 |
| `ConvertArgs`        | `ParamTuple`, `ArgTuple` | `std::tuple`          | 合并参数   |

**主要方法：**
#### TransformArgToPtr
```cpp
template <typename T>
auto TransformArgToPtr(T&& value);
```
- **功能**：将 `DeviceTensor` 转换为指针，标量保持原值
- **用途**：Kernel 启动前，将 DeviceTensor 转为 GM 地址指针
#### TransformArgsToPtr
```cpp
template <typename ArgTup>
auto TransformArgsToPtr(const ArgTup& argTuple);
```
- **功能**：批量转换参数组中所有 DeviceTensor 为指针
- **用途**：供 `kernel_launcher` 调用
#### ConstructParam
```cpp
template <typename ParamType, typename ArgTup, template <typename> class TensorT>
constexpr auto ConstructParam(ArgTup& args);
```
- **功能**：根据 ParamType 从 args 中提取并构造参数
- **逻辑**：
  - 若是 Tensor 类型 + 标量参数 → 构造 `TensorT<T>(args[n-1])`
  - 否则 → 直接返回 `args[n-1]`
#### PrepareParams
```cpp
template <typename Params, template <typename> class TensorT, typename ArgTup>
constexpr auto PrepareParams(ArgTup& argTuple);
```
- **功能**：根据 Params 类型列表，批量构造所有参数
- **输入**：`Params` = `TypeList<Param<1>, Param<2>, ...>`
- **输出**：`std::tuple<DeviceTensor<T1>, DeviceTensor<T2>, ...>`
#### ConvertArgs
```cpp
template <typename Params, typename ParamTup, typename ArgTup>
auto ConvertArgs(ParamTup& params, ArgTup& args);
```
- **功能**：合并构造后的 params 和原始 args
- **逻辑**：
  - 对于 Params 中定义的参数 → 使用 `params` 中的值
  - 对于额外参数 → 使用 `args` 中的原值
- **示例**：
  ```
  Params = TypeList<Param<1>, Param<2>>
  args = (in_tensor, out_tensor, axes_ptr, scale_value)
  params = (DeviceTensor(in), DeviceTensor(out))
  
  ConvertArgs 结果:
  (DeviceTensor(in), DeviceTensor(out), axes_ptr, scale_value)
  ```
---
### 2.5 kernel_launcher.h (执行模块)
**模块职责：**
封装 Reduce 算子 Kernel 的启动逻辑，包括参数转换、Kernel 入口函数和实际启动调用。
**输入输出：**

| 接口              | 输入                                          | 输出  | 说明                 |
| --------------- | ------------------------------------------- | --- | ------------------ |
| `KernelWrapper` | `OpParam&`, `ArgTuple`                      | 无   | Kernel 内部执行包装，参数展开 |
| `KernelCustom`  | `OpParam`, `ArgTuple`                       | 无   | Kernel 入口函数        |
| `LaunchKernel`  | `blockNum`, `stream`, `OpParam`, `ArgTuple` | 无   | Host侧启动 Kernel入口   |

**主要方法：**
#### KernelWrapper
```cpp
template <class KernelOp, typename OpParam, typename ArgTuple, std::size_t... Is>
__aicore__ inline void KernelWrapper(OpParam& cfg, ArgTuple args, AscendC::Std::index_sequence<Is...>);
```
- **功能**：展开参数 tuple，调用 `KernelOp::Run`
#### KernelCustom
```cpp
template <class KernelOp, typename OpParam, typename ArgTuple>
__global__ __aicore__ void KernelCustom(OpParam cfg, ArgTuple args);
```
- **功能**：Kernel 入口点，设置任务类型，调用 `KernelWrapper`
#### LaunchKernel
```cpp
template <class KernelOp, typename OpParam, typename ArgTup>
void LaunchKernel(uint32_t blockNum, aclrtStream stream, OpParam& cfg, const ArgTup& argTuple);
```
- **功能**：Host 端启动 Kernel
---
### 2.6 device_adapter.h (协调模块)
**模块职责：**
Reduce 算子的顶层协调器，串联各个模块完成完整的 Device 执行流程。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `Run` | `Args` (用户完整参数), `aclrtStream` | `int64_t` | 执行结果码 |


**主要方法：**
#### Run
```cpp
template <typename Args>
int64_t Run(const Args& arguments, aclrtStream stream = nullptr);
```

**完整执行流程：**

```
DeviceAdapter::Run(arguments, stream)
    │
    ├─► [Step 1] 表达式推导 (utils/expr_deriver.h)
    │       using Params = ParamsDeriver<ExprMaker, Tensor>::Params
    │       // 使用公共模块，一行完成推导
    │
    ├─► [Step 2] 参数准备 (param_handler)
    │       params = PrepareParams<Params, Tensor>(argTuple)
    │
    ├─► [Step 3] 参数合并 (param_handler)
    │       convertArgs = ConvertArgs<Params>(params, argTuple)
    |
    ├─► [Step 4] Tiling 计算 (utils/tiling.h)
    │       CalculateTiling<KernelOp>(arguments, opParam)
    │
    └─► [Step 5] Kernel 启动 (kernel_launcher)
            LaunchKernel<ReduceTilingKey<0, 0, 0>, KernelOp>(blockNum, stream, opParam, convertArgs)
```
---

### 2.7 reduce_tiling_key.h (TilingKey 模块)

**模块职责：**
定义 Reduce 算子的 TilingKey 配置和调度循环信息（SchLoopInfo），用于 Kernel 启动时的策略选择和循环参数传递。

**核心结构体：**

#### ReduceTilingKey
```cpp
template<uint32_t PatternID = 0, uint32_t LoopARCount = 0, uint32_t LoopInnerARCount = 0>
struct ReduceTilingKey {
    static constexpr uint32_t PATTERN_ID = PatternID;
    static constexpr uint32_t LOOP_AR_COUNT = LoopARCount;
    static constexpr uint32_t LOOP_INNER_AR_COUNT = LoopInnerARCount;
};
```

**参数说明：**

| 参数 | 含义 | 计算方式 |
|------|------|----------|
| `PatternID` | Reduce 模式标识 | `Pattern * 10 + InnerPattern` |
| `LoopARCount` | 核间 A/R 轴循环次数 | `LoopACount * 10 + LoopRCount` |
| `LoopInnerARCount` | 核内 A/R 轴循环次数 | `LoopInnerACount * 10 + LoopInnerRCount` |

**用途：**
- 作为 `LaunchKernel` 的模板参数传递给 `KernelCustom`
- 在 Kernel 内部通过静态常量成员访问配置值

---

#### ReduceSchLoopInfo
```cpp
struct ReduceSchLoopInfo {
    int32_t patternID;              // 输入 Reduce 模式 ID
    int32_t innerPatternID;         // 二分 Reduce 模式 ID
    int32_t loopACount;             // 核间 A 轴循环次数
    int32_t loopAAxis[MAX_LOOP_DIM]; // 核间 A 轴维度列表
    int32_t loopRCount;             // 核间 R 轴循环次数
    int32_t loopRAxis[MAX_DIM];     // 核间 R 轴维度列表

    int32_t loopInnerACount;        // 核内 A 轴循环次数
    int32_t loopInnerAAxis[MAX_DIM]; // 核内 A 轴维度列表
    int32_t loopInnerRCount;        // 核内 R 轴循环次数
    int32_t loopInnerRAxis[MAX_DIM]; // 核内 R 轴维度列表
};
```

**用途：**
- 作为非类型模板参数传递给 `KernelWrapper` 和后续调度层
- 在 `schedule.h` 中通过 `if constexpr (SchLoopInfo->loopRCount == 0)` 判断是否需要跨核 Reduce

---

#### GetSchLoopInfo 系列函数

```cpp
// 主函数：根据 LoopARCount 选择对应的配置函数
template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo();

// 配置函数 0-5：对应不同的 loopACount（0~5）
template <uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount>
__aicore__ inline constexpr ReduceSchLoopInfo GetSchLoopInfo0();
// ... GetSchLoopInfo1 ~ GetSchLoopInfo5

// Group 特殊配置
__aicore__ inline constexpr ReduceSchLoopInfo GetGroupSchLoopInfo();
```
---
**关键点：**
- `ReduceTilingKey` 是编译时类型，携带静态常量配置
- `SchLoopInfo` 是 `constexpr` 变量，通过 `GetSchLoopInfo` 函数生成
- `SchLoopInfo` 作为指针类型模板参数传递（`&SchLoopInfo`），在后续层使用 `->` 访问成员

---