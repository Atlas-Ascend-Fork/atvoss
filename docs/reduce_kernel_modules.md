# Reduce Kernel 模块设计文档

## 一、模块及依赖关系概览
### 1.1 模块概览
```shell
# Kernel 层核心模块
include/reduce/kernel/
├── builder.h             # 构建模块（协调模块）
├── schedule.h            # 调度模块
├── param_handler.h       # 参数处理模块
├── gm_offset.h           # GM 偏移计算模块
└── policy.h              # 策略配置模块

# Kernel 层依赖模块
include/reduce/utils/
└── reduce_tiling_key.h   # TilingKey 模块

# 公共模块
include/utils/
├── expr_deriver.h        # 表达式推导（公共推导模块）
└── tiling.h              # Tiling 参数计算（公共配置模块）
```
### 1.2 模块关系图
![[ATVOSS 2.0 Reduce Kernel 模块设计文档 2026-05-12 09.15.25.excalidraw]]
**依赖说明：**
- `builder.h` → `policy.h`, `schedule.h`
- `schedule.h` → `param_handler.h`, `reduce/utils/reduce_tiling_key.h`
- `param_handler.h` → `gm_offset.h`
- `gm_offset.h` → 无其他模块依赖
- `policy.h` → 无其他模块依赖
---
## 二、各模块详细说明
### 2.1 模块职责总览

| 模块                | 路径        | Namespace                            | 职责   | 关键特征                     |
| ----------------- | --------- | ------------------------------------ | ---- | ------------------------ |
| `policy.h`        | `kernel/` | `Atvoss::Reduce`                     | 策略配置 | 配置结构、策略枚举、默认策略           |
| `builder.h`       | `kernel/` | `Atvoss::Reduce`                     | 流程协调 | 串联 schedule，入口定义，OpParam |
| `param_handler.h` | `kernel/` | `Atvoss::Reduce::KernelParamHandler` | 参数处理 | 构造、转换、合并参数               |
| `gm_offset.h`     | `kernel/` | `Atvoss::Reduce::KernelGmOffset`     | 偏移计算 | 多核切分、GM 地址偏移             |
| `schedule.h`      | `kernel/` | `Atvoss::Reduce`                     | 调度执行 | Block 调度、跨核 Reduce、配置生成  |

---
### 2.2 policy.h (策略配置模块)
**模块职责：**
定义 Kernel 层的 Tiling 配置结构和多核调度策略枚举。
**核心结构体：**
#### DefaultKernelConfig
```cpp
struct DefaultKernelConfig {
    uint64_t tilingKey = 0;     // TilingKey
    uint32_t blockNum = 0;      // 核数
};
```
**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `tilingKey` | `uint64_t` | TilingKey，用于选择不同的调度策略 |
| `blockNum` | `uint32_t` | AI Core 核数 |

---
#### MultiCoreTilingPolicy
```cpp
enum class MultiCoreTilingPolicy {
    MemoryFirst = 0U,   // 带宽受限算子优先利用 UB
    ComputeFirst,       // 计算受限算子优先利用所有核
};
```
**策略说明：**

| 策略 | 适用场景 | 优化目标 |
|------|---------|---------|
| `MemoryFirst` | 带宽受限算子 | 优先利用 UB，减少 GM 访问 |
| `ComputeFirst` | 计算受限算子 | 优先利用所有核，最大化并行度 |

---
#### DefaultKernelPolicy
```cpp
struct DefaultKernelPolicy {
    MultiCoreTilingPolicy multiCoreTilingPolicy = MultiCoreTilingPolicy::MemoryFirst;
};

static constexpr DefaultKernelPolicy defaultKernelPolicy{MultiCoreTilingPolicy::MemoryFirst};
```
---
### 2.3 builder.h (构建模块)
**模块职责：**
Kernel 层的入口协调器，串联 Schedule 完成完整的 Kernel 执行流程。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `Run` | `OpParam&`, `Args...` | 无 | Kernel 层执行入口 |

**主要方法：**
#### KernelBuilder
```cpp
template <
    typename BlockOp, 
    const auto& Policy = defaultKernelPolicy, 
    typename ScheduleCfg = DefaultKernelConfig,
    template <typename, const auto&, typename> class Schedule = DefaultKernelSchedule>
class KernelBuilder {
public:
    struct OpParam {
        ScheduleCfg kernelParam;                        // Kernel 层配置
        typename BlockOp::ScheduleCfgClz blockParam;    // Block 层配置
    };
    using ScheduleClz = Schedule<BlockOp, Policy, ScheduleCfg>;
    using ScheduleCfgClz = OpParam;

    template <auto SchLoopInfo, typename OpParam, typename... Args>
    __aicore__ inline void Run(OpParam& cfg, Args... args);
};
```
**模板参数说明：**

| 参数 | 说明 |
|------|------|
| `BlockOp` | Block 操作类型（如 BlockBuilder<...>） |
| `Policy` | Kernel 策略配置（默认 MemoryFirst） |
| `ScheduleCfg` | Tiling 配置类型（默认 DefaultKernelConfig） |
| `Schedule` | Kernel 调度类型（默认 DefaultKernelSchedule） |

**执行流程：**
```
KernelBuilder::Run(cfg, args...)
    │
    └─► ScheduleClz::Run<SchLoopInfo>(cfg, args...)
            │
            ├─► [参数处理] PrepareParams → ConvertArgs
            │
            └─► [调度执行] BlockOp::Run 或 跨核 Reduce
```
---
### 2.4 param_handler.h (参数处理模块)
**模块职责：**
处理 Kernel 层参数的完整生命周期，包括参数构造（计算 GM 偏移）、转换和参数合并。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `PrepareParams` | `ScheduleCfg&`, `ArgTup&` | `tuple` | 构造偏移后的参数列表 |
| `ConstructParam` | `ScheduleCfg&`, `ArgTup&` | 单个参数 | 构造单个参数 |
| `ConvertArgs` | `ParamTup&`, `ArgTup&` | `tuple` | 合并偏移参数和原参数 |
| `ConvertOneArg` | `ParamTup&`, `ArgTup&` | 单个参数 | 转换单个参数 |

**主要方法：**
#### PrepareParams
```cpp
template <typename Params, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple);
```
- **功能**：根据 Params 类型列表，批量构造所有参数并计算 GM 偏移
- **输入**：`Params` = `TypeList<Param<1>, Param<2>, ...>`
- **输出**：`tuple<偏移in_gm, 偏移out_gm, ...>`
#### ConstructParam
```cpp
template <typename ParamType, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto ConstructParam(ScheduleCfg& cfg, ArgTup& args);
```
- **功能**：构造单个参数
- **逻辑**：
  - 若是 Tensor 类型 → 计算 GM 偏移，返回偏移后地址
  - 若是标量指针 → 解引用返回值
  - 若是标量值 → 直接返回
#### ConvertArgs
```cpp
template <typename Params, typename ParamTup, typename ArgTup>
__aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args);
```
- **功能**：合并构造后的 params 和原始 args
- **逻辑**：
  - 对于 Params 中定义的参数 → 使用 `params` 中的值（偏移后）
  - 对于额外参数 → 使用 `args` 中的原值
- **示例**：
  ```
  Params = TypeList<Param<1>, Param<2>>
  args = (in_gm, out_gm, axes_ptr, scale_value)
  params = (偏移in_gm, 偏移out_gm)
  
  ConvertArgs 结果:
  (偏移in_gm, 偏移out_gm, axes_ptr, scale_value)
  ```
---
### 2.5 gm_offset.h (GM 偏移计算模块)
**模块职责：**
根据 Block ID 和 Tiling 配置，计算当前核需要处理的 GM 数据偏移量。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `CalGMOffsetByUsage` | `ScheduleCfg&` | `uint64_t` | 根据用途计算偏移 |
| `CalGMOffsetForInput` | `ScheduleCfg&`, `blockId` | `uint64_t` | 输入偏移计算 |
| `CalGMOffsetForOutput` | `ScheduleCfg&`, `blockId` | `uint64_t` | 输出偏移计算 |

**主要方法：**
#### CalGMOffsetByUsage
```cpp
template <Atvoss::ParamUsage Usage, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetByUsage(ScheduleCfg& cfg);
```
- **功能**：根据参数用途（IN/OUT/IN_OUT）计算 GM 偏移
- **逻辑**：
  - `IN` 或 `IN_OUT` → 调用 `CalGMOffsetForInput`
  - `OUT` → 调用 `CalGMOffsetForOutput`
#### CalGMOffsetForInput
```cpp
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetForInput(ScheduleCfg& cfg, int32_t blockId);
```
- **功能**：计算输入数据的 GM 偏移量
- **典型计算**：
  ```
  offset = blockId * (totalTileACnt / blockNum) * basicBlock
  ```
#### CalGMOffsetForOutput
```cpp
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetForOutput(ScheduleCfg& cfg, int32_t blockId);
```
- **功能**：计算输出数据的 GM 偏移量
- **典型计算**：
  ```
  offset = blockId * (totalTileRCnt / blockNum) * resultBlock
  ```
---
### 2.6 schedule.h (调度模块)
**模块职责：**
Kernel 层的核心调度器，负责参数处理、Block 执行调度和跨核 Reduce 处理。
**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `MakeScheduleConfig` | `Args&`, `ScheduleCfg&` | `bool` | Host 端配置生成 |
| `Run` | `OpParam&`, `Args...` | 无 | Device 端调度执行 |
| `TilingForGroup` | `OpParam&` | 无 | 跨核 Reduce 处理 |

**主要方法：**
#### MakeScheduleConfig
```cpp
template <typename Args>
static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam);
```
- **功能**：Host 端生成 Kernel 层 Tiling 配置
- **输出**：填充 `kernelParam.blockNum`、`kernelParam.tilingKey`
#### Run
```cpp
template <auto SchLoopInfo, typename OpParam, typename... Args>
__aicore__ inline void Run(OpParam& cfg, Args... args);
```
- **功能**：Device 端调度执行
- **执行流程**：
  ```
  Run(cfg, args...)
      │
      ├─► [参数处理]
      │       params = KernelParamHandler::PrepareParams(cfg.kernelParam, argTuple)
      │       convertArgs = KernelParamHandler::ConvertArgs(params, argTuple)
      │
      ├─► [判断跨核 Reduce]
      │       if (SchLoopInfo->loopRCount == 0) → 单次 Block 执行
      │       else → 二次 Block 执行 + SyncAll + TilingForGroup
      │
      └─► [Block 执行]
              BlockOp::Run(blockConfig, convertArgs)
  ```
#### TilingForGroup
```cpp
template <typename OpParam>
__aicore__ inline auto TilingForGroup(OpParam& cfg);
```
- **功能**：跨核 Reduce 的 Tiling 处理（如 GM 地址重定向）
- **触发条件**：`SchLoopInfo->loopRCount > 0`
---
## 三、与 Device 层模块对比
### 3.1 模块对应关系

| Device 层 | Kernel 层 | 共同特征 |
|-----------|-----------|---------|
| `device_adapter.h` | `builder.h` | 入口协调，串联流程 |
| `param_handler.h` | `param_handler.h` | 参数处理（构造、转换、合并） |
| `kernel_launcher.h` | 无 | Kernel 层不负责 Host 启动 |
| 无 | `policy.h` | Kernel 策略是 Reduce 特有 |
| 无 | `gm_offset.h` | Reduce 多核切分需要偏移计算 |
| `reduce_tiling_key.h` | `reduce_tiling_key.h` | SchLoopInfo 定义（共享） |

### 3.2 层级职责差异

| 层级 | 执行环境 | 核心职责 |
|------|---------|---------|
| **Device** | Host | 表达式推导、参数准备、Tiling 计算、Kernel 启动 |
| **Kernel** | AI Core | GM 偏移计算、Block 调度、跨核 Reduce |

### 3.3 参数处理差异

| 维度 | Device 层 | Kernel 层 |
|------|-----------|-----------|
| **数据转换** | Tensor → DeviceTensor → 指针 | 指针 → 偏移后指针 |
| **偏移计算** | 无（Host 端无法获取 blockId） | 有（Device 端可获取 blockId） |
| **参数合并** | 合并 DeviceTensor 和原参数 | 合并偏移指针和原参数 |

---
## 四、完整调用链示例
### 4.1 执行流程
```
DeviceAdapter::Run(arguments, stream)
    │
    ├─► [Step 1] 表达式推导
    │
    ├─► [Step 2] 参数准备
    │
    ├─► [Step 3] Tiling 计算
    │
    └─► [Step 4] Kernel 启动
            │
            └─► KernelCustom<ReduceTilingKey, KernelBuilder<...>>(opParam, args)
                    │
                    ├─► GetSchLoopInfo<...>()
                    │
                    └─► KernelBuilder::Run<&SchLoopInfo>(opParam, args)
                            │
                            └─► KernelSchedule::Run<SchLoopInfo>(opParam, args)
                                    │
                                    ├─► PrepareParams → 计算偏移
                                    │       │
                                    │       └─► CalGMOffsetByUsage
                                    │               (CalGMOffsetForInput / CalGMOffsetForOutput)
                                    │
                                    ├─► ConvertArgs → 合并参数
                                    │
                                    └─► Block 调度
                                            │
                                            ├─► if (loopRCount == 0)
                                            │       BlockOp::Run
                                            │
                                            └─► else
                                                    BlockOp::Run (Step1)
                                                    SyncAll
                                                    TilingForGroup
                                                    BlockOp::Run (Step2)
```
### 4.2 参数流转示例
**场景**：8 核 ReduceSum，Block ID = 3
```
Device 层:
args = (in_gm=0x1000, out_gm=0x2000, axes_ptr, scale)
→ TransformArgsToPtr → (指针形式)

Kernel 层:
argTuple = (in_gm, out_gm, axes_ptr, scale)
    │
    ├─► PrepareParams
    │       in_gm 偏移 = 3 * (24/8) * 4096 * sizeof(float) = 147456 字节
    │       out_gm 偏移 = 3 * (8/8) * 1024 * sizeof(float) = 12288 字节
    │       params = (0x124000, 0x22C00)
    │
    ├─► ConvertArgs
    │       convertArgs = (0x124000, 0x22C00, axes_ptr, scale)
    │
    └─► BlockOp::Run(blockConfig, convertArgs)
            Block 层收到偏移后的 GM 地址
```
---
## 五、扩展指南
### 5.1 新增偏移策略

修改 `gm_offset.h`：
```cpp
template <typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetForInput(ScheduleCfg& cfg, int32_t blockId)
{
    // 自定义偏移计算逻辑
    return blockId * cfg.customOffset;
}
```
### 5.2 新增调度策略
继承 `BaseKernelSchedule`：
```cpp
template <typename BlockOp, const auto& Policy, typename ScheduleCfg>
class CustomKernelSchedule : public BaseKernelSchedule<BlockOp, Policy, ScheduleCfg> {
    // 自定义调度逻辑
};
```
### 5.3 新增配置参数
修改 `policy.h`：
```cpp
struct CustomKernelConfig : DefaultKernelConfig {
    uint64_t customParam = 0;  // 新增参数
};
```
---
## 六、设计原则
### 6.1 模块化原则
1. **单一职责**：每个模块只负责一个明确的功能
2. **依赖最小化**：模块间依赖清晰，避免循环依赖
3. **接口稳定**：核心接口保持稳定，内部实现可替换
### 6.2 与 Device 层保持一致

| 设计模式 | Device 层 | Kernel 层 |
|----------|-----------|-----------|
| 入口协调 | device_adapter.h | builder.h |
| 参数处理 | param_handler.h | param_handler.h |
| 策略配置 | utils/tiling.h | policy.h |
| 执行逻辑 | kernel_launcher.h | schedule.h |
