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
├── reduce_tiling_key.h   # TilingKey 模块
└── reduce_utils.h        # 常量 / Pattern 定义

# 公共模块
include/utils/
├── expr_deriver.h        # 表达式推导（公共推导模块）
└── tiling.h              # Tiling 参数计算（公共配置模块）
```
### 1.2 模块关系图
![[ATVOSS 2.0 Reduce Kernel 模块设计文档 2026-05-12 09.15.25.excalidraw]]
**依赖说明：**
- `builder.h` → `policy.h`, `schedule.h`
- `schedule.h` → `param_handler.h`, `reduce/utils/reduce_utils.h`
- `param_handler.h` → `gm_offset.h`, `reduce/utils/reduce_utils.h`
- `gm_offset.h` → 无其他模块依赖
- `policy.h` → 无其他模块依赖

---
## 二、各模块详细说明
### 2.1 模块职责总览

| 模块 | 路径 | Namespace | 职责 | 关键特征 |
| ----------------- | --------- | ------------------------------------ | ---- | ------------------------ |
| `policy.h` | `kernel/` | `Atvoss::Reduce` | 策略配置 | 配置结构、策略枚举、默认策略 |
| `builder.h` | `kernel/` | `Atvoss::Reduce` | 流程协调 | 串联 schedule，入口定义，OpParam |
| `param_handler.h` | `kernel/` | `Atvoss::Reduce::KernelParamHandler` | 参数处理 | 构造、转换、合并参数 |
| `gm_offset.h` | `kernel/` | `Atvoss::Reduce::KernelGmOffset` | 偏移计算 | 多核切分、GM 地址偏移 |
| `schedule.h` | `kernel/` | `Atvoss::Reduce` | 调度执行 | Block 调度、跨核 Reduce、配置生成 |

---
### 2.2 policy.h (策略配置模块)
**模块职责：**
定义 Kernel 层的 Tiling 配置结构和多核调度策略枚举。
**核心结构体：**
#### DefaultKernelConfig
```cpp
struct DefaultKernelConfig {
    uint64_t tilingKey = 0;             // TilingKey
    uint32_t coreNum = 0;               // 核数
    uint64_t tileACntPerCore = 0;       // 每个核处理的 TileA 数量
    uint64_t tileRCntPerCore = 0;       // 每个核处理的 TileR 数量
};
```
**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `tilingKey` | `uint64_t` | TilingKey，用于选择不同的调度策略 |
| `coreNum` | `uint32_t` | AI Core 核数 |
| `tileACntPerCore` | `uint64_t` | 每个核处理的 A 轴 tile 数量 |
| `tileRCntPerCore` | `uint64_t` | 每个核处理的 R 轴 tile 数量 |

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
        ScheduleCfg kernelParam;                        // Kernel 层配置参数
        typename BlockOp::ScheduleCfgClz blockParam;    // Block 层配置参数
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

**Run 方法执行流程：**
```cpp
template <auto SchLoopInfo, typename OpParam, typename... Args>
__aicore__ inline void Run(OpParam& cfg, Args... args)
{
    static_assert(
        (... && (std::is_scalar_v<Args> || Util::IsTensor_v<Args>)),
        "KernelBuilder::Run only accepts scalar or Tensor types");
    ScheduleClz schedule;
    schedule.template Run<SchLoopInfo>(cfg, args...);
}
```
- 参数校验: 只接受标量或 Tensor 类型
- 委托给 `ScheduleClz::Run` 执行实际调度

---
### 2.4 param_handler.h (参数处理模块)
**模块职责：**
处理 Kernel 层参数的完整生命周期，包括参数构造（计算 GM 偏移）、转换和参数合并。
**命名空间：** `Atvoss::Reduce::KernelParamHandler`

**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `PrepareParams` | `ScheduleCfg&`, `ArgTup&` | `tuple` | 构造偏移后的参数列表 |
| `ConstructParam` | `ScheduleCfg&`, `ArgTup&` | 单个参数 | 构造单个参数 |
| `ConvertArgs` | `ParamTup&`, `ArgTup&` | `tuple` | 合并偏移参数和原参数 |
| `ConvertOneArg` | `ParamTup&`, `ArgTup&` | 单个参数 | 转换单个参数 |

**主要方法：**
#### ConstructParam
```cpp
template <typename ParamType, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto ConstructParam(ScheduleCfg& cfg, ArgTup& args)
```
- **功能**：构造单个参数
- **逻辑**：
  - 若是 Tensor 类型 → 计算 GM 偏移 (`KernelGmOffset::CalGMOffsetByUsage`)，返回偏移后 `__gm__ uint8_t*`
  - 若是标量指针 → 解引用返回值
  - 若是标量值 → 直接返回

#### PrepareParams
```cpp
template <typename Params, typename ScheduleCfg, typename ArgTup>
__aicore__ inline auto PrepareParams(ScheduleCfg& cfg, ArgTup& argTuple)
```
- **功能**：根据 Params 类型列表，批量构造所有参数并计算 GM 偏移
- **输入**：`Params` = `TypeList<Param<1>, Param<2>, ...>`
- **输出**：`tuple<偏移in_gm, 偏移out_gm, ...>`

#### ConvertOneArg
```cpp
template <typename Params, size_t Index, typename ParamTup, typename ArgTup>
__aicore__ inline constexpr auto ConvertOneArg(ParamTup& params, ArgTup& args)
```
- **功能**：转换单个参数
- **逻辑**：
  - `pos < Size_v<Params>` → 使用 `params[pos]`（偏移后）
  - 否则 → 使用 `args[Index]`（额外参数原值）

#### ConvertArgs
```cpp
template <typename Params, typename ParamTup, typename ArgTup>
__aicore__ inline auto ConvertArgs(ParamTup& params, ArgTup& args)
```
- **功能**：合并构造后的 params 和原始 args
- **示例**：
  ```
  Params = TypeList<Param<1>, Param<2>>
  args = (in_gm, out_gm, axes_ptr)
  params = (偏移in_gm, 偏移out_gm)

  ConvertArgs 结果:
  (偏移in_gm, 偏移out_gm, axes_ptr)
  ```
---
### 2.5 gm_offset.h (GM 偏移计算模块)
**模块职责：**
根据 Block ID、Tiling 配置、参数用途和 Reduce 阶段，计算当前核需要处理的 GM 数据偏移量。

**设计要点：** 偏移计算需要 **两个维度** 才能正确区分：
1. **ParamUsage** — 输入还是输出 (IN / OUT / IN_OUT)
2. **ReducePhase** — 参数出现在 reduce 前还是 reduce 后 (PreReduce / PostReduce)

原因是 DAG 分析将表达式拆为 PreReduce 和 Reduced 两阶段，不同阶段的参数 GM 数据布局不同（如 PreReduce 输入为 ARAR 交织，PostReduce 输出为 AARR 已归约），偏移公式也不同。

**命名空间：** `Atvoss::Reduce::KernelGmOffset`

**类型定义：**
#### ReducePhase
```cpp
enum class ReducePhase {
    PreReduce,   // Reduce 前: 参数出现在 PreReduceList
    PostReduce,  // Reduce 后: 参数出现在 ReducedList
};
```

**输入输出：**

| 接口 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `CalGMOffset<ParamType>` | `ScheduleCfg&` | `uint64_t` | 统一入口：自动推导 Phase 并计算偏移 |
| `CalGMOffsetImpl<Usage, Phase>` | `ScheduleCfg&` | `uint64_t` | 底层分发：根据 Usage + Phase 选择偏移公式 |
| `CalPreReduceInputOffset` | `ScheduleCfg&` | `uint64_t` | PreReduce 阶段 + IN 参数的偏移 |
| `CalPostReduceInputOffset` | `ScheduleCfg&` | `uint64_t` | PostReduce 阶段 + IN 参数的偏移 |
| `CalPreReduceOutputOffset` | `ScheduleCfg&` | `uint64_t` | PreReduce 阶段 + OUT 参数的偏移 |
| `CalPostReduceOutputOffset` | `ScheduleCfg&` | `uint64_t` | PostReduce 阶段 + OUT 参数的偏移 |

**场景矩阵：**

| | PreReduce | PostReduce |
|--|-----------|------------|
| **IN** | `CalPreReduceInputOffset` | `CalPostReduceInputOffset` |
| **OUT** | `CalPreReduceOutputOffset` | `CalPostReduceOutputOffset` |
| **IN_OUT** | `CalPreReduceInputOffset` | `CalPostReduceOutputOffset` |

**示例：** 某算子有 3 输入 2 输出：
```
PreReduce:  x0(IN,ARAR), x1(IN,ARAR), y0(OUT,ARAR)
PostReduce: x2(IN,AARR), y1(OUT,AARR)

x0: CalGMOffsetImpl<IN,  PreReduce>  → CalPreReduceInputOffset   ✓
x1: CalGMOffsetImpl<IN,  PreReduce>  → CalPreReduceInputOffset   ✓
x2: CalGMOffsetImpl<IN,  PostReduce> → CalPostReduceInputOffset  ✓
y0: CalGMOffsetImpl<OUT, PreReduce>  → CalPreReduceOutputOffset  ✓
y1: CalGMOffsetImpl<OUT, PostReduce> → CalPostReduceOutputOffset ✓
```
**主要方法：**
#### CalGMOffset (统一入口)
```cpp
template <typename ParamType, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffset(ScheduleCfg& cfg)
```
- **功能**：`param_handler` 调用的唯一入口，根据 ParamType 自动推导 Phase
- **Phase 默认映射**：`IN/IN_OUT → PreReduce`，`OUT → PostReduce`
- **设计意图**：Phase 推导逻辑封装在 `gm_offset.h` 内部，后续 DAG 阶段信息若传入 ParamType，只需修改此函数

#### CalGMOffsetImpl (底层分发)
```cpp
template <Atvoss::ParamUsage Usage, ReducePhase Phase, typename ScheduleCfg>
__aicore__ inline uint64_t CalGMOffsetImpl(ScheduleCfg& cfg)
```
- **功能**：根据 Usage + Phase 分发到正确的偏移公式
- **逻辑**：
  - `IN, PreReduce`    → `CalPreReduceInputOffset`
  - `IN, PostReduce`   → `CalPostReduceInputOffset`
  - `OUT, PreReduce`   → `CalPreReduceOutputOffset`
  - `OUT, PostReduce`  → `CalPostReduceOutputOffset`
  - `IN_OUT, PreReduce`  → `CalPreReduceInputOffset` (按输入算)
  - `IN_OUT, PostReduce` → `CalPostReduceOutputOffset` (按输出算)
- **Block ID 获取**：内部通过 `AscendC::GetBlockIdx()` 获取


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
static bool MakeScheduleConfig(const Args& arguments, ScheduleCfg& kernelParam)
```
- **功能**：Host 端生成 Kernel 层 Tiling 配置
- **当前实现**：硬编码 `coreNum = 49`, `tilingKey = 2`（后续由实际 tiling 计算覆盖）
- **输出**：填充 `kernelParam.coreNum`、`kernelParam.tilingKey`

#### Run
```cpp
template <auto SchLoopInfo, typename OpParam, typename... Args>
__aicore__ inline void Run(OpParam& cfg, Args... args)
```
- **功能**：Device 端调度执行
- **执行流程**：
  ```
  Run(cfg, args...)
      │
      ├─► [提取 blockConfig]
      │       blockConfig = cfg.blockParam
      │
      ├─► [参数处理]
      │       argTuple = forward_as_tuple(args...)
      │       params = KernelParamHandler::PrepareParams(cfg.kernelParam, argTuple)
      │       convertArgs = KernelParamHandler::ConvertArgs(params, argTuple)
      │
      ├─► [判断跨核 Reduce]
      │       if (loopRCount == 0) → BlockOp::Run(blockConfig, convertArgs)  // 单次执行
      │       else → BlockOp::Run (Step1) → SyncAll → TilingForGroup → BlockOp::Run (Step2)
      │
      └─► [Block 执行]
              BlockOp::Run(blockConfig, convertArgs)
  ```

#### TilingForGroup
```cpp
template <typename OpParam>
__aicore__ inline auto TilingForGroup(OpParam& cfg)
```
- **功能**：跨核 Reduce 的 Tiling 处理（如 GM 地址重定向）
- **触发条件**：`SchLoopInfo->loopRCount > 0`
- **当前状态**：空实现，待后续完成

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
    │       │
    │       ├─► KernelOp::ScheduleClz::MakeScheduleConfig → kernelParam (coreNum=49, tilingKey=2)
    │       └─► BlockOp::ScheduleClz::MakeScheduleConfig → blockParam
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
                                    ├─► blockConfig = cfg.blockParam
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
                                            │       BlockOp::Run(blockConfig, convertArgs)
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
args = (in_gm=0x1000, out_gm=0x2000, axes_ptr)
→ TransformArgsToPtr → (指针形式)

Kernel 层:
argTuple = (in_gm, out_gm, axes_ptr)
    │
    ├─► PrepareParams
    │       in_gm 偏移 = 3 * (24/8) * 4096 * sizeof(float) = 147456 字节
    │       out_gm 偏移 = 3 * (8/8) * 1024 * sizeof(float) = 12288 字节
    │       params = (0x124000, 0x22C00)
    │
    ├─► ConvertArgs
    │       convertArgs = (0x124000, 0x22C00, axes_ptr)
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
__aicore__ inline uint64_t CalGMOffsetForInput(ScheduleCfg& cfg)
{
    int32_t blockId = AscendC::GetBlockIdx();
    // 自定义偏移计算逻辑
    return blockId * cfg.tileACntPerCore * cfg.basicBlock;
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
