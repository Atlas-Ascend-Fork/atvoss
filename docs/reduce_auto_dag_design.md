# ReduceAutoDag 设计文档

## 1. 概述

`ReduceAutoDag` 是用于处理包含 Reduce 操作的表达式 DAG（有向无环图）的核心模板结构。其主要功能是将表达式列表划分为 PreReduce 和 Reduced 两个阶段，并自动生成相应的缓冲区 ID 映射。

## 2. 设计目标

1. **阶段划分**：将表达式划分为 Reduce 前和 Reduce 后两个阶段
2. **依赖追踪**：正确识别依赖 Reduce 输出的表达式
3. **缓冲区管理**：自动分配和管理缓冲区 ID
4. **参数收集**：收集输入/输出参数信息

## 3. 核心数据结构

### 3.1 FindLocalVarReferences

**功能**：递归查找依赖 Reduce 输出的所有表达式。

#### 3.1.1 输入参数

| 参数 | 类型 | 说明 |
|-----|------|------|
| `ExprList` | `TypeList` | ReduceOp 列表（起始节点） |
| `FullList` | `TypeList` | 完整表达式列表 |
| `ExcludeList` | `TypeList` | 已处理的输出变量（Reduce 输出） |
| `ResultList` | `TypeList` | 已收集的结果表达式 |

#### 3.1.2 处理流程

```
对于每个 ReduceOp (First)：
│
├── 1. 获取 Reduce 输出
│   Output = First::LhsType
│
├── 2. 找直接引用
│   DirectReferences = FilterRefLocalVar<FullList, Output>
│   // 在 FullList 中查找 RHS 包含 Output 的表达式
│
├── 3. 更新排除列表
│   NewExclude = ExcludeList + [Output]
│   // 防止重复处理同一输出
│
├── 4. 找其他输入
│   DirectOtherInputs = FindAllUnhandledInputs<DirectReferences, FullList, NewExclude>
│   // 找 DirectReferences 表达式的未处理输入
│
├── 5. 递归处理 DirectReferences
│   RecursiveRefResult = FindLocalVarReferences<DirectReferences, FullList, NewExcludeForRecursive>
│   // 深度优先：处理引用链的子依赖
│
├── 6. 处理剩余 ReduceOps
│   NextRefResult = FindLocalVarReferences<Rest, FullList, NewExcludeWithOtherInputs>
│   // 广度优先：处理同级其他 Reduce
│
├── 7. 合并所有输入
│   CombinedOtherInputs = DirectAndRecursiveOtherInputs + NextOtherInputs
│
├── 8. 收集其他输入引用
│   OtherInputRefs = CollectOtherInputRefs<CombinedOtherInputs, FullList, ...>
│   // 处理多输入表达式的依赖
│
└── 9. 去重输出
    Type = Unique_t<OtherInputRefs>
```

#### 3.1.3 关键辅助结构

**FilterRefLocalVar**：过滤引用目标变量的表达式
```cpp
// 判断表达式 RHS 是否包含目标变量
ContainsNodeInExpr<RHS, TargetNode>::value == true
```

**FindAllUnhandledInputs**：查找表达式的未处理输入
```cpp
// 对于每个表达式，提取其 RHS 的输入
Inputs = ExtractInputs<RHS>::Type
// 过滤掉已在 ExcludeList 中的输入
FilteredInputs = FilterNotInExclude<Inputs, ExcludeList>
// 在 FullList 中找定义这些输入的表达式
FindUnhandledInputsExpr<FilteredInputs, FullList>
```

**FindUnhandledInputsExpr**：在表达式列表中找定义特定变量的表达式
```cpp
// 检查表达式的 LHS 是否为目标输入
IsLocalVarInput = Contains_v<Inputs, LHS>
// 排除已收集的表达式
NotCollected = !Contains_v<CollectedExprs, FirstExpr>
```

#### 3.1.4 递归策略

**深度优先 + 广度优先结合**：

```
ReduceOp1 ──┬── DirectRef1 ──┬── SubRef1
            │                └── SubRef2
            └── DirectRef2
ReduceOp2 ──┬── DirectRef3
            └── DirectRef4

处理顺序：
ReduceOp1 → DirectRef1 → SubRef1 → SubRef2 → DirectRef2 → ReduceOp2 → DirectRef3 → DirectRef4
```

#### 3.1.5 输出成员

| 成员 | 说明 |
|-----|------|
| `Type` | 所有依赖 Reduce 输出的表达式（去重） |
| `OtherInputs` | 所有其他输入表达式（去重） |

### 3.2 CollectOtherInputRefs

**功能**：收集并处理其他输入表达式的引用关系。

**处理流程**：
```
处理 First 节点：
├── 提取输入 Inputs
├── 过滤已排除的输入 FilteredInputs
├── 查找输入定义表达式 InputsDefExprs
├── 查找输出引用表达式 OutputRefExprs
├── 组合结果：InputsDefExprs + First + OutputRefExprs
├── 更新排除列表 NestExclude
└── 递归处理剩余节点
```

### 3.3 SortExprsByOutput

**功能**：按表达式输出变量排序。

**排序规则**：
- LocalVar 按 `number` 升序排列
- Param 输出的表达式排在最后

## 4. ReduceAutoDag 处理流程

### 4.1 表达式分类

```cpp
// Step 1: 识别 Reduce 操作
ReduceOpList = Filter_t<ContainsReduceOp, ExprList>;
ReduceOpResList = Map_t<ExtractResultType, ReduceOpList>;

// Step 2: 查找依赖 Reduce 输出的表达式
ReduceRefListRaw = FindLocalVarReferences<ReduceOpList, ExprList, ReduceOpResList>::Type;
ReduceRefList = SortExprsByOutput<ReduceRefListRaw>::Type;

// Step 3: 划分 PreReduce 和 Reduced
PreReduceWithReduceList = Difference_t<ExprList, ReduceRefList>;
ReducedWithReduceList = Concatenate_t<ReduceOpList, ReduceRefList>;
```

### 4.2 CopyIn/CopyOut 插入

```cpp
// 插入 CopyIn 操作（将 GM 数据搬运到 UB）
ReduceRefCopyInList = InsertCopyIn<ReduceRefList>::Type;
PreReduceWithReduceCopyInList = InsertCopyIn<PreReduceWithReduceList>::Type;

// 插入 CopyOut 操作（将 UB 数据搬运到 GM）
PreReduceList = InsertCopyOut<PreReduceWithReduceCopyInList>::Type;
ReducedList = InsertCopyOut<ReducedWithReduceCopyInList>::Type;
```

### 4.3 缓冲区 ID 生成

```cpp
// PreReduce 阶段缓冲区
PreReduceBufGen = BufferIdGenerator<PreReduceWoReduceCopyInList, ReduceOpList, IsBinaryAcc>;

// Reduced 阶段缓冲区  
ReducedBufGen = BufferIdGenerator<ReducedWithReduceCopyInList, ReduceOpList, false>;
```

#### 4.3.1 缓冲区类型定义

| 类型 | 标志位 | 说明 |
|-----|--------|------|
| `BUF_PARAM` | `0b001` | 参数缓冲区（GM 数据） |
| `BUF_LOCAL_VAR` | `0b010` | 局部变量缓冲区（UB 数据） |
| `BUF_REDUCE_VAR` | `0b100` | Reduce 变量缓冲区（需双缓冲） |

#### 4.3.2 缓冲区 ID 分配规则

**起始 ID 计算**：
```cpp
StartBufIdForLocalVar = IsSingleExprNoBinaryAcc ? 1 :
                        (IsBinaryAcc ? ResultParamCount + UniqueReduceInputCount_v<ReduceOpList> : ResultParamCount);
```

**分配策略**：
- **Param 缓冲区**：从 `0` 开始，按表达式顺序递增
- **Reduce 输入预留**：为 ReduceOp 的不同输入预留缓冲区（数量 = `UniqueReduceInputCount_v`）
- **LocalVar 缓冲区**：从 `StartBufIdForLocalVar` 开始分配

#### 4.3.3 GenerateBufferIdOrderAux 递归处理

**核心逻辑**：

```
处理每个表达式 OpAssign<LHS, RHS>：
├── 判断缓冲区类型
│   ├── IsLocalVarBuffer(): LHS 是否为 LocalVar
│   ├── IsReduceVarBuffer(): 是否为 Reduce 直连输出（需双缓冲）
│   └── NeedsDoubleBuffer(): Param 或 ReduceVar 需双缓冲
│
├── 分配缓冲区 ID
│   ├── NeedsDoubleBuffer(): 使用 nextParamBufId（递增）
│   └── CanReuseBuffer() && HasFreeBuffer(): 从 FreeBufferPool 取用
│   └── 否则：使用 nextLocalBufId（递增）
│
├── 更新 FreeBufferPool
│   ├── 从池中移除已分配的缓冲区
│   └── 释放不再使用的输入缓冲区（CanRelease）
│
└── 递归处理下一个表达式
```

#### 4.3.4 缓冲区复用机制

**释放条件** (`CanRelease`)：
```cpp
// 判断节点是否在位置 pos 之后被使用
IsNodeUsedAfter<ExprList, Node, pos + 1, 0>::value == false
```

**复用流程**：
1. 当前表达式处理完毕，检查输入是否可释放
2. 可释放的 LocalVar 缓冲区加入 FreeBufferPool
3. 下个表达式优先从 FreeBufferPool 分配

#### 4.3.5 Ping-Pong 缓冲区

**双缓冲需求场景**：
- Param 缓冲区（GM → UB 拷贝）
- Reduce 直连输出缓冲区（流水线并行）

**ID 计算**：
```cpp
// Ping 缓冲区 ID
bufferId = AllocBufferId();

// Pong 缓冲区 ID（仅 Param/ReduceVar）
pongBufferId = bufferId + pongOffset;  // pongOffset = maxBufferId + 1
```

#### 4.3.6 BufferIdGenerator 输出

| 成员 | 类型 | 说明 |
|-----|------|------|
| `size` | `size_t` | 缓冲区总数 |
| `maxId` | `int32_t` | 最大缓冲区 ID |
| `offset` | `uint32_t` | Pong 偏移量 |
| `tempCalcNum` | `size_t` | 临时计算缓冲区数量 |
| `BufferMapType` | `TypeList` | 变量到缓冲区的映射 |
| `GetBufferIds()` | `int32_t*[2]` | Ping/Pong 缓冲区 ID 数组 |

#### 4.3.7 数据类型统计

**追踪的最大/最小数据类型大小**：
- `InputParamMaxDTypeSize` / `InputParamMinDTypeSize`
- `OutputParamMaxDTypeSize` / `OutputParamMinDTypeSize`
- `LocalVarMaxDTypeSize` / `LocalVarMinDTypeSize`

**用途**：确定 UB 内存分配大小。

## 5. 关键算法

### 5.1 依赖追踪算法

**问题**：如何正确找到所有依赖 Reduce 输出的表达式？

**解决方案**：
1. 从 ReduceOp 的输出变量开始
2. 找到引用该变量的所有表达式
3. 递归处理这些表达式的其他输入
4. 使用 ExcludeList 避免重复处理

### 5.2 防止无限递归

**机制**：
- `ExcludeList`：记录已处理的输出变量
- `Unique_t`：结果去重
- 检查 `AlreadyProcessed`：跳过已处理节点

### 5.3 表达式排序

**目的**：确保表达式执行顺序正确（先定义后使用）。

**实现**：
- 使用插入排序算法
- LocalVar 按 number 排序保证定义顺序
- Param 放最后（作为最终输出）

## 6. 公开接口

### 6.1 阶段列表

| 接口 | 说明 |
|-----|------|
| `PreReduceList` | PreReduce 阶段表达式列表（含 CopyIn/CopyOut） |
| `ReducedList` | Reduced 阶段表达式列表（含 CopyIn/CopyOut） |

### 6.2 缓冲区信息

| 接口 | 说明 |
|-----|------|
| `PreReduceBufIds` | PreReduce 阶段缓冲区 ID 数组 |
| `ReducedBufIds` | Reduced 阶段缓冲区 ID 数组 |

### 6.3 参数统计

| 接口 | 说明 |
|-----|------|
| `GetMTE2Num<IsPre>()` | 获取 CopyIn 参数数量 |
| `GetMTE3Num<IsPre>()` | 获取 CopyOut 参数数量 |
| `GetTempBufNum<IsPre>()` | 获取临时缓冲区数量 |

### 6.4 数据类型信息

| 接口 | 说明 |
|-----|------|
| `GetInputMaxDTypeSize<IsPre>()` | 输入参数最大数据类型大小 |
| `GetInputMinDTypeSize<IsPre>()` | 输入参数最小数据类型大小 |
| `GetMaxDTypeSize<IsPre>()` | 所有参数最大数据类型大小 |
| `GetMinDTypeSize<IsPre>()` | 所有参数最小数据类型大小 |

## 7. 使用示例

```cpp
// 定义表达式列表
using ExprList = TypeList<
    OpAssign<LocalVar<1, half>, ReduceSum<Param<1, half>>>,
    OpAssign<LocalVar<2, half>, LocalVar<1, half> * LocalVar<1, half>>,
    OpAssign<Param<2, half>, Broadcast<LocalVar<2, half>>>
>;

// 生成 DAG
using Dag = ReduceAutoDag<ExprList, true>;

// 获取各阶段表达式
using PreReduce = Dag::PreReduceList;
using Reduced = Dag::ReducedList;

// 获取缓冲区 ID
constexpr auto bufIds = Dag::ReducedBufIds;
```

## 8. 复杂用例：多输入依赖场景

### 8.1 用例定义

```cpp
// 场景：RMS Norm 计算
// 1. 计算输入平方
// 2. ReduceSum 求和
// 3. Broadcast + 其他输入参与后续计算
// 4. 最终输出到 GM

using ExprList = TypeList<
    // E0: 输入平方
    OpAssign<LocalVar<1, half>, Param<1, half> * Param<1, half>>,
    
    // E1: ReduceSum（ReduceOp）
    OpAssign<LocalVar<2, half>, ReduceSum<LocalVar<1, half>>>,
    
    // E2: Broadcast Reduce 结果
    OpAssign<LocalVar<3, half>, Broadcast<LocalVar<2, half>>>,
    
    // E3: 其他输入参与计算（sqrt + divide）
    OpAssign<LocalVar<4, half>, LocalVar<3, half> + Param<2, half>>,
    
    // E4: 最终计算（使用 LocalVar<4> 和 Param<1>）
    OpAssign<LocalVar<5, half>, Param<1, half> / LocalVar<4, half>>,
    
    // E5: 输出到 GM
    OpAssign<Param<3, half>, LocalVar<5, half>>
>;
```

### 8.2 FindLocalVarReferences 处理过程

#### Step 1: 识别 ReduceOp

```
ReduceOpList = [E1]  // LocalVar<2, half> = ReduceSum<LocalVar<1, half>>
ReduceOpResList = [LocalVar<2, half>]
```

#### Step 2: 查找直接引用

```
处理 E1 (ReduceSum)：
│
├── Output = LocalVar<2, half>
│
├── DirectReferences = FilterRefLocalVar<ExprList, LocalVar<2>>
│   结果: [E2]  // E2 的 RHS (Broadcast) 包含 LocalVar<2>
│
├── NewExclude = [LocalVar<2, half>]
│
├── DirectOtherInputs = FindAllUnhandledInputs<[E2], ExprList, NewExclude>
│   E2 的 RHS = Broadcast<LocalVar<2>>
│   Inputs = [LocalVar<2>]  // 已在 ExcludeList，过滤后为空
│   结果: []
│
├── 递归处理 DirectReferences = [E2]
│   ├── Output = LocalVar<3>
│   ├── DirectReferences = [E3, E4]  // E3, E4 的 RHS 包含 LocalVar<3>
│   │   E3: LocalVar<3> + Param<2>
│   │   E4: 不包含 LocalVar<3>（实际检查）
│   │   实际结果: [E3]  // 仅 E3 包含 LocalVar<3>
│   ├── DirectOtherInputs for E3
│   │   E3 的 RHS = LocalVar<3> + Param<2>
│   │   Inputs = [LocalVar<3>, Param<2>]
│   │   FilteredInputs = [Param<2>]  // LocalVar<3> 在 ExcludeList
│   │   查找定义 Param<2> 的表达式: 无（Param 是外部输入）
│   │   结果: []
│   ├── 递归处理 [E3]
│   │   ├── Output = LocalVar<4>
│   │   ├── DirectReferences = [E4]  // E4 包含 LocalVar<4>
│   │   ├── DirectOtherInputs for E4
│   │   │   E4 的 RHS = Param<1> / LocalVar<4>
│   │   │   Inputs = [Param<1>, LocalVar<4>]
│   │   │   FilteredInputs = [Param<1>]
│   │   │   结果: []
│   │   └── 递归处理 [E4]
│   │       ├── Output = LocalVar<5>
│   │       ├── DirectReferences = [E5]
│   │       └── 最终结果收集
│   └── 合并结果
│
└── 最终 OtherInputRefs = [E2, E3, E4, E5]
```

#### Step 3: 排序结果

```
ReduceRefListRaw = [E2, E3, E4, E5]
ReduceRefList = SortExprsByOutput:
  E2: LocalVar<3> (number=3)
  E3: LocalVar<4> (number=4)
  E4: LocalVar<5> (number=5)
  E5: Param<3>    (是 Param，放最后)

结果: [E2, E3, E4, E5]  // 按 number 升序，Param 在最后
```

### 8.3 BufferIdGenerator 缓冲区分配

#### 8.3.1 Reduced 阶段表达式列表

```
ReducedWithReduceList = [E1, E2, E3, E4, E5]
```

#### 8.3.2 计算缓冲区起始 ID

```
// 提取结果参数
ResultParamCount = Params_t<Map_t<ExtractResultType, [E1..E5]>>
                 = [Param<3>]  // 仅 E5 输出到 Param
                 = 1

// Reduce 输入计数
UniqueReduceInputCount_v<[E1]>
  E1 的 RHS = ReduceSum<LocalVar<1>>
  Inputs = [LocalVar<1>]
  Unique = [LocalVar<1>]
  Count = 1

// 起始 ID
StartBufIdForLocalVar = ResultParamCount + UniqueReduceInputCount
                      = 1 + 1 = 2
```

#### 8.3.3 缓冲区分配过程

```
处理 E1: LocalVar<2> = ReduceSum<LocalVar<1>>
│
├── IsReduceVarBuffer = true  // Reduce 直连输出
├── NeedsDoubleBuffer = true
├── AllocBufferId = nextParamBufId = 0
│   BufferWrapper<0, BUF_REDUCE_VAR>
├── BufferMap: {LocalVar<2> -> BufId=0}
├── CurrentInputs = [LocalVar<1>]
├── LocalVar<1> 在 E0 定义，不在 ReducedList 中（由 PreReduce 处理）
└── nextParamBufId = 1, nextLocalBufId = 2

处理 E2: LocalVar<3> = Broadcast<LocalVar<2>>
│
├── IsLocalVarBuffer = true
├── IsReduceVarBuffer = false  // 非 Reduce 直连
├── NeedsDoubleBuffer = false
├── HasFreeBuffer = false  // FreePool 为空
├── AllocBufferId = nextLocalBufId = 2
│   BufferWrapper<2, BUF_LOCAL_VAR>
├── BufferMap: {LocalVar<2> -> 0, LocalVar<3> -> 2}
├── CurrentInputs = [LocalVar<2>]
├── LocalVar<2> 在 E1 输出，bufferId=0
├── CanRelease(LocalVar<2>)? 检查 E3/E4/E5 是否使用 → E3 使用 → 不能释放
└── nextLocalBufId = 3

处理 E3: LocalVar<4> = LocalVar<3> + Param<2>
│
├── IsLocalVarBuffer = true
├── AllocBufferId = nextLocalBufId = 3
│   BufferWrapper<3, BUF_LOCAL_VAR>
├── BufferMap: {..., LocalVar<4> -> 3}
├── CurrentInputs = [LocalVar<3>, Param<2>]
│   LocalVar<3> -> BufId=2
│   Param<2> -> 外部输入，需要分配 Param 缓冲区
├── Param<2> 分配: nextParamBufId = 1
│   BufferWrapper<1, BUF_PARAM>
├── CanRelease(LocalVar<3>)? E4 不使用 → 可释放
│   FreePool = [2]
└── nextParamBufId = 2, nextLocalBufId = 3

处理 E4: LocalVar<5> = Param<1> / LocalVar<4>
│
├── IsLocalVarBuffer = true
├── HasFreeBuffer = true (FreePool=[2])
├── AllocBufferId = Get from FreePool = 2  // 复用 E2 的缓冲区
│   BufferWrapper<2, BUF_LOCAL_VAR>
├── BufferMap: {..., LocalVar<5> -> 2}
├── CurrentInputs = [Param<1>, LocalVar<4>]
│   Param<1> 分配: nextParamBufId = 2
│   BufferWrapper<2, BUF_PARAM>
│   LocalVar<4> -> BufId=3
├── CanRelease(LocalVar<4>)? E5 不使用 → 可释放
│   FreePool = [3]
└── nextParamBufId = 3, FreePool = [3] - 2 = [3]

处理 E5: Param<3> = LocalVar<5>
│
├── IsParam = true (输出参数)
├── NeedsDoubleBuffer = true
├── AllocBufferId = nextParamBufId = 3
│   BufferWrapper<3, BUF_PARAM>
├── BufferMap: {..., Param<3> -> 3}
├── CurrentInputs = [LocalVar<5>]
│   LocalVar<5> -> BufId=2 (复用的)
└── 完成
```

#### 8.3.4 最终缓冲区映射

| 变量 | BufferId | Usage | 说明 |
|-----|----------|-------|------|
| `LocalVar<2>` | 0 | BUF_REDUCE_VAR | Reduce 输出，双缓冲 |
| `Param<2>` | 1 | BUF_PARAM | E3 输入参数 |
| `LocalVar<3>` | 2 | BUF_LOCAL_VAR | E2 输出，后被 E4 复用 |
| `LocalVar<4>` | 3 | BUF_LOCAL_VAR | E3 输出，后被 E5 输入 |
| `Param<1>` | 2 | BUF_PARAM | E4 输入参数 |
| `LocalVar<5>` | 2 | BUF_LOCAL_VAR | E4 输出，复用 BufId=2 |
| `Param<3>` | 3 | BUF_PARAM | E5 输出参数 |

#### 8.3.5 Ping-Pong 缓冲区数组

```
maxBufferId = 3
pongOffset = 4

Ping: [0, 1, 2, 3, 2, 2, 3]
Pong: [4, 5, 2, 3, 6, 2, 7]  // BUF_LOCAL_VAR 不偏移

最终 GetBufferIds():
  [Ping数组, Pong数组]
```

### 8.4 关键设计要点

**缓冲区复用**：
- LocalVar<3> (BufId=2) 使用完毕后被 LocalVar<5> 复用
- 通过 FreeBufferPool 实现缓冲区池化管理
- CanRelease 判断变量是否后续仍被使用

**多输入处理**：
- E3 有两个输入：LocalVar<3>（已分配）和 Param<2>（新分配）
- E4 有两个输入：Param<1>（新分配）和 LocalVar<4>（已分配）
- Param 缓冲区编号独立递增

**Reduce 直连双缓冲**：
- LocalVar<2> 作为 ReduceSum 直连输出，需要 BUF_REDUCE_VAR
- 用于流水线并行：上一轮 Reduce 输出同时下一轮输入计算

## 10. 相关文件

| 文件 | 说明 |
|-----|------|
| `reduce/graph/dag.h` | ReduceAutoDag 主定义 |
| `reduce/graph/helper.h` | 辅助工具（排序、收集等） |
| `reduce/graph/buffer.h` | 缓冲区 ID 生成 |
| `graph/expr_operations.h` | 表达式操作工具 |

## 11. 设计要点总结

1. **两阶段划分**：PreReduce 处理非依赖表达式，Reduced 处理 Reduce 及依赖表达式
2. **递归依赖追踪**：通过 `FindLocalVarReferences` 递归查找所有依赖链
3. **排序保证顺序**：`SortExprsByOutput` 确保表达式按定义顺序执行
4. **缓冲区自动分配**：`BufferIdGenerator` 根据 UniqueReduceInputCount 分配缓冲区
5. **防重复机制**：ExcludeList + Unique_t 防止无限递归和重复处理