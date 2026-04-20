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

## 8. 相关文件

| 文件 | 说明 |
|-----|------|
| `reduce/graph/dag.h` | ReduceAutoDag 主定义 |
| `reduce/graph/helper.h` | 辅助工具（排序、收集等） |
| `reduce/graph/buffer.h` | 缓冲区 ID 生成 |
| `graph/expr_operations.h` | 表达式操作工具 |

## 9. 设计要点总结

1. **两阶段划分**：PreReduce 处理非依赖表达式，Reduced 处理 Reduce 及依赖表达式
2. **递归依赖追踪**：通过 `FindLocalVarReferences` 递归查找所有依赖链
3. **排序保证顺序**：`SortExprsByOutput` 确保表达式按定义顺序执行
4. **缓冲区自动分配**：`BufferIdGenerator` 根据 UniqueReduceInputCount 分配缓冲区
5. **防重复机制**：ExcludeList + Unique_t 防止无限递归和重复处理