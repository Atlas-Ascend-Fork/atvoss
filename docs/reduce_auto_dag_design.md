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

**递归逻辑**：
```
对于每个 ReduceOp：
├── 提取 LHS（Reduce 输出变量）
├── 找到引用该变量的所有表达式 DirectReferences
├── 递归处理 DirectReferences 的其他输入
├── 继续处理剩余的 ReduceOps
└── 收集所有 OtherInputRefs 并去重
```

**关键成员**：
- `DirectReferences`：直接引用 Reduce 输出的表达式
- `RecursiveResult`：递归处理的引用链
- `OtherInputRefs`：所有需要处理的其他输入表达式

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

**缓冲区 ID 分配规则**：
- Param 使用编号：0 ~ ResultParam-1
- Reduce 输入预留：ResultParam ~ ResultParam + UniqueReduceInputCount
- LocalVar 从：ResultParam + UniqueReduceInputCount 开始

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