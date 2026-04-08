# ReduceSum算子样例
## 概述

本样例介绍了利用ATVOSS的Reduce模板实现ReduceSum算子，展示如何使用Reduce表达式和调度策略。

## 算子描述

- 算子数学计算公式：
  $$
  \operatorname{ReduceSum}(in, axis) = \sum_{i} in[i]
  $$

- 算子参数：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| in | 输入 | 输入张量 | float | ND |
| out | 输出 | 沿指定轴求和后的输出张量 | float | ND |

## 样例支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构

| 文件名 | 描述 |
|--------|------|
| [reduce_sum.cpp](./reduce_sum.cpp) | ReduceSum样例算子代码实现 |
| [CMakeLists.txt](./CMakeLists.txt) | ReduceSum样例算子的编译构建文件 |
| [README.md](./README.md) | ReduceSum样例算子的说明文档 |

## ReduceSum样例算子的编译和运行
- 编译
在代码仓根目录下执行：
```bash
bash scripts/build.sh -DSOC=ascend950 reduce_sum
```
- 运行
在代码仓目录下执行：
```bash
output/bin/reduce_sum --help // 查看帮助
output/bin/reduce_sum --shape=16,32 // 运行样例
```