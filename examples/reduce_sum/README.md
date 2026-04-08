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
| axes | 输入 | 需要reduce的轴，支持负数索引（-1表示倒数第一根轴） | int32/int64 | list |
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
output/bin/reduce_sum --shape=16,32 --axes=0 // 运行样例，沿第0轴reduce
output/bin/reduce_sum --shape=16,32,64 --axes=0,2 // 沿第0和第2轴reduce
output/bin/reduce_sum --shape=16,32 --axes=-1 // 沿倒数第一根轴reduce
```

## 空Tensor处理

本样例支持空Tensor场景的处理：

1. **非reduce轴为0**：当输入shape中存在非reduce轴的维度值为0时，输出为空Tensor
2. **仅reduce轴为0**：当输入shape中只有reduce轴的维度值为0时，输出为全1的Tensor（size为其他非reduce轴的乘积）

示例：
```bash
output/bin/reduce_sum --shape=0,32,64 --axes=1,2 // 非reduce轴为0，输出为空
output/bin/reduce_sum --shape=16,0,64 --axes=1 // 仅reduce轴为0，输出为全1的Tensor
```