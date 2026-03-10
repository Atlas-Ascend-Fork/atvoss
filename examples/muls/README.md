# Muls算子样例
## 概述

样例概述：本样例介绍了利用ATVOSS实现Muls单算子并完成功能验证，重点展示了根据用户不同输入信息，选择不同Compute表达的运算过程
- Muls算子功能：实现一个Tensor和一个Scalar进行乘法运算

## 算子描述

- 算子数学计算公式：
  $$
  \operatorname{Muls}(in, scalar)= {in * scalar}
  $$

- 算子参数：

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>in</td>
      <td>输入</td>
      <td>公式中的Tensor输入。</td>
      <td>float、int32_t</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>公式中的Scalar输入。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>执行Muls运算后的输出Tensor。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 样例支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构

| 文件名                    | 描述               |
|------------------------|------------------|
| [muls.cpp](./muls.cpp) | Muls样例算子代码实现 |
| [CMakeLists.txt](./CMakeLists.txt) | Muls样例算子的编译构建文件 |
| [README.md](./README.md) | Muls样例算子的说明文档 |

## Muls样例算子的编译和运行
- 编译
在代码仓根目录下执行：
```bash
bash scripts/build.sh -DSOC=ascend950 muls
```
- 运行
在代码仓目录下执行：
```bash
output/bin/muls --help // 查看帮助
output/bin/muls --shape=16,32 // 运行样例
```