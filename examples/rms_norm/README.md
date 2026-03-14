# RmsNorm算子样例
## 概述

样例概述：本样例介绍了利用ATVOSS实现RmsNorm单算子并完成功能验证
- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分
- 调用方式：Kernel直调


## 样例支持的产品
- Ascend 950PR/Ascend 950DT


## 算子描述

- 算子数学计算公式：
  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}
  $$

- 算子规格：
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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>in1</td>
      <td>输入</td>
      <td>表示进行归一化计算的输入。公式中的`x`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>in2</td>
      <td>输入</td>
      <td>表示进行归一化计算的缩放因子（权重），公式中的`g`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示进行归一化后的最终输出，公式中的`RmsNorm(x)`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
  </tbody></table>
规格说明：  

- 当前只支持二维输入
- 总的输入Shape(M, N)要满足：
    - M < 8160，N <= 7168
    - N需要32元素对齐
- Tile块的Shape(m, n)，要满足n = N，m * n <=7168
- 目前只支持float类型

## 目录结构

| 文件名                    | 描述               |
|------------------------|------------------|
| [rms_norm.cpp](./rms_norm.cpp) | RmsNorm样例算子代码实现 |
| [CMakeLists.txt](./CMakeLists.txt) | RmsNorm样例算子的编译构建文件 |
| [README.md](./README.md) | RmsNorm样例算子的说明文档 |

## RmsNorm样例算子的编译和运行
- 编译
在代码仓根目录下执行：
```bash
bash scripts/build.sh -DSOC=ascend950 rms_norm
```
- 运行
在代码仓目录下执行：
```bash
output/bin/rms_norm --help // 查看帮助
output/bin/rms_norm --shape=16,32 // 运行样例
```