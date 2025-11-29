<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# RmsNorm算子样例
## 概述

样例概述：本样例介绍了利用ATVOSS实现RmsNorm单算子并完成功能验证
- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分
- 调用方式：Kernel直调


## 样例支持AI处理器型号
- Ascend 910C
- Ascend 910B


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
      <td>x</td>
      <td>输入</td>
      <td>表示进行归一化计算的输入。公式中的`x`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示进行归一化计算的缩放因子（权重），公式中的`g`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行归一化后的最终输出，公式中的`RmsNorm(x)`。</td>
      <td>float</td>
      <td>ND</td>
    </tr>
  </tbody></table>
规格说明：  

- 当前只支持二维输入，
- 总的输入Shape(M, N)要满足：
    - M < 8160，N <= 7168
    - N需要32元素对齐
- Tile块的Shape(m, n)，要满足n = N，m * n <=7168
- 目前只支持float类型

## 目录结构

| 文件名                                                         | 描述                                     |
| ------------------------------------------------------------ | ------------------------------------------ |
| [rms_norm.cpp](./rms_norm.cpp) | RmsNorm算子代码实现以及调用样例               |

## 算子运行
在代码仓目录下执行：
```bash
cd ./examples
bash run_examples.sh rms_norm
```