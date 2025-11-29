<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->
# Cast算子样例
## 概述

样例概述：本样例介绍了利用ATVOSS实现Cast单算子并完成功能验证
- 算子功能：根据源操作数和目的操作数Tensor的数据类型进行精度转换
- 调用方式：Kernel直调


## 样例支持AI处理器型号
- Ascend 910C
- Ascend 910B


## 算子描述

- 算子数学计算公式：
  $$
  \operatorname{Cast}(x)= {\operatorname{Cast<CastMode>}(\mathbf{x})}
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
      <td>表示进行精度转换的输入。公式中的`x`。</td>
      <td>half、float、int32_t</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>精度转换后的结果。</td>
      <td>half、float、int32_t</td>
      <td>ND</td>
    </tr>
  </tbody></table>
  算子规格说明：  

- 总的输入Shape(M1, ..., Mk)要满足：
    - k <= 4
    - Mi <= 10240，i <= k
- Tile块的Shape(m, n)，要满足m * n <=10240
- 目前只支持half、float、int32_t之间的类型转换，详见[AscendC文档](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0073.html)

CastMode说明：  

- 精度转换处理模式

- CastMode为枚举类型，用以控制精度转换处理模式，具体定义为：
```cpp
enum class CastMode {
    CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
    CAST_RINT,      // rint，四舍六入五成双舍入
    CAST_FLOOR,     // floor，向负无穷舍入
    CAST_CEIL,      // ceil，向正无穷舍入
    CAST_ROUND,     // round，四舍五入舍入
    CAST_TRUNC,     // trunc，向零舍入
    CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入    
};

```
更多详细说明参考[AscendC文档](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0073.html)


## 目录结构

| 文件名                    | 描述               |
|------------------------|------------------|
| [cast.cpp](./cast.cpp) | Cast算子代码实现以及调用样例 |

## 算子运行
在代码仓目录下执行：
```bash
cd ./examples
bash run_examples.sh cast
```