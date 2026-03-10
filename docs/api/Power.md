# Power
## 功能说明
幂运算。
## 所属头文件链接
[/include/operators/math_expression.h](../../include/operators/math_expression.h)
## 函数原型
```Cpp
template<auto scalarValue, typename T>
struct OpPower : UnaryOp<T>

template<auto scalarValue, typename T>
__host_aicore__ constexpr auto Power(Expression<T> lhs)

template<auto scalarValue, typename T>
__host_aicore__ constexpr auto Power(T &&lhs)
```
## 参数说明
<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
<col style="width: 150px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 600px">
<col style="width: 200px">
</colgroup>
<thead>
  <tr>
    <th>参数名称</th>
    <th>参数类型</th>
    <th>输入/输出</th>
    <th>数据类型</th>
    <th>参数说明</th>
    <th>默认值</th>
  </tr></thead>
<tbody>
  <tr>
    <td>scalarValue</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>Power操作的幂次</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>T</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>Power操作数的数据类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>lhs</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>NA</td>
    <td>Power操作数，当类型是Expression&lt;T&gt;时，是张量，当类型是T时，是标量</td>
    <td>NA</td>
  </tr>
</tbody>
</table>

## 返回值说明
<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
<col style="width: 200px">
<col style="width: 1000px">
</colgroup>
<thead>
  <tr>
    <th>返回值数据类型</th>
    <th>返回值说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Expression<OpPower<scalarValue, T>></td>
    <td>返回一个OpPower的表达式</td>
  </tr>
</tbody>
</table>

## 约束说明
NA
## 使用示例
```Cpp
template <typename InputDtype, typename OutputDtype>
struct Config {
    struct Compute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            return (out = Power<2>(in));
            // 🔥🔥🔥 使用示例 🔥🔥🔥
        };
    };
};

template <typename InputDtype, typename OutputDtype>
struct Config {
    struct Compute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto scalar = Atvoss::PlaceHolder<1, InputDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<2, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            return (out = Power<2>(scalar));
            // 🔥🔥🔥 使用示例 🔥🔥🔥
        };
    };
};
```