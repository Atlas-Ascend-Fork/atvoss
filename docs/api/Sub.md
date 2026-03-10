# Sub
## 功能说明
减法运算，支持张量-张量，张量-标量，标量-张量。
## 所属头文件链接
[/include/operators/math_expression.h](../../include/operators/math_expression.h)
## 函数原型
```Cpp
template<typename T, typename U>
struct OpSub : BinaryOp<T, U>

template<typename T, typename U>
__host_aicore__ constexpr auto operator-(Expression<T> lhs, Expression<U> rhs)

template<typename T, typename U>
__host_aicore__ constexpr auto operator-(Expression<T> lhs, U &&rhs)

template<typename T, typename U>
__host_aicore__ constexpr auto operator-(T &&lhs, Expression<U> rhs)
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
    <td>T</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>减法左操作数数据类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>U</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>减法右操作数数据类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>lhs</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>NA</td>
    <td>减法左操作数，当类型是Expression&lt;T&gt;时，是张量，当类型是T时，是标量</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>rhs</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>NA</td>
    <td>减法右操作数，当类型是Expression&lt;U&gt;时，是张量，当类型是U时，是标量</td>
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
    <td>Expression<OpSub<T, U>></td>
    <td>返回一个OpSub的表达式</td>
  </tr>
</tbody>
</table>

## 约束说明
不支持广播
## 使用示例
```Cpp
template <typename InputDtype, typename OutputDtype>
struct Config {
    struct Compute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            return (out = in1 - in2);
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
            auto in = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto scalar = Atvoss::PlaceHolder<2, InputDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            return (out = in - scalar);
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
            auto in = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto scalar = Atvoss::PlaceHolder<2, InputDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<3, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            return (out = scalar - in);
            // 🔥🔥🔥 使用示例 🔥🔥🔥
        };
    };
};
```