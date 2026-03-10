# UnaryOp
## 功能说明
一元运算符，所有一元运算符的基类。
## 所属头文件链接
[/include/expression/expr_template.h](../../include/expression/expr_template.h)
## 函数原型
```Cpp
template <typename T, typename R = typename std::decay_t<T>::RetType>
struct UnaryOp
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
    <td>一元操作对象的类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>R</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>一元操作符返回结果的类型</td>
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
    <td>UnaryOp</td>
    <td>返回一个一元运算符对象</td>
  </tr>
</tbody>
</table>

## 约束说明
NA
## 使用示例
```Cpp
template<auto scalarValue, typename T>
    // 🔥🔥🔥 使用示例 🔥🔥🔥
    struct OpPower : UnaryOp<T> {
    // 🔥🔥🔥 使用示例 🔥🔥🔥
        OpPower() = default;
        constexpr OpPower(T t) : UnaryOp<T>(t) {}
    };
```