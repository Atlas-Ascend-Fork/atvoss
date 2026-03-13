# PlaceHolderTmpLike
## 功能说明
在Compute表达中，用户定义临时对象的函数方法。
## 所属头文件链接
[/include/expression/expr_template.h](../../include/expression/expr_template.h)
## 函数原型
```Cpp
template <std::size_t N, typename T = void, typename L>
__host_aicore__ constexpr auto PlaceHolderTmpLike(Expression)
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
    <td>N</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>临时对象位序，从1开始顺序编号</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>T</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>临时对象类型，可以是基础类型和Tensor，如果不传，使用L模板参数的类型</td>
    <td>void</td>
  </tr>
  <tr>
    <td>L</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>临时对象按照L指定的对象来生成，L必须是struct Param类型</td>
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
    <td>Expression<LocalVar<N, T, U>></td>
    <td>返回一个LocalVar表达式对象</td>
  </tr>
</tbody>
</table>

## 约束说明
NA
## 使用示例
```Cpp
template <typename InputDtype, typename OutputDtype>
struct AddSubConfig {
    struct AddSubCompute {
        template <template <typename> class Tensor>
        __host_aicore__ constexpr auto Compute() const
        {
            auto in1 = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto in3 = Atvoss::PlaceHolder<3, InputDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<4, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();

            // 🔥🔥🔥 使用示例 🔥🔥🔥
            auto tmp = Atvoss::PlaceHolderTmpLike<1>(in1);
            // 🔥🔥🔥 使用示例 🔥🔥🔥

            return (tmp = in1 + in2;
                    out = tmp - in3);
        };
    };

    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<AddSubCompute, ArchTag>;
    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp>;
    using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
};

template <typename InputDtype, typename OutputDtype>
static void Run() {
    /* ACL init and stream create */
    ...

    Atvoss::Tensor<InputDtype> in1(deviceIn1, {{3, 4, 0, 0, 0, 0, 0, 0}}, 2);
    Atvoss::Tensor<InputDtype> in2(deviceIn2, {{3, 4, 0, 0, 0, 0, 0, 0}}, 2);
    InputDtype in3 = 5.0;
    Atvoss::Tensor<OutputDtype> out(deviceOut, {{3, 4, 0, 0, 0, 0, 0, 0}}, 2);

    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in1, in2, in3, out).attr("dim", 5).build();

    using DeviceOp = typename AddSubConfig<InputDtype, OutputDtype>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);
}

int main(int argc, char const* argv[]) {
    Run<float, float>();
    return 0;
}
```