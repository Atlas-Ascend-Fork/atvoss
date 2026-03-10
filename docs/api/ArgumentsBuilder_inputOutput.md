# ArgumentsBuilder::inputOutput
## 功能说明
用户侧输入输出对象构造方法，支持用户传入输入/输出tensor，以及host侧的一些属性参数，函数参数中的参数列表和Compute表达中的参数列表在时序上完全对应，两者配合使用。
## 所属头文件链接
[/include/utils/arguments/arguments.h](../../include/utils/arguments/arguments.h)
## 函数原型
```Cpp
struct ArgumentsBuilder {
    template<typename... InitialInputOutput>
    constexpr auto inputOutput(InitialInputOutput&... inputOutput) const
}
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
    <td>InitialInputOutput</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>可变参数列表类型，其中每个参数类型必须为Tensor或非指针类的scalar，根据用户传入的参数列表实例化</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>inputOutput</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>InitialInputOutput</td>
    <td>用户传入的参数列表</td>
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
    <td>ArgumentsBuilderImpl</td>
    <td>返回参数构建器对象</td>
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
            return (out = in1 + in2 - in3);
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

    // 🔥🔥🔥 使用示例 🔥🔥🔥
    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in1, in2, in3, out).build();
    // 🔥🔥🔥 使用示例 🔥🔥🔥

    using DeviceOp = typename AddSubConfig<InputDtype, OutputDtype>::DeviceOp;
    DeviceOp deviceOp;
    deviceOp.Run(arguments, stream);
}

int main(int argc, char const* argv[]) {
    Run<float, float>();
    return 0;
}
```