# ParamUsage
## 功能说明
用来指定参数的数据流向，这里的参数是指struct Param。
## 所属头文件链接
[/include/expression/expr_template.h](../../include/expression/expr_template.h)
## 数据结构
```Cpp
enum class ParamUsage {
    IN,
    OUT,
    IN_OUT,
}
```
## 数据结构成员说明
<table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 1000px">
<col style="width: 200px">
</colgroup>
<thead>
  <tr>
    <th>成员名称</th>
    <th>成员类型</th>
    <th>成员说明</th>
    <th>默认值</th>
  </tr></thead>
<tbody>
  <tr>
    <td>IN</td>
    <td>NA</td>
    <td>输入</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>OUT</td>
    <td>NA</td>
    <td>输出</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>IN_OUT</td>
    <td>NA</td>
    <td>输入&输出</td>
    <td>NA</td>
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
            // 🔥🔥🔥 使用示例 🔥🔥🔥
            auto in1 = Atvoss::PlaceHolder<1, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto in2 = Atvoss::PlaceHolder<2, Tensor<InputDtype>, Atvoss::ParamUsage::IN>();
            auto in3 = Atvoss::PlaceHolder<3, InputDtype, Atvoss::ParamUsage::IN>();
            auto out = Atvoss::PlaceHolder<4, Tensor<OutputDtype>, Atvoss::ParamUsage::OUT>();
            // 🔥🔥🔥 使用示例 🔥🔥🔥

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