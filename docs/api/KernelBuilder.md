# KernelBuilder
## 功能说明
kernel层对象构建类，负责创建kernel层对象，kernel层对象包含kernel层policy和kernel层调度。
## 所属头文件链接
[/include/elewise/kernel/builder.h](../../include/elewise/kernel/builder.h)
## 函数原型
```Cpp
template <typename BlockOp, const auto &Policy = defaultKernelPolicy, typename ScheduleCfg = DefaultKernelConfig,
  template <typename, const auto&, typename> class Schedule = DefaultKernelSchedule>
class KernelBuilder
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
    <td>BlockOp</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>block层对象类型，跟kernel层是被包含关系</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>Policy</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层的用户静态策略类型</td>
    <td>DefaultKernelPolicy</td>
  </tr>
  <tr>
    <td>ScheduleCfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层调度配置类型</td>
    <td>DefaultKernelConfig</td>
  </tr>
  <tr>
    <td>Schedule</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层调度类型</td>
    <td>DefaultKernelSchedule</td>
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
    <td>KernelBuilder</td>
    <td>返回kernel层对象</td>
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

    // 🔥🔥🔥 使用示例 🔥🔥🔥
    using KernelOp = Atvoss::Ele::KernelBuilder<BlockOp>;
    // 🔥🔥🔥 使用示例 🔥🔥🔥

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