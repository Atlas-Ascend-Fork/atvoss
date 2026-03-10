# DeviceAdapter::Run
## 功能说明
device适配层的主运行接口，负责完成host侧的参数解析，以及准备device侧入参的数据结构对象。
## 所属头文件链接
[/include/elewise/device/device_adapter.h](../../include/elewise/device/device_adapter.h)
## 函数原型
```Cpp
template <typename KernelOp>
class DeviceAdapter{
    template <typename Args>
    int64_t Run(const Args& arguments, aclrtStream stream = nullptr)
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
    <td>Args</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>用户的输入参数列表，类型根据用户传入的参数实例化</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>arguments</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>Args</td>
    <td>用户传入的参数列表</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>aclrtStream</td>
    <td>用户创建的stream流对象</td>
    <td>nullptr</td>
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
    <td>int64_t</td>
    <td>device层执行的结果，0：成功，-1：失败</td>
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

    auto arguments = Atvoss::ArgumentsBuilder{}.inputOutput(in1, in2, in3, out).attr("dim", 5).build();

    using DeviceOp = typename AddSubConfig<InputDtype, OutputDtype>::DeviceOp;
    DeviceOp deviceOp;

    // 🔥🔥🔥 使用示例 🔥🔥🔥
    deviceOp.Run(arguments, stream);
    // 🔥🔥🔥 使用示例 🔥🔥🔥
}

int main(int argc, char const* argv[]) {
    Run<float, float>();
    return 0;
}
```