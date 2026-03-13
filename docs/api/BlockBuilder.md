# BlockBuilder
## 功能说明
block层对象构建类，负责创建block层对象，block层对象包含block层policy和block层调度。
## 所属头文件链接
[/include/elewise/block/builder.h](../../include/elewise/block/builder.h)
## 函数原型
```Cpp
template <typename Compute, typename ArchTagcfg = Atvoss::Arch::DAV_3510, const auto& Policy = defaultBlockPolicy, 
  typename ScheduleCfg = DefaultBlockConfig, template <typename, const auto&, typename, typename> class Schedule = DefaultBlockSchedule>
class BlockBuilder
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
    <td>Compute</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>Tile层计算表达图对象类型，跟block层是被包含关系</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>ArchTagcfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>用户配置的芯片版本型号</td>
    <td>Atvoss::Arch::DAV_3510</td>
  </tr>
  <tr>
    <td>Policy</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>block层的用户静态策略类型</td>
    <td>DefaultBlockPolicy</td>
  </tr>
  <tr>
    <td>ScheduleCfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>block层调度配置类型</td>
    <td>DefaultBlockConfig</td>
  </tr>
  <tr>
    <td>Schedule</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>block层调度类型</td>
    <td>DefaultBlockSchedule</td>
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
    <td>BlockBuilder</td>
    <td>返回block层对象</td>
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

    // 🔥🔥🔥 使用示例 🔥🔥🔥
    using BlockOp = Atvoss::Ele::BlockBuilder<AddSubCompute, ArchTag>;
    // 🔥🔥🔥 使用示例 🔥🔥🔥

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