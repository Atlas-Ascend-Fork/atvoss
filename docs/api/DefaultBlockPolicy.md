# DefaultBlockPolicy
## 功能说明
默认的block层静态policy。
## 所属头文件链接
[/include/elewise/block/builder.h](../../include/elewise/block/builder.h)
## 数据结构
```Cpp
template <typename Shape>
struct DefaultBlockPolicy {
    using TileShape = Shape;
    Shape tileShape{};
    Atvoss::MemMngPolicy memPolicy = Atvoss::MemMngPolicy::AUTO;
};
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
    <td>tileShape</td>
    <td>Atvoss::Shape</td>
    <td>tile块的shape信息</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>memPolicy</td>
    <td>Atvoss::MemMngPolicy</td>
    <td>ATVOSS内存管理策略</td>
    <td>Atvoss::MemMngPolicy::AUTO</td>
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
    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    using BlockOp = Atvoss::Ele::BlockBuilder<AddSubCompute, ArchTag, blockPolicy>;
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