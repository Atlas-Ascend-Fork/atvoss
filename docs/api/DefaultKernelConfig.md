# DefaultKernelConfig
## 功能说明
默认的kernel层schedule配置参数的数据结构。
## 所属头文件链接
[/include/elewise/kernel/builder.h](../../include/elewise/kernel/builder.h)
## 数据结构
```Cpp
struct DefaultKernelConfig {
    uint32_t blockNum = 1;
    uint64_t unitNumPerCore = 0;
    uint64_t moreUnitCoreNum = 0;
    uint64_t tailNum = 0;
    uint64_t unitNum = 1;
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
    <td>blockNum</td>
    <td>uint32_t</td>
    <td>启用的核的数量</td>
    <td>1</td>
  </tr>
  <tr>
    <td>unitNumPerCore</td>
    <td>uint64_t</td>
    <td>平均每个核处理的基本块个数</td>
    <td>0</td>
  </tr>
  <tr>
    <td>moreUnitCoreNum</td>
    <td>uint64_t</td>
    <td>核均分后，需要处理额外多出来的基本块的核的数量</td>
    <td>0</td>
  </tr>
  <tr>
    <td>tailNum</td>
    <td>uint64_t</td>
    <td>最后一个核要处理的尾块元素数量</td>
    <td>0</td>
  </tr>
  <tr>
    <td>unitNum</td>
    <td>uint64_t</td>
    <td>基本块的元素数量</td>
    <td>1</td>
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

    static constexpr Atvoss::Ele::DefaultKernelPolicy kernelPolicy{Atvoss::Ele::DefaultSegmentPolicy::UniformSegment};
    using ArchTag = Atvoss::Arch::DAV_3510;
    using BlockOp = Atvoss::Ele::BlockBuilder<AddSubCompute, ArchTag>;
    using KernelOp = Atvoss::Ele::KernelBuilder<
      BlockOp, 
      kernelPolicy,

      // 🔥🔥🔥 使用示例 🔥🔥🔥
      Atvoss::Ele::DefaultKernelConfig
      // 🔥🔥🔥 使用示例 🔥🔥🔥

      >;

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