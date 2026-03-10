# DefaultBlockConfig
## 功能说明
默认的block层schedule配置参数的数据结构。
## 所属头文件链接
[/include/elewise/block/schedule.h](../../include/elewise/block/schedule.h)
## 数据结构
```Cpp
struct DefaultBlockConfig {
    uint32_t wholeLoop = 0;
    uint32_t tileCnt = 0;
    uint32_t basicNum = 0;
    uint32_t totalElemCnt = 0;
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
    <td>wholeLoop</td>
    <td>uint32_t</td>
    <td>当前核上tile块的轮询次数</td>
    <td>0</td>
  </tr>
  <tr>
    <td>tileCnt</td>
    <td>uint32_t</td>
    <td>当前轮询处理的tile块中包含的元素个数，整块tile块时，为0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>basicNum</td>
    <td>uint32_t</td>
    <td>整块tile块包含的元素个数</td>
    <td>0</td>
  </tr>
  <tr>
    <td>totalElemCnt</td>
    <td>uint32_t</td>
    <td>当前核上处理的总元素个数</td>
    <td>0</td>
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

    static constexpr Atvoss::Ele::DefaultBlockPolicy<TileShape> blockPolicy{TileShape{}};
    using ArchTag = Atvoss::Arch::DAV_3510;

    using BlockOp = Atvoss::Ele::BlockBuilder<
      AddSubCompute, 
      ArchTag, 
      blockPolicy, 

      // 🔥🔥🔥 使用示例 🔥🔥🔥
      Atvoss::Ele::DefaultBlockConfig
      // 🔥🔥🔥 使用示例 🔥🔥🔥

      >;
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