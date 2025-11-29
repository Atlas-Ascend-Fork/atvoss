# ATVOSS分层结构
## 概述
ATVOSS针对昇腾AI处理器的Vector计算的多核并行计算过程，提供了一套统一的编程模型。编程模型包含如下分层，由高到低分别是：

- Device层：Host侧调用总入口，完成参数校验、Acl资源管理、Host侧与Device侧的数据管理、切分计算、Workspace管理、Kernel调用等逻辑。
- Kernel层：Kernel函数总入口，负责多核间的任务分解，控制Block的调度。
- Block层：负责单核的任务分解到多个Tile块，控制数据搬运/计算数据流进行流水编排处理。
- Tile层：对Ascend C基础API进行封装，提供更大Tile块的搬运、计算等能力。
- Basic层：使用Ascend C基础API能力完成数据搬运计算等基础操作。
<br><img src="./images/architecture.png" width="50%" height="50%" style="margin: 20px 0;"><br>

## ATVOSS编程范式

ATVOSS基于上述分层结构，配套封装了计算表达模板，可实现简洁的Vector计算表达：

```cpp
static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;

// 描述计算逻辑
using DtypeV1 = T1;
using DtypeV2 = T2;
using DtypeV3 = T3;
using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;

struct RmsNormCompute {
    template <template <typename> class Tensor>
    __host_aicore__ constexpr auto Compute() const
    {
        auto in1 = Atvoss::PlaceHolder<1, Tensor<DtypeV1>, Atvoss::ParamUsage::in>();
        auto in2 = Atvoss::PlaceHolder<2, Tensor<DtypeV2>, Atvoss::ParamUsage::in>();
        auto out = Atvoss::PlaceHolder<3, Tensor<DtypeV3>, Atvoss::ParamUsage::out>();
        auto temp = Atvoss::PlaceHolderTmpLike<1>(in1);

        return (temp = in1 * in1,
                out = ReduceSum<Atvoss::Pattern::AR>(temp),
                out = Broadcast<Atvoss::Pattern::AB>(out),
                temp = Divs<WIDTH>(out),
                out = Sqrt(temp),
                temp = in1 / out,
                out = in2 * temp);
    }
};
```

以上代码为例，定义一个`RmsNormCompute`的Expr模板类，主要是用于描述算子计算流、输入输出参数信息、Tile块的切分设置等。`RmsNormOp`包含`__host_aicore__ constexpr auto Compute() const`接口完成相关配置。
- Tile块的切分设置，使用如下示例的固定写法：
```cpp
using TileShape = Atvoss::Shape<HEIGHT, WIDTH>;
```
- `Compute()`接口内的设置说明如下：

| 表达式元素| 功能描述|
| ------------ | ------------ |
| `PlaceHolder` | 定义输入输出参数信息，传入的模板参数为<序号，Type，layout，输入输出标识>|
| `PlaceHolderTmpLike` | 定义计算时需要使用的临时buffer|
| 返回值 | 使用Tile层API描述计算过程，1个Tile层的API操作实现一次计算过程，每个计算过程是按“,”隔开|


- 各层模板组装
```cpp
static constexpr Atvoss::EleWise::BlockPolicy<TileShape> blockPolicy {
    190 * 1024,
    TileShape{}
};
static constexpr Atvoss::EleWise::KernelPolicy kernelPolicy {
    48, 
    Atvoss::EleWise::PolicySegment::UniformSegment
};
// 选择Block模板类，使用RmsNormOp Expr模板封装BlockOp层模板
using BlockOp = Atvoss::EleWise::BlockBuilder<
    RmsNormCompute,
    blockPolicy,
    Atvoss::EleWise::Config>;
    
// 选择Kernel模板类，使用BlockOp层模板封装Kernel层模板
using KernelOp = Atvoss::EleWise::KernelBuilder<
    BlockOp,
    kernelPolicy,
    Atvoss::EleWise::Config>;
// 使用DeviceAdapter模板，使用KernelOp封装Device适配器
using DeviceOp = Atvoss::DeviceAdapter<KernelOp>;
```
更多模板使用指导，参见[开发指南](./02_developer_guide.md)

