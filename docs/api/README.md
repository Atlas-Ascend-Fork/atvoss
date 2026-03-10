# ATVOSS接口列表
#### 根据ATVOSS架构分层和模板归属不同，提供给用户可用接口列表如[表1](#table_ATVOSS)所示
**表1**  ATVOSS接口列表
<a name="table_ATVOSS"></a>
<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
<col style="width: 110px">
<col style="width: 120px">
<col style="width: 160px">
<col style="width: 710px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>分类</th>
    <th>模板</th>
    <th>接口</th>
    <th>说明</th>
    <th>所属头文件</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="3">入参构造</td>
    <td rowspan="3">NA</td>
    <td><a href="./ArgumentsBuilder_inputOutput.md">ArgumentsBuilder::inputOutput</a></td>
    <td>用户输入输出构造方法</td>
    <td rowspan="3">/include/utils/arguments/arguments.h</td>
  </tr>
  <tr>
    <td><a href="./ArgumentsBuilder_attr.md">ArgumentsBuilder::attr</a></td>
    <td>用户属性构造方法</td>
  </tr>
  <tr>
    <td><a href="./ArgumentsBuilderImpl_build.md">ArgumentsBuilderImpl::build</a></td>
    <td>用户参数对象生成器</td>
  </tr>
  <tr>
    <td rowspan="2">Device层</td>
    <td rowspan="2">Elewise</td>
    <td><a href="./DeviceAdapter.md">DeviceAdapter</a></td>
    <td>device适配层对象构造函数</td>
    <td rowspan="2">/include/elewise/device/device_adapter.h</td>
  </tr>
  <tr>
    <td><a href="./DeviceAdapter_Run.md">DeviceAdapter::Run</a></td>
    <td>device适配层主运行接口</td>
  </tr>
  <tr>
    <td rowspan="6">Kernel层</td>
    <td rowspan="6">Elewise</td>
    <td><a href="./KernelBuilder.md">KernelBuilder</a></td>
    <td>kernel层对象构造函数</td>
    <td rowspan="3">/include/elewise/kernel/builder.h</td>
  </tr>
  <tr>
    <td><a href="./DefaultKernelPolicy.md">DefaultKernelPolicy</a></td>
    <td>默认的kernel层静态policy</td>
  </tr>
  <tr>
    <td><a href="./DefaultKernelConfig.md">DefaultKernelConfig</a></td>
    <td>默认的kernel层schedule配置参数的数据结构</td>
  </tr>
  <tr>
    <td><a href="./DefaultKernelSchedule.md">DefaultKernelSchedule</a></td>
    <td>默认的kernel层schedule调度策略</td>
    <td rowspan="3">/include/elewise/kernel/schedule.h</td>
  </tr>
  <tr>
    <td><a href="./BaseKernelSchedule_MakeScheduleConfig.md">BaseKernelSchedule::MakeScheduleConfig</a></td>
    <td>kernel层schedule基类的生成scheduleCfg配置信息方法</td>
  </tr>
  <tr>
    <td><a href="./BaseKernelSchedule_Run.md">BaseKernelSchedule::Run</a></td>
    <td>kernel层schedule基类的执行调度策略方法</td>
  </tr>
  <tr>
    <td rowspan="6">Block层</td>
    <td rowspan="6">Elewise</td>
    <td><a href="./BlockBuilder.md">BlockBuilder</a></td>
    <td>block层对象构造函数</td>
    <td rowspan="3">/include/elewise/block/builder.h</td>
  </tr>
  <tr>
    <td><a href="./DefaultBlockPolicy.md">DefaultBlockPolicy</a></td>
    <td>默认的block层静态policy</td>
  </tr>
  <tr>
    <td><a href="./DefaultBlockConfig.md">DefaultBlockConfig</a></td>
    <td>默认的block层schedule配置参数的数据结构</td>
  </tr>
  <tr>
    <td><a href="./DefaultBlockSchedule.md">DefaultBlockSchedule</a></td>
    <td>默认的block层schedule调度策略</td>
    <td rowspan="3">/include/elewise/block/schedule.h</td>
  </tr>
  <tr>
    <td><a href="./BaseBlockSchedule_MakeScheduleConfig.md">BaseBlockSchedule::MakeScheduleConfig</a></td>
    <td>block层schedule基类的生成scheduleCfg配置信息方法</td>
  </tr>
  <tr>
    <td><a href="./BaseBlockSchedule_Run.md">BaseBlockSchedule::Run</a></td>
    <td>block层schedule基类的执行调度策略方法</td>
  </tr>
  <tr>
    <td rowspan="7">Compute层</td>
    <td rowspan="7">NA</td>
    <td><a href="./Compute.md">Compute</a></td>
    <td>用户表达Compute运算逻辑关系的静态配置</td>
    <td>NA</td>
  </tr>
  <tr>
    <td><a href="./PlaceHolder.md">PlaceHolder</a></td>
    <td>在Compute表达中，用户定义参数对象的函数方法</td>
    <td rowspan="6">/include/expression/expr_template.h</td>
  </tr>
  <tr>
    <td><a href="./PlaceHolderTmpLike.md">PlaceHolderTmpLike</a></td>
    <td>在Compute表达中，用户定义临时对象的函数方法</td>
  </tr>
  <tr>
    <td><a href="./ParamUsage.md">ParamUsage</a></td>
    <td>指定参数的数据流向</td>
  </tr>
  <tr>
    <td><a href="./UnaryOp.md">UnaryOp</a></td>
    <td>一元运算符</td>
  </tr>
  <tr>
    <td><a href="./BinaryOp.md">BinaryOp</a></td>
    <td>二元运算符</td>
  </tr>
  <tr>
    <td><a href="./TernaryOp.md">TernaryOp</a></td>
    <td>三元运算符</td>
  </tr>
</tbody>
</table>

#### 为了方便用户表达compute运算逻辑，提供给用户运算符接口列表如[表2](#table_Operator)所示
**表2**  Operator接口列表
<a name="table_Operator"></a>
<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
<col style="width: 200px">
<col style="width: 1000px">
</colgroup>
<thead>
  <tr>
    <th>操作符名称</th>
    <th>操作符说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="./Add.md">+</a></td>
    <td>加法运算</td>
  </tr>
  <tr>
    <td><a href="./Sub.md">-</a></td>
    <td>减法运算</td>
  </tr>
  <tr>
    <td><a href="./Mul.md">*</a></td>
    <td>乘法运算</td>
  </tr>
  <tr>
    <td><a href="./Div.md">/</a></td>
    <td>除法运算</td>
  </tr>
  <tr>
    <td><a href="./Exp.md">Exp</a></td>
    <td>以自然常数e为底的指数运算</td>
  </tr>
  <tr>
    <td><a href="./Power.md">Power</a></td>
    <td>幂运算</td>
  </tr>
  <tr>
    <td><a href="./Sqrt.md">Sqrt</a></td>
    <td>开平方运算</td>
  </tr>
  <tr>
    <td><a href="./Cast.md">Cast</a></td>
    <td>数据类型转换运算</td>
  </tr>
  <tr>
    <td><a href="./Abs.md">Abs</a></td>
    <td>绝对值运算</td>
  </tr>
</tbody>
</table>