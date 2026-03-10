# BaseBlockSchedule::MakeScheduleConfig
## 功能说明
block层schedule基类，默认调度策略和用户自定义调度策略必须继承自该类，MakeScheduleConfig基类接口根据传入的参数信息和kernel层的参数信息，生成block层的scheduleCfg配置信息。
## 所属头文件链接
[/include/elewise/block/schedule.h](../../include/elewise/block/schedule.h)
## 函数原型
```Cpp
template <typename Compute, const auto& Policy, typename ScheduleCfg, typename ArchTagCfg=void>
class BaseBlockSchedule {
    template<typename Args, typename KernelScheduleCfg>
    static bool MakeScheduleConfig(const Args& arguments, const KernelScheduleCfg& kernelConfig, ScheduleCfg& blockConfig)
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
    <td>Compute</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>Tile层计算表达图对象类型，跟kernel层是被包含关系</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>Policy</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>block层的用户静态策略类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>ScheduleCfg</td>
    <td>模板参数</td>
    <td>输出</td>
    <td>NA</td>
    <td>block层调度配置类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>ArchTagCfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>用户配置的芯片版本型号</td>
    <td>void</td>
  </tr>
  <tr>
    <td>Args</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>用户的输入参数列表，类型根据用户传入的参数实例化</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>KernelScheduleCfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层schedule配置参数类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>arguments</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>NA</td>
    <td>用户的输入参数列表</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>kernelConfig</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层schedule配置参数</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>blockConfig</td>
    <td>函数形参</td>
    <td>输出</td>
    <td>NA</td>
    <td>block层schedule配置参数</td>
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
    <td>bool</td>
    <td>生成scheduleCfg配置信息成功还是失败，true：成功，false：失败</td>
  </tr>
</tbody>
</table>

## 约束说明
NA
## 使用示例
暂无