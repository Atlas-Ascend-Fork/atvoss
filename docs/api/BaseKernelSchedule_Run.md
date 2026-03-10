# BaseKernelSchedule::Run
## 功能说明
kernel层schedule基类，默认调度策略和用户自定义调度策略必须继承自该类，Run基类接口执行调度策略。
## 所属头文件链接
[/include/elewise/kernel/schedule.h](../../include/elewise/kernel/schedule.h)
## 函数原型
```Cpp
template <typename BlockOp, const auto &Policy, typename ScheduleCfg>
class BaseKernelSchedule {
    template <typename OpParam, typename... Args>
    aicore inline void Run(OpParam& cfg, Args... args)
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
    <td>NA</td>
  </tr>
  <tr>
    <td>ScheduleCfg</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>kernel层调度配置类型</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>OpParam</td>
    <td>模板参数</td>
    <td>输入</td>
    <td>NA</td>
    <td>scheduleCfg，根据用户设置的类型实例化</td>
    <td>NA</td>
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
    <td>cfg</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>OpParam</td>
    <td>用户定义的schedule配置</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>args</td>
    <td>函数形参</td>
    <td>输入</td>
    <td>Args</td>
    <td>用户的输入参数列表</td>
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
    <td>void</td>
    <td>NA</td>
  </tr>
</tbody>
</table>

## 约束说明
NA
## 使用示例
暂无