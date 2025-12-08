# 样例介绍
| 样例名                    | 描述                                                                       | 模板                                             | 算子调用方式   |
|------------------------|--------------------------------------------------------------------------|------------------------------------------------|----------|
| [rms_norm](./rms_norm) | 使用ATVOSS模板实现rms_norm算子以及调用样例 | KernelBuilder模板+BlockBuilder模板+DeviceAdapter模板 | Kernel直调 |
| [cast](./cast)         | 使用ATVOSS模板实现cast算子以及调用样例     | KernelBuilder模板+BlockBuilder模板+DeviceAdapter模板 | Kernel直调 |

# 运行模式介绍
以cast为例，其他样例类似：
- 默认运行模式  
运行命令如下：
```bash
cd ./examples
bash run_examples.sh cast
```
- profiling运行模式  
该模式下可以使用性能调优工具来采集和分析运行在昇腾处理器上的任务各个运行阶段的关键性能指标，用户可根据输出的性能数据，快速定位软、硬件性能瓶颈，提升性能分析的效率。详见[性能调优工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/devaids/Profiling/atlasprofiling_16_0001.html)。  
运行命令如下：
```bash
cd ./examples
bash run_examples.sh cast --run-mode=profiling
```