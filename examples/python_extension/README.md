# PyTorch 调用

## 简介

本文档演示如何基于 **ATVOSS** 与 [PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html) 机制开发自定义 NPU 算子，并将其无缝集成到 PyTorch API 中，供用户直接调用。

---

## 环境准备

在开始之前，请确保完成以下准备工作：

1. **基础环境搭建**  
   请参考 [环境部署指南](../../docs/01_quick_start.md) 完成 Ascend CANN、驱动及开发工具链的安装。

2. **PyTorch 依赖**  
   安装兼容的 PyTorch 及 torch_npu 包，版本要求如下：
   - `torch >= 2.6.0`
   - 对应版本的 [`torch_npu`](https://gitcode.com/Ascend/pytorch/releases)

---

## ATVOSS 安装

进入 python_extension目录并完成安装：

```sh
cd examples/python_extension
```

### 1. 安装 Python 依赖
```sh
python3 -m pip install -r requirements.txt
```

### 2. 构建 Wheel 包
```sh
# -n: 使用当前环境（非隔离构建）
python3 -m build --wheel -n
```

构建成功后，Wheel 文件将生成在 `dist/` 目录下，命名格式为：  
`ascend_ops-1.0.0-${python_version}-abi3-${arch}.whl`

其中：
- `${python_version}`：Python 版本标识（如 `cp38` 表示 Python 3.8）
- `${arch}`：CPU 架构（如 `x86_64` 或 `aarch64`）

### 3. 安装扩展包
```sh
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

> **说明**：`--no-deps` 避免重复安装已满足的依赖；`--force-reinstall` 确保覆盖旧版本。

### 4. （可选）清理编译缓存
若需重新构建，请先清理历史编译产物：
```sh
python setup.py clean
```

---

完成上述步骤后，即可在 Python 中通过 `import ascend_ops` 调用 ATVOSS 提供的自定义 NPU 算子。

当前已提供算子: [torch.abs](./csrc/abs/ascend950/abs.cpp) 。


## 算子调用

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以abs算子调用为例。
```python
import torch
import torch_npu
import ascend_ops

a = torch.randn(2, 3, dtype=torch.float32)
a_npu = a.npu()
result_npu = torch.ops.ascend_ops.abs(a_npu)
```


## 开发指南：新增一个算子

为了实现一个新算子(如`abs`)，您只需要提供一个C++实现即可。

1. 首先您需要在csrc目录下使用算子名`abs`建立一个文件夹，在此文件夹内使用你当前想要开发的soc名建立一个子文件夹`ascend950`。

2. 在soc目录下新建一个`CMakeLists.txt`
    ```
    add_sources("--npu-arch=dav-3510")
    ```
    这里`dav-3510`为ascend950芯片对应的编译参数，获取方法参考[NpuArch说明和使用指导](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md)。

3. 在soc目录下新建一个`abs.cpp`(建议使用算子名为文件名)。这个文件包含了开发一个AI Core算子所需要的全部模块。
    - 算子Schema注册
    ```cpp
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("abs(Tensor x) -> Tensor");
    }
    ```

    - 算子Meta Function实现 & 注册
    ```cpp
    // Meta function implementation of Abs
    torch::Tensor abs_meta(const torch::Tensor &x)
    {
        auto y = torch::empty_like(x);
        return y;
    }

    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("abs", abs_meta);
    }
    ```

    - 算子Kernel实现 (Ascend C)
    ```cpp
    template <typename T>
    struct AbsConfig {
        // 算子Kernel实现
    };
    ```

    - 算子NPU调用实现 & 注册
    ```cpp
    torch::Tensor abs_npu(const torch::Tensor &x)
    {
        auto y = abs_meta(x);
        uint32_t shape[2] = {};
        std::copy(x.sizes().begin(), x.sizes().end(), shape);

        Atvoss::Tensor<float> t1(x.data_ptr<float>(), shape);
        Atvoss::Tensor<float> t2(y.data_ptr<float>(), shape);

        auto arguments = Atvoss::ArgumentsBuilder{}
                             .inputOutput(t1, t2)
                             .build();
        using Config = AbsConfig<float>;
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        Config::DeviceOp deviceOp;
        deviceOp.Run(arguments, stream);
        return y;
    }

    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("abs", abs_npu);
    }
    ```
具体代码可以参考[abs.cpp](./csrc/abs/ascend950/abs.cpp)

4. 参考[安装步骤](#atvoss-安装)章节重新构建Wheel包并安装。
5. 基于pytest测试算子API，请参考[test_abs.py](tests/abs/test_abs.py)的实现; 

