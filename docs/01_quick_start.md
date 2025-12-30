# 快速入门

## 依赖项

ATVOSS支持源码编译，进行源码编译前，首先确保系统满足以下要求：

-  **AI处理器型号**
    - Ascend 910C
    - Ascend 910B


- 安装如下依赖：
    - python >= 3.7.0

    - gcc >= 7.3.0

    - cmake >= 3.16.0

    - gtest（可选，仅执行UT时依赖）

        下载gtest[源码](https://github.com/google/googletest.git)后执行以下命令安装：
        ```bash
        mkdir temp && cd temp                # 在gtest源码根目录下创建临时目录并进入
        cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
        make
        make install                         # root用户安装gtest
        # sudo make install                  # 非root用户安装gtest
        ```

## 环境准备
- **安装驱动与固件（运行态依赖）**

   模板算子运行时必须安装驱动与固件，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。
    - 建议安装2025年10月1日之后发布的驱动固件，驱动版本信息查询方式如下：
        ```bash
        # 查看版本
        cat /usr/local/Ascend/driver/version.info

        # 结果显示示例如下：
        Version=8.5.T9.0.B067
        timestamp=20251127_001525319
        ```
    其中`timestamp`为版本发布时间。


- **安装社区尝鲜版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run/software/8.5.0-beta.1/x86_64/Ascend-cann-toolkit_8.5.0-beta.1_linux-x86_64.run)、[toolkit aarch64包](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run/software/8.5.0-beta.1/aarch64/Ascend-cann-toolkit_8.5.0-beta.1_linux-aarch64.run)。
    
    ```bash
    # 需要确保安装目录权限至少为755
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径。
    - 缺省--install-path时， 则使用默认路径安装。
    若使用root用户安装，安装完成后相关软件存储在“/usr/local/Ascend/”路径下；若使用非root用户安装，安装完成后相关软件存储在“$HOME/Ascend/”路径下。


**环境变量设置示例：**

- 默认路径，root用户安装

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

- 默认路径，非root用户安装
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

- 指定路径安装
    ```bash
    source ${install_path}/cann/set_env.sh
    ```


## 源码下载

```bash
git clone https://gitcode.com/cann/atvoss.git
```
用户可直接通过导入源码路径下的include路径中对外头文件（include/atvoss.h）使能模板库能力。
在算子开发之前，可以通过如下[UT测试验证](#ut测试可选)和[样例验证](#样例运行验证)，确保当前环境是否完备。

## UT测试（可选）

在开源仓根目录执行下列命令之一，将依次批跑test目录下的用例，得到结果日志，用于看护编译是否正常。

```bash
bash build.sh -u
```

或

```bash
bash build.sh --utest
```
### UT测试显示覆盖率

- 依赖项
    - lcov >= 1.14

- 执行命令

```bash
bash build.sh --utest --cov
```

## 样例运行验证
开发者调用ATVOSS实现自定义算子开发后，可通过单算子调用的方式验证算子功能。本仓提供部分算子实现及其调用样例，具体请参考[examples](../examples)目录下的样例。

以`rms_norm`算子样例为例，说明样例运行验证的步骤：调用以下命令编译代码并执行：
```bash
# 切换到样例的目录
cd ../examples
# 编译执行命令
bash run_examples.sh rms_norm
```

若提示如下信息，则说明算子运行成功，精度校验通过。更详细的用例执行流程请参阅[样例介绍](../examples/README.md)。
```bash
Accuracy verification passed.
Sample rms_norm passed!
```