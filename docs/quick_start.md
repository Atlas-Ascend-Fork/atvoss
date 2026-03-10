# ATVOSS开发快速入门

## 介绍

ATVOSS（Ascend C Templates for Vector Operator Subroutines）是一套基于[Ascend C](https://hiascend.com/cann/ascend-c)开发的Vector算子模板库，致力于为昇腾硬件上的Vector类融合算子提供极简、高效、高性能、高拓展的编程方式。

环境搭建一般分为如下场景，您可以按需安装：

- 编译态：针对仅编译不运行本项目的场景，只需安装前置依赖和CANN toolkit包。
- 运行态：针对运行本项目的场景（编译运行或纯运行），除了安装前置依赖和CANN toolkit包，还需安装驱动与固件、CANN ops包。

## 环境准备

### 系统要求
ATVOSS支持源码编译，进行源码编译前，请确保如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0（建议版本 <= 3.10） 
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - gawk
   - make
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作。
   
   单击[下载链接](https://www.hiascend.com/hardware/firmware-drivers/community)，根据实际产品型号和环境架构，获取对应的`Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`、`Ascend-hdk-<chip_type>-npu-firmware_<version>.run`包。

   安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》中“安装指南 > 安装NPU驱动和固件”。


## 环境准备
- **支持的产品**
    - Ascend 950PR/Ascend 950DT

### 手动安装CANN包

#### 1. 下载软件包

根据实际产品型号和环境架构, 获取[x86_64版本](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260306_newest/Ascend-cann-toolkit_9.0.0_linux-x86_64.run)或者[arm版本](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260306_newest/Ascend-cann-toolkit_9.0.0_linux-aarch64.run)的```Ascend-cann-toolkit_${cann_version}_linux-${arch}.run```安装包。

#### 2. 安装软件包
1. **安装社区CANN toolkit包**

    ```bash
    # 需要确保安装目录权限至少为755
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

## 环境验证

安装完CANN包或进入Docker容器后，需验证环境和驱动是否正常。

-   **检查NPU设备**（仿真执行，跳过此步骤）：
    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```
-   **检查CANN安装**：
    ```bash
    # 查看CANN Toolkit版本信息（非root用户，将/usr/local替换为${HOME}）
    cat /usr/local/Ascend/cann/opp/version.info
    ```
## 环境变量配置

按需选择合适的命令使环境变量生效。
```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
source ${install_path}/cann/set_env.sh
```

## 源码下载

```bash
# 下载项目源码
git clone -b master git@gitcode.com:cann/atvoss.git
```

> [!NOTE] 注意
> gitcode平台在使用 SSH 协议时，请在本地生成 SSH 公钥进行克隆、推送等操作。

## 编译执行

开发者调用ATVOSS实现自定义算子开发后，可通过单算子调用的方式验证算子功能。本仓提供部分算子实现及其调用样例，具体请参考[examples](../examples)目录下的样例。

1. 编译example
   ATVOSS仓提供一键式编译examples的能力，可以指定单个example编译(例如，编译examples/rms_norm目录里的用例)
   ```bash
   bash scripts/build.sh -DSOC=ascend950 rms_norm
   ```
2. 单样例执行（依赖Device环境）
   编译完成后会生成`output/bin/rms_norm`可执行文件
   ```bash
   ./output/bin/rms_norm --shape=512,3
   ```
3. 仿真执行（不依赖Device环境）
    利用cannsim实现仿真运行, 将上述执行脚本写到自建的Shell脚本，如run.sh中
    ```shell
    # run.sh
    ./output/bin/rms_norm --shape=512,3
    ```
    给自建的run.sh增加可执行权限，然后使用仿真命令cannsim执行
    ```bash
    chmod +x run.sh
    cannsim record ./run.sh -s Ascend950 --gen-report
    ```
    执行完成后会在当前目录生成一个cannsim_xxx的结果文件夹。
    
若提示如下信息，则说明算子运行成功，精度校验通过。更详细的用例执行流程请参阅[样例介绍](../examples/README.md)。
```bash
Accuracy verification passed.
```

## UT测试（可选）

1. 编译&执行host_ut
   ```bash
   bash scripts/build.sh -DSOC=ascend950 --host_ut
   ```
   编译完成会直接运行host_ut并输出ut执行结果。

## 相关文档
- [项目整体介绍](./summary.md) - 项目介绍
- [目录结构](./directory_structure.md) - 目录结构
- [开发者指南](./tutorials/developer_guide.md) - 详细的开发流程
- [API文档](./api/README.md) - API文档
