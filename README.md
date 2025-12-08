# ATVOSS
## 🔥Latest News

- [2025/11] ATVOSS项目首次上线。
## 🚀概述
ATVOSS（Ascend C Templates for Vector Operator Subroutines）是一套基于[Ascend C](https://hiascend.com/cann/ascend-c)开发的Vector算子模板库，致力于为昇腾硬件上的Vector类融合算子提供极简、高效、高性能、高拓展的编程方式。


## 🔍目录结构

ATVOSS代码目录结构如下：

```
├── build.sh                            # 项目工程编译脚本
├── cmake                               # 项目工程编译目录
├── CMakeLists.txt                      # 编译配置文件
├── docs                                # 项目文档介绍
├── examples                            # ATVOSS 样例
├── include                             # 项目公共头文件
├── scripts                             # 项目脚本文件存放目录
├── README.md
├── test                                # UT测试工程目录
```

## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](./docs/01_quick_start.md)获取简易教程，包括环境搭建、编译执行、本地验证等操作。

- [环境准备](./docs/01_quick_start.md#环境准备)：安装软件包之前，需要完成搭建基础环境，包括第三方依赖等；基础环境搭建后需要完成社区版CANN软件包安装、环境变量配置等。
- [源码下载](./docs/01_quick_start.md#源码下载)：本项目源码下载。
- [编译安装](./docs/01_quick_start.md#编译安装)：环境准备好后，可对源码修改编译生成可部署的安装包。
- [UT测试](./docs/01_quick_start.md#ut测试(可选))：基于项目根目录的build.sh脚本，可执行UT用例，快速验证功能。
- [样例运行验证](./docs/01_quick_start.md#样例运行验证)：基础样例的编译、执行。

## 📖文档介绍

| 文档 | 说明 |
|------|------|
|[快速入门](./docs/01_quick_start.md)|快速体验项目的简易教程。|
|[编程指南](./docs/02_developer_guide.md)|使用ATVOSS实现算子开发的教程。|
|[分层设计](./docs/03_design.md)|介绍ATVOSS分层模型。|
|[表达式模板原理](./docs/04_expression.md)|介绍ATVOSS表达式模板原理。|

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)

