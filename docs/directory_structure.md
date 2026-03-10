# Atvoss 项目目录结构说明

## 项目概述

Atvoss 是一个基于 AscendC 的高性能深度学习算子开发框架，提供声明式算子定义和自动代码生成能力。

## 根目录结构

```
atvoss/
├── CMakeLists.txt           # 顶层 CMake 构建配置
├── CONTRIBUTING.md          # 贡献指南
├── LICENSE                  # 开源许可证
├── README.md               # 项目说明文档
├── SECURITY.md            # 安全策略
├── version.info            # 版本信息
├── docs/                   # 项目文档目录
├── examples/               # 示例代码目录
├── include/                # 头文件目录（核心框架）
├── scripts/                # 构建脚本目录
├── tests/                  # 测试代码目录
└── third_party/            # 第三方依赖
```

## include/ 核心头文件目录

```
include/
├── atvoss.h               # 主入口头文件
│
├── common/                # 公共工具和类型定义
│   ├── types.h           # 数据类型定义
│   └── utils.h           # 通用工具函数
│
├── expression/           # 表达式系统
│   └── expr_template.h   # 表达式模板（DeclareUnaryOp/DeclareBinaryOp宏定义）
│
├── evaluator/            # 求值器系统
│   └── eval_base.h       # 求值器基类和接口定义
│
├── operators/            # 算子定义和实现
│   ├── math_expression.h     # 数学表达式声明（Sqrt/Exp/Abs/Sin等）
│   ├── math_evaluator.h      # 数学表达式求值器实现
│   ├── tensor_expression.h   # 张量表达式定义
│   ├── tensor_evaluator.h     # 张量求值器实现
│   ├── tile_shape.h          # Tiling 形状配置
│   ├── transcendental_expression.h   # 超越函数表达式
│   └── transcendental_evaluator.h     # 超越函数求值器
│
├── ele_wise/             # 逐元素操作模板
│   ├── elemwise.h        # 逐元素操作通用定义
│   └── ...
│
├── graph/                # 计算图系统
│   ├── compute_graph.h   # 计算图定义
│   └── ...
│
└── utils/                # 工具函数
    ├── tensor_utils.h    # 张量操作工具
    └── memory_utils.h   # 内存管理工具
```

## examples/ 示例代码目录

```
examples/
├── CMakeLists.txt        # 示例整体构建配置
├── README.md            # 示例使用说明
├── common/              # 公共示例代码
│   ├── command_line.h   # 命令行参数解析
│   └── example_common.h # 示例通用功能
│
├── muls/                # Muls 标量乘法示例
│   ├── muls.cpp         # Muls 算子实现
│   ├── CMakeLists.txt   # 构建配置
│   └── README.md        # 使用说明
│
└── python_extension/    # Python 扩展示例
    ├── CMakeLists.txt
    ├── requirements.txt
    ├── build_and_test.sh
    └── csrc/  
        ├── extension.cpp
        ├── abs/
        ├── add/
        ├── cast/
        └── ...
```

## docs/ 文档目录

```
docs/
├── quick_start.md        # 快速开始指南
├── summary.md               # 项目整体介绍
├── directory_structure.md   # 项目目录结构
├── api/                     # API文档目录
├── tutorials/               # 教程 
│   └── developer_guide.md   # 算子开发指南
│
└── images/                  # 文档配图
```

## tests/ 测试目录

```
tests/
├── CMakeLists.txt       # 测试构建配置
│
├── st/                  # 系统测试（System Tests）
│   ├── CMakeLists.txt
│   ├── test_op_*.cpp    # 各算子测试
│   ├── test_compute_*.cpp  # 计算相关测试
│   └── ...
│
└── ut/                  # 单元测试（Unit Tests）
    ├── host/            # 主机端测试
    │   ├── test_*.cpp
    │   └── CMakeLists.txt
    └── compile_perf/    # 编译性能测试
```

## scripts/ 构建脚本目录

```
scripts/
└── build.sh            # 主要构建脚本
                        # 支持参数：-DSOC=<arch>
                        # 示例：./scripts/build.sh -DSOC=ascend950 sin
```

## 相关文档
- [项目整体介绍](./summary.md) - 项目介绍
- [快速开始](./quick_start.md) - 快速开始
- [开发者指南](./tutorials/developer_guide.md) - 详细的开发流程
- [API文档](./api/README.md) - API文档