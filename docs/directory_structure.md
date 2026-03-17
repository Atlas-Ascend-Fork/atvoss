# Atvoss 项目目录结构说明

## 项目概述

Atvoss 是一个基于 AscendC 的高性能深度学习算子开发框架，提供声明式算子定义和自动代码生成能力。

## 根目录结构

```
atvoss/
├── CMakeLists.txt           # 顶层 CMake 构建配置
├── README.md                # 项目说明文档
├── version.info             # 版本信息
├── docs/                    # 项目文档目录
├── examples/                # 示例代码目录
├── include/                 # 头文件目录（核心框架）
├── scripts/                 # 构建脚本目录
└── tests/                   # 测试代码目录
```

## include/ 核心头文件目录

```
include/
├── atvoss.h                 # 主入口头文件
│
├── common/                  # 公共工具和类型定义
│   ├── arch.h               # 软件架构平台定义
│   ├── compile_info.h       # 编译信息
│   └── type_def.h           # 类型定义
│
├── expression/              # 表达式系统
│   └── expr_template.h      # 表达式模板（DeclareUnaryOp/DeclareBinaryOp宏定义）
│
├── evaluator/               # 求值器系统
│   └── eval_base.h          # 求值器基类和接口定义
│
├── operators/               # 算子定义和实现
│   ├── math_expression.h   # 数学表达式声明（Sqrt/Exp/Abs等）
│   ├── tensor_expression.h # 张量表达式定义
│   └── tile_shape.h         # Tiling 形状配置
│
├── elewise/                 # 逐元素操作模板
│   ├── block/               # Block 层实现
│   │   ├── builder.h        # Block 构建器
│   │   └── schedule.h       # Block 调度器
│   ├── kernel/              # Kernel 层实现
│   │   ├── builder.h        # Kernel 构建器
│   │   └── schedule.h       # Kernel 调度器
│   ├── device/              # Device 层实现
│   │   └── device_adapter.h # Device 适配器
│   └── tile/                # Tile 层实现
│       └── tile_evaluate.h  # Tile 求值器
│
├── graph/                   # 计算图系统
│   ├── buffer.h             # Buffer 管理
│   ├── dag.h                # DAG 图实现
│   └── expr_linearizer.h   # 表达式线性化
│
└── utils/                   # 工具函数
    ├── tensor.h              # 张量工具
    ├── utility.h             # 通用工具
    └── log.h                 # 日志工具
    ...
```

## examples/ 示例代码目录

```
examples/
├── CMakeLists.txt            # 示例整体构建配置
├── README.md                 # 示例使用说明
├── common/                   # 公共示例代码
│   └── example_common.h     # 示例通用功能
│
├── muls/                     # Muls 标量乘法示例
│   ├── muls.cpp              # Muls 算子实现
│   └── CMakeLists.txt        # 构建配置
│
├── abs/                      # Abs 绝对值示例
└── python_extension/         # Python 扩展示例
    ├── CMakeLists.txt
    ├── setup.py
    └── csrc/
        └── extension.cpp
```

## docs/ 文档目录

```
docs/
├── quick_start.md            # 快速开始指南
├── summary.md                # 项目整体介绍
├── directory_structure.md    # 项目目录结构
├── api/                      # API文档目录
├── tutorials/                # 教程
│   └── developer_guide.md    # 算子开发指南
└── images/                   # 文档配图
```

## tests/ 测试目录

```
tests/
├── CMakeLists.txt            # 测试构建配置
│
├── st/                       # 系统测试（System Tests）
│   ├── CMakeLists.txt
│   ├── test_op_*.cpp         # 各算子测试
│   ├── test_compute_*.cpp     # 计算相关测试
│   └── test_tile_*.cpp       # Tile 相关测试
│
└── ut/                       # 单元测试（Unit Tests）
    ├── host/                 # 主机端测试
    │   ├── test_arguments.cpp
    │   └── test_expr_linearizer.cpp
    ├── builtin_kernel/       # 内置 Kernel 测试
    └── compile_perf/         # 编译性能测试
```

## scripts/ 构建脚本目录

```
scripts/
└── build.sh                  # 主要构建脚本
                            # 支持参数：-DSOC=<arch>
                            # 示例：./scripts/build.sh -DSOC=ascend950 sin
```

## 相关文档
- [项目整体介绍](./summary.md) - 项目介绍
- [快速开始](./quick_start.md) - 快速开始
- [开发者指南](./tutorials/developer_guide.md) - 详细的开发流程
- [API文档](./api/README.md) - API文档
