# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

include(${ATVOSS_DIR}/cmake/third_party/gtest.cmake)

set(ATVOSS_TEST_HEADER_FILES
  ${ATVOSS_DIR}/include
  ${ASCEND_INCLUDE_DIRS}
  ${ATVOSS_DIR}/examples/common
  ${GTEST_INCLUDE}
)

# ==============================
# Step 1: 自动收集所有 .cpp 测试文件
# ==============================
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

# ==============================
# Step 2: 为每个 .cpp 创建独立的 test target
# ==============================
set(ALL_UT_TARGETS "")  # 用于收集所有单测 target 名

foreach(test_src IN LISTS TEST_SOURCES)
    # 获取文件名（不含路径和后缀），如 "test_arguments.cpp" → "test_arguments"
    get_filename_component(target_name ${test_src} NAME_WE)

    # 添加可执行文件
    add_executable(${target_name} ${test_src})

    # 设置编译定义
    target_compile_definitions(${target_name} PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=0
        -DASCEND_DIR="${ASCEND_DIR}"
        -DPROJECT_DIR="${PROJECT_SOURCE_DIR}"
        -DBINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    # 设置头文件目录
    target_include_directories(${target_name} PRIVATE ${ATVOSS_TEST_HEADER_FILES})

    # 设置编译选项
    target_compile_options(${target_name} PRIVATE
        -fdiagnostics-color=always
        -w
        --npu-arch=dav-2201
        -xasc
    )

    # 链接库
    target_link_libraries(${target_name} PRIVATE
        gtest_main
        gtest
        pthread
        m
    )

    list(APPEND ALL_UT_TARGETS ${target_name})
endforeach()

# ==============================
# Step 3: 创建聚合 target：host_ut（全量运行）
# ==============================
add_custom_target(host_ut
    DEPENDS ${ALL_UT_TARGETS}
    COMMENT "Building all host unit tests"
)

# 添加一个脚本，在构建 host_ut 后自动运行所有测试
add_custom_command(
    TARGET host_ut POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running all host unit tests ==="
)

foreach(test IN LISTS ALL_UT_TARGETS)
    add_custom_command(
        TARGET host_ut POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "=== Running ${test} ==="
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endforeach()



