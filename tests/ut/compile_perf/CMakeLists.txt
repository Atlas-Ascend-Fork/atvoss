
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
message(STATUS "ASCEND_DIR=${ASCEND_DIR}")
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_LINKER ld)
add_executable(compile_perf_test compile_perf_test.cpp)
target_link_libraries(compile_perf_test PRIVATE
        gtest_main
        gtest
        pthread
)

target_compile_definitions(compile_perf_test PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=0
        -DASCEND_DIR="${ASCEND_DIR}"
        -DCASES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/cases"
        -DPROJECT_DIR="${PROJECT_SOURCE_DIR}"
        -DBINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
