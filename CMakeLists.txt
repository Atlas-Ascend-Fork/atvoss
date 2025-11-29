# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME atvoss)
project(${PKG_NAME} VERSION 1.0.0)

include(CMakePrintHelpers)

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build atvoss with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build atvoss with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()
set(ATVOSS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 外部传参
option(ENABLE_ASAN "Enable asan" OFF)
option(ENABLE_TEST "Enable test" OFF)
option(ENABLE_UT_EXEC "Enable exec ut" OFF)
option(ENABLE_PACKAGE "Enable build package" OFF)

if(ENABLE_TEST)
  set(COMPILE_OP_MODE "-O0 -g")
endif()

# Cmake Compile配置
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
include(cmake/dependencies.cmake)
include(cmake/variables.cmake)
include(cmake/intf_pub_linux.cmake)

message(STATUS "Variables in ${CMAKE_PROJECT_NAME} project:")
cmake_print_variables(CMAKE_SOURCE_DIR)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
cmake_print_variables(CANN_3RD_LIB_PATH)
cmake_print_variables(CANN_3RD_PKG_PATH)
cmake_print_variables(ENABLE_PACKAGE)
cmake_print_variables(ENABLE_TEST)
cmake_print_variables(ENABLE_UT_EXEC)
cmake_print_variables(ENABLE_COVERAGE)
cmake_print_variables(ENABLE_ASAN)

if(ENABLE_TEST)
  add_subdirectory(test)
endif()

install(DIRECTORY include
  DESTINATION atvoss/
)

if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  install(DIRECTORY include
    DESTINATION lib/asc/atvoss/
  )
endif()

if(ENABLE_PACKAGE)
  include(cmake/package.cmake)
endif()
