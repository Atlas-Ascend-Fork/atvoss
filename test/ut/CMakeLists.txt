# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(atvoss_test)
cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)

set(PRODUCT_TYPE_LIST ascend910b
  CACHE STRING "Supported product types"
)

add_compile_definitions(__CCE_KT_TEST__=1 ASCENDC_CPU_DEBUG=1 ASCENDC_DUMP=0)

include(${ATVOSS_DIR}/cmake/intf_pub_llt.cmake)
include(${ATVOSS_DIR}/cmake/third_party/gtest.cmake)

set(ASCENDC_SYS_DIR ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux)

set(ASCEND_DEP_LIB_DIR
  ${ASCENDC_SYS_DIR}/lib64/
  ${ASCEND_DIR}/tools/tikicpulib/lib/
)

set(ATVOSS_TEST_HEADER_FILES
  ${UT_PATH}/common
  ${ATVOSS_DIR}
  ${ATVOSS_DIR}/include
  ${GTEST_INCLUDE}
  ${C_SEC_INCLUDE}
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/interface
  ${ASCENDC_SYS_DIR}/tikcpp/tikcfw/impl
  ${ASCENDC_SYS_DIR}/include
  ${ASCENDC_SYS_DIR}/include/base
  ${ASCEND_DIR}/tools/tikicpulib/lib/include
  ${ASCEND_DIR}/include
)

set(ATVOSS_TEST_ascend910b_HEADER_FILES
  ${ATVOSS_DIR}
)


set(ATVOSS_TEST_ascend910b_PARAM_CASE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/param/test_param.cpp
)

set(ATVOSS_TEST_ascend910b_KERNEL_RMS_NROM_CASE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_rms_norm.cpp
)
# 仅针对ATVOSS产品类型构建和运行测试
foreach(product_type ${PRODUCT_TYPE_LIST})
  add_executable(atvoss_${product_type}_ut
    ${UT_PATH}/main_global.cpp
    $<$<STREQUAL:${product_type},ascend910b>:${UT_PATH}/common/k3_pvwrap.cpp>
    ${ATVOSS_TEST_${product_type}_PARAM_CASE_SRC_FILES}
    ${ATVOSS_TEST_${product_type}_KERNEL_RMS_NROM_CASE_SRC_FILES}
  )

    # add soc version flags
    if(${product_type} STREQUAL "ascend910b")
        target_compile_definitions(atvoss_${product_type}_ut PRIVATE
            UT_TEST
            __CCE_AICORE__=220
            __NPU_ARCH__=2201
        )
        target_link_directories(atvoss_${product_type}_ut PRIVATE
            ${ASCEND_DEP_LIB_DIR}
            ${ASCEND_DIR}/tools/tikicpulib/lib/Ascend910B1/
            ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    endif()

    target_compile_definitions(atvoss_${product_type}_ut PRIVATE
        __CCE_KT_TEST__=1
        ASCENDC_CPU_DEBUG=1
        ASCENDC_OOM=1
    )

    target_include_directories(atvoss_${product_type}_ut PRIVATE
        ${ATVOSS_TEST_HEADER_FILES}
        ${ATVOSS_TEST_${product_type}_HEADER_FILES}
    )

    target_compile_options(atvoss_${product_type}_ut PRIVATE
        -fno-access-control
    )

    target_link_libraries(atvoss_${product_type}_ut PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        -Wl,--no-as-needed
        gtest
        gcov
        tikicpulib_npuchk
        tikicpulib_cceprint
        tikicpulib_stubreg
        tikcpp_debug
        $<$<STREQUAL:${product_type},ascend910b>:pem_davinci>
        c_sec
        error_manager
        -Wl,--as-needed
    )

  if(ENABLE_UT_EXEC)
    if(ENABLE_ASAN)
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE LIBASAN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE result
      )
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "compiler not support asan, please disable asan")
      endif()

      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE LIBSTDC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE stdc_result
      )
      if(NOT stdc_result EQUAL 0)
        message(FATAL_ERROR "not find libstdc++.so, please confirm.")
      endif()

      set(PRELOAD "LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}")
      add_custom_command(
        TARGET atvoss_${product_type}_ut POST_BUILD
        COMMAND
        COMMAND bash -c "source ${ASCEND_DIR}/bin/setenv.bash && \
        ${PRELOAD} ASAN_OPTIONS=detect_leaks=0 ./atvoss_${product_type}_ut"
        VERBATIM
        COMMENT "Run atvoss utest with asan"
      )
    else()
      add_custom_command(
        TARGET atvoss_${product_type}_ut POST_BUILD
        COMMAND bash -c "source ${ASCEND_DIR}/bin/setenv.bash && ./atvoss_${product_type}_ut"
        VERBATIM
        COMMENT "Run atvoss utest"
      )
    endif()

    if(ENABLE_COVERAGE)
      # set(_collect_coverage_data_target collect_coverage_data)
      get_filename_component(_atvoss_builtin_bin_path ${CMAKE_BINARY_DIR} DIRECTORY)
      set(_cov_report ${CMAKE_BINARY_DIR}/cov_report)
      set(_cov_html ${_cov_report})
      set(_cov_data ${_cov_report}/coverage.info)
      set(_generate_cov_script ${UT_PATH}/scripts/generate_cpp_cov.sh)

      add_custom_command(
          TARGET atvoss_${product_type}_ut POST_BUILD
          COMMAND bash ${_generate_cov_script} ${_atvoss_builtin_bin_path} ${_cov_data} ${_cov_html}
          COMMENT "Run collect coverage data"
      )
    endif()
  endif()
endforeach()
