#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
UT_TARGETS=("atvoss_ascend910b_ut")
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93")

# 所有支持的短选项
SUPPORTED_SHORT_OPTS="hj:vcu-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "verbose" "pkg" "clean" "cann_3rd_lib_path" "utest" "noexec" "cov" "asan" "cann_3rd_lib_path"
  "run_example" "build-type"
)

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

check_option_validity() {
  local arg="$1"

  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}

    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')

    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"

      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        echo "[ERROR] Invalid short option: -$char"
        return 1
      fi

      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"

    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      # with "=" in long options
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        # without "=" in long options
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done

    echo "[ERROR] Invalid long option: --$opt_name"
    return 1
  fi

  return 0
}

dotted_line="----------------------------------------------------------------"
export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="atvoss"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"

# print usage message
usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      package)
        echo "Package Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run package"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    --cann_3rd_lib_path=<PATH>"
        echo "                           Set ascend third_party package install path, default ./third_party"
        echo "    --build-type=<TYPE>    Specify build type (TYPE options: Release/Debug), Default:Release"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg -j16"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    -c, --clean            Clean build artifacts"
        echo $dotted_line
        return
        ;;
      utest)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u, --utest            Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute, only can bu used with -u"
        echo "    --cov                  Enable code coverage for unit tests, only can bu used with -u"
        echo "    --asan                 Enable ASAN (Address Sanitizer)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u --noexec"
        echo "    bash build.sh --utest --cov"
        return
        ;;
      run_example)
        echo "Run examples Options:"
        echo $dotted_line
        echo "    --run_example example_name   Compile and execute example"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example 00_basic_matmul"
        echo "    bash build.sh --run_example 00_basic_matmul,01_misplace_core_matmul"
        echo "    bash build.sh --run_example all"
        return
        ;;
    esac
  fi

  echo "build script for atvoss repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-u] [--pkg]"
  echo ""
  echo ""
  echo "Options:"
  echo $dotted_line
  echo "    Build parameters "
  echo $dotted_line
  echo "    -h                           Print usage"
  echo "    -j[n]                        Compile thread nums, default is 8, eg: -j8"
  echo "    -v                           Cmake compile verbose"
  echo "    --pkg                        Build run package"
  echo "    -u, --utest                  Compile all ut"
  echo $dotted_line
  echo "    examples, Build unit test and do not execute."
  echo "    ./build.sh --utest --noexec"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    -c, --clean                  Make clean"
  echo "    --noexec                     Only compile ut, do not execute the compiled executable file, only can bu used with -u"
  echo "    --cov                        When building uTest locally, count the coverage, only can bu used with -u"
  echo "    --asan                       Enable ASAN (Address Sanitizer)"
  echo "    --run_example                Compile and execute example"
  echo "to be continued ..."
}

check_help_combinations() {
  local args=("$@")
  local has_t=false
  local has_package=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u | --utest) has_t=true ;;
      --pkg) has_package=true ;;
      --help | -h) ;;
    esac
  done

  # Check the invalid command combinations in help
  if [[ "$has_package" == "true" && "$has_t" == "true" ]]; then
    echo "[ERROR] --pkg cannot be used with test(-u, --utest)"
    return 1
  fi
  return 0
}

check_param() {
  # -pkg不能与-t（UT模式）同时存在
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with test(-u, --utest)"
      exit 1
    fi
  fi
  # --noexec需要与-t（UT模式）同时存在
  if [[ "$ENABLE_UT_EXEC" == "FALSE" ]]; then
    if [[ "$ENABLE_TEST" == "FALSE" ]]; then
      echo "[ERROR] --noexec must be used with test(-u, --utest)"
      exit 1
    fi
  fi
  # --cov需要与-t（UT模式）同时存在
  if [[ "$ENABLE_COVERAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "FALSE" ]]; then
      echo "[ERROR] --cov must be used with test(-u, --utest)"
      exit 1
    fi
  fi
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

checkopts() {
  THREAD_NUM=8
  VERBOSE=""
  UT_TEST_ALL=FALSE
  SHOW_HELP=""
  EXAMPLE_NAME=""

  ENABLE_COVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_TEST=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  BUILD_TYPE="Release"

  # 首先检查所有参数是否合法
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then # 只检查以-开头的参数
      if ! check_option_validity "$arg"; then
        echo "Use 'bash build.sh --help' for more information."
        exit 1
      fi
    fi
  done

  # 检查并处理--help
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      # 检查帮助信息中的组合参数
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"

      # 检查 --help 前面的命令
      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="package" ;;
          -u | --utest) SHOW_HELP="utest" ;;
          -c | --clean) SHOW_HELP="clean" ;;
          --run_example) SHOW_HELP="run_example" ;;
        esac
      done

      usage "$SHOW_HELP"
      exit 0
    fi
  done

  # Process the options
  while getopts $SUPPORTED_SHORT_OPTS opt; do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      u) ENABLE_TEST=TRUE ;;
      c)
        clean_build
        clean_build_out
        exit 0
        ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        utest) ENABLE_TEST=TRUE ;;
        cov) ENABLE_COVERAGE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        pkg) ENABLE_PACKAGE=TRUE ;;
        cann_3rd_lib_path=*) CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})" ;;
        asan) ENABLE_ASAN=TRUE ;;
        run_example) ENABLE_RUN_EXAMPLE=TRUE ;;
        build-type=*) BUILD_TYPE="${OPTARG#build-type=}" ;;
        clean)
          clean_build
          clean_build_out
          exit 0
          ;;
        *)
          echo "[ERROR] Invalid option: --$OPTARG"
          usage
          exit 1
          ;;
      esac ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
        ;;
    esac
  done

  if [[ "$1" == "--run_example" ]]; then
    EXAMPLE_NAME="$2"
  fi

  check_param
}

assemble_cmake_args() {
  if [[ "$ENABLE_ASAN" == "TRUE" ]]; then
    set +e
    echo 'int main() {return 0;}' | gcc -x c -fsanitize=address - -o asan_test >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "This environment does not have the ASAN library, no need enable ASAN"
      ENABLE_ASAN=FALSE
    else
      $(rm -f asan_test)
      CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=TRUE"
    fi
    set -e
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=${ENABLE_PACKAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=${ENABLE_TEST}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=${ENABLE_UT_EXEC}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=${ENABLE_COVERAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}/*
  fi
}

build_package() {
  echo "--------------- build package start ---------------"
  clean_build_out

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM
  echo "--------------- build package end ---------------"
}

build_ut() {
  echo $dotted_line
  echo "Start to build ut"
  clean_build

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target ${UT_TARGETS[@]} -- ${VERBOSE} -j $THREAD_NUM
}

build_example() {
  return # 当前子包bisheng有编译问题，暂时下线
  echo $dotted_line
  echo "Start to run examples,name:${EXAMPLE_NAME}"
  clean_build

  if [[ "${EXAMPLE_NAME}" == "" ]]; then
    echo "Failed to get examples"
    exit 1
  fi

  example_list=""
  if [[ "${EXAMPLE_NAME}" == *","* ]]; then
    example_list=(${EXAMPLE_NAME//,/ })
  elif [[ "${EXAMPLE_NAME}" == "all" ]]; then
    example_list=(00_basic_matmul 01_misplace_core_matmul 02_batch_matmul 03_quant_matmul
                  04_l2_misplace_core_matmul 05_l2_misplace_core_batchmatmul
                  06_l2_misplace_core_quant_matmul 07_naive_matmul 08_sparse_matmul)
  elif [[ "${EXAMPLE_NAME}" == "smoke" ]]; then
      example_list=(00_basic_matmul 05_l2_misplace_core_batchmatmul
                    06_l2_misplace_core_quant_matmul 07_naive_matmul 08_sparse_matmul)
  else
    example_list=("$EXAMPLE_NAME")
  fi
  echo "example_list : ${example_list[@]}"

  if [[ -n "${ASCEND_HOME_PATH}" ]]; then
    echo "env exists ASCEND_HOME_PATH : ${ASCEND_HOME_PATH}"
    export PATH=${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/ascc/:$PATH
    env| grep PATH
  elif [[ $EUID -eq 0 ]]; then
    if [[ -d "/usr/local/Ascend/ascend-toolkit/cann" ]]; then
      export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/cann
    else
      export ASCEND_HOME_PATH=/usr/local/Ascend/cann
    fi
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
  else
    if [[ -d "${HOME}/Ascend/ascend-toolkit/cann" ]]; then
      export ASCEND_HOME_PATH=${HOME}/Ascend/ascend-toolkit/cann
    else
      export ASCEND_HOME_PATH=${HOME}/Ascend/cann
    fi
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
  fi

  echo "Start to examples"
  for i in "${!example_list[@]}"; do
    echo $dotted_line
    example_name=${example_list[$i]}
    example_dir="${BASE_PATH}/examples/${example_name}"
    echo "Start to run example ${example_name}"
    export ASCEND_PROCESS_LOG_PATH="${example_dir}/log"
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    export ASCEND_GLOBAL_LOG_LEVEL=3
    cd "${example_dir}" || { echo "Failed to enter directory ${UT_PATH}"; exit 1; }
    output_dir="${example_dir}/output"
    if [[ -d "${output_dir}" ]]; then
      rm -rf "${output_dir}"
    fi
    bash run.sh -r npu -p 0
    test_result=0
    cat ${output_dir}/*.csv | grep "Fail\|None" || test_result=1
    if [[ $test_result -eq 0 ]]; then
      echo "run example ${example_name} failed."
      grep -rn ERROR "${example_dir}/log"
      exit 1
    fi
    echo "run example ${example_name} successful."
  done
}

main() {
  checkopts "$@"
  if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
    echo "compile thread num:$THREAD_NUM over core num:$CORE_NUMS, adjust to core num"
    THREAD_NUM=$CORE_NUMS
  fi
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    build_package
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
  fi
  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    build_example
  fi
}

set -o pipefail
if [ $# -eq 0 ]; then
  usage
  exit 1
fi
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
