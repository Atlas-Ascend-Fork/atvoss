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
SUPPORTED_SHORT_OPTS="hj:vt-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "verbose" "pkg" "make_clean" "cann_3rd_lib_path" "test" "noexec" "cov" "disable_asan" "cann_3rd_lib_path"
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
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg -j16"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean           Clean build artifacts"
        echo $dotted_line
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -t, --test             Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -t --noexec"
        echo "    bash build.sh --test --cov"
        return
        ;;
    esac
  fi

  echo "build script for atvoss repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-t] [--pkg]"
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
  echo "    -t, --test                   Compile all ut"
  echo $dotted_line
  echo "    examples, Build unit test and do not execute."
  echo "    ./build.sh --test --noexec"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    --make_clean                 Make clean"
  echo "    --noexec                     Only compile ut, do not execute the compiled executable file"
  echo "    --cov                        When building uTest locally, count the coverage"
  echo "    --disable_asan               Disable ASAN (Address Sanitizer)"
  echo "to be continued ..."
}

check_help_combinations() {
  local args=("$@")
  local has_t=false
  local has_package=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -t | --test) has_t=true ;;
      --pkg) has_package=true ;;
      --help | -h) ;;
    esac
  done

  # Check the invalid command combinations in help
  if [[ "$has_package" == "true" && "$has_t" == "true" ]]; then
    echo "[ERROR] --pkg cannot be used with test(-t, --test)"
    return 1
  fi
  return 0
}

check_param() {
  # -pkg不能与-t（UT模式）同时存在
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with test(-t, --test)"
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

  ENABLE_COVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=TRUE
  ENABLE_PACKAGE=FALSE
  ENABLE_TEST=FALSE

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
          -t | --test) SHOW_HELP="test" ;;
          --make_clean) SHOW_HELP="clean" ;;
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
      t) ENABLE_TEST=TRUE ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        test) ENABLE_TEST=TRUE ;;
        cov) ENABLE_COVERAGE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        pkg) ENABLE_PACKAGE=TRUE ;;
        cann_3rd_lib_path=*) CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})" ;;
        disable_asan) ENABLE_ASAN=FALSE ;;
        make_clean)
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
  # if [[ "$ENABLE_COVERAGE" =~ "TRUE" ]]; then
  #   cmake --build . --target generate_atcos_cpp_cov -- -j $THREAD_NUM
  # fi
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
}

set -o pipefail
if [ $# -eq 0 ]; then
  usage
  exit 1
fi
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
