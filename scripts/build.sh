#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -o errexit
set -o nounset
set -o pipefail

NC="\033[0m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"

ERROR="${RED}[ERROR]"
INFO="${GREEN}[INFO]"
WARN="${YELLOW}[WARN]"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CMAKE_SOURCE_DIR=$(realpath "$SCRIPT_DIR/..")
BUILD_DIR="$CMAKE_SOURCE_DIR/build"
OUTPUT_DIR="$CMAKE_SOURCE_DIR/output"

MODE=""          # "host_ut", "example", "st", or "legacy"
TARGET_NAME=""
CMAKE_BUILD_TYPE="Release"
declare -a CMAKE_OPTIONS=()
CLEAN=false
POST_BUILD_INFO=""

echo -e "     _  _______     _____  ____ ____  "
echo -e "    / \|_   _\ \   / / _ \/ ___/ ___| "
echo -e "   / _ \ | |  \ \ / / | | \___ \___ \ "
echo -e "  / ___ \| |   \ V /| |_| |___) |__) |"
echo -e " /_/   \_\_|    \_/  \___/|____/____/ "

function show_help() {
    echo -e "${GREEN}Usage:${NC} $0 [options] [--host_ut|--example|--st] [target_name]"
    echo -e "\n${BLUE}Options:${NC}"
    echo "  --clean         Clean build directories"
    echo "  -DSOC  NPU arch. Only supports ascend950."
    echo "  -D<option>      Additional CMake options"
    echo -e "\n${BLUE}Modes (mutually exclusive):${NC}"
    echo "  --host_ut [name]     Build unit test(s) in host/ (default: host_ut)"
    echo "  --example [name] Build example(s) (default: atvoss_examples)"
    echo "  --st [name]     Build system test (default: st)"
    echo -e "\n${BLUE}Legacy mode:${NC}"
    echo "  $0 <target>     Directly build CMake target (e.g., abs, host_ut)"
}

if [[ $# -eq 0 ]] || [[ "$1" = "-h" ]] || [[ "$1" = "--help" ]]; then
    show_help
    exit 0
fi

if [[ ! -v ASCEND_HOME_PATH ]]; then
    echo -e "${ERROR}ASCEND_HOME_PATH environment variable is not set!${NC}"
    echo -e "${ERROR}Please set ASCEND_HOME_PATH before running this script.${NC}"
    exit 1
fi

# ----------------------------
# 参数解析
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        -D*)
            CMAKE_OPTIONS+=("$1")
            shift
            ;;
        --host_ut|--example|--st)
            if [[ -n "$MODE" ]]; then
                echo -e "${ERROR}Error: Only one mode (--host_ut/--example/--st) allowed.${NC}" >&2
                exit 1
            fi
            MODE="${1#--}"
            shift
            if [[ $# -gt 0 && "$1" != -* ]]; then
                TARGET_NAME="$1"
                shift
            fi
            ;;
        -*)
            echo -e "${ERROR}Unknown option: $1${NC}" >&2
            show_help
            exit 1
            ;;
        *)
            if [[ -n "$MODE" ]]; then
                echo -e "${ERROR}Cannot mix mode and direct target.${NC}" >&2
                exit 1
            fi
            TARGET_NAME="$1"
            MODE="legacy"
            shift
            ;;
    esac
done

# ----------------------------
# 清理
# ----------------------------
if [[ "$CLEAN" == true ]]; then
    echo -e "${INFO}Cleaning build directories...\c"
    rm -rf "$BUILD_DIR" "$OUTPUT_DIR"
    echo -e "Complete!${NC}"
    exit 0
fi

# ----------------------------
# 校验并确定 CMake target
# ----------------------------
if [[ -z "$MODE" ]]; then
    echo -e "${ERROR}No mode or target specified.${NC}" >&2
    show_help
    exit 1
fi

CMAKE_TARGET=""
case "$MODE" in
    host_ut)
        CMAKE_TARGET=${TARGET_NAME:-host_ut}
        ;;
    example)
        CMAKE_TARGET=${TARGET_NAME:-atvoss_examples}
        ;;
    st)
        CMAKE_TARGET=${TARGET_NAME:-st}
        ;;
    legacy)
        CMAKE_TARGET="$TARGET_NAME"
        ;;
esac

# ----------------------------
# 构建准备
# ----------------------------
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

# 仅当未配置时运行 cmake
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo -e "${INFO}Configuring project...${NC}"
    if [[ "$CMAKE_TARGET" == "abs" ]] || [[ "$CMAKE_TARGET" == "atvoss_examples" ]]; then
        cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
            -DCOMPILE_DYNAMIC_OPTIMIZED_MATMUL=ON \
            "${CMAKE_OPTIONS[@]}"
    else
        cmake -S "$CMAKE_SOURCE_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
            "${CMAKE_OPTIONS[@]}"
    fi
fi

# ----------------------------
# 执行构建
# ----------------------------
echo -e "${INFO}Building target: $CMAKE_TARGET (${MODE}${TARGET_NAME:+: $TARGET_NAME})...${NC}"
if ! cmake --build "$BUILD_DIR" --target "$CMAKE_TARGET" -j; then
    echo "构建失败，终止。"
    exit 1
fi

if ! cmake --install "$BUILD_DIR" --component "$CMAKE_TARGET"; then
    echo "构建失败，终止。"
    exit 1
fi

POST_BUILD_INFO="${INFO}Target '$CMAKE_TARGET' built successfully.${NC}"

# ----------------------------
# 单个 UT 编译后运行
# ----------------------------
if [[ "$MODE" == "host_ut" && -n "$TARGET_NAME" ]]; then
    UT_EXECUTABLE="$BUILD_DIR/tests/ut/host/$TARGET_NAME"
    if [[ -x "$UT_EXECUTABLE" ]]; then
        echo -e "\n${INFO}Running unit test: $TARGET_NAME${NC}"
        $UT_EXECUTABLE
    fi
fi

echo -e "\n$POST_BUILD_INFO"

# ----------------------------
# examples 仿真运行
# ----------------------------
if [[ "$TARGET_NAME" == "atvoss_examples" ]]; then

    ST_TEST_LIST=(
        "abs"
        "muls"
        "rms_norm"
    )

    for test_name in "${ST_TEST_LIST[@]}"; do
        echo "Running system test: $test_name"

        UT_EXECUTABLE="$CMAKE_SOURCE_DIR/output/bin/$test_name"

        # 检查可执行文件是否存在
        if [[ ! -f "$UT_EXECUTABLE" || ! -x "$UT_EXECUTABLE" ]]; then
            echo "Error: $UT_EXECUTABLE not found or not executable."
            exit 1
        fi

        SCRIPT_FILE="${test_name}.sh"
        cat > "$SCRIPT_FILE" <<EOF
$UT_EXECUTABLE --shape=32
EOF
        chmod +x "$SCRIPT_FILE"
        cannsim record  ./$SCRIPT_FILE -s Ascend950 --gen-report

        # TODO 
        cat ./cannsim*${test_name}*/cannsim.log
        # 检查日志中是否包含成功标志
        if grep -r "Accuracy verification passed" ./cannsim*${test_name}*; then
            echo "$test_name passed."
        else
            echo "$test_name failed: 'Accuracy verification passed' not found."
            exit 1
        fi
    done
    echo "All system tests passed!"
fi

# ----------------------------
# st 仿真运行
# ----------------------------
if [[ "$MODE" == "st" && ! -n "$TARGET_NAME" ]]; then
    ST_TEST_LIST=(
        "test_block_cast12"
        "test_tile_rms_norm_14"
        "test_compute_buffer_reuse"
    )    
    for test_name in "${ST_TEST_LIST[@]}"; do
        echo "Running system test: $test_name"

        UT_EXECUTABLE="$CMAKE_SOURCE_DIR/output/bin/$test_name"

        # 检查可执行文件是否存在
        if [[ ! -f "$UT_EXECUTABLE" || ! -x "$UT_EXECUTABLE" ]]; then
            echo "Error: $UT_EXECUTABLE not found or not executable."
            exit 1
        fi

        SCRIPT_FILE="${test_name}.sh"
        cat > "$SCRIPT_FILE" <<EOF
$UT_EXECUTABLE
EOF
        chmod +x "$SCRIPT_FILE"
        cannsim record  ./$SCRIPT_FILE -s Ascend950 --gen-report

        # 检查日志中是否包含成功标志
        if grep -r "Accuracy verification passed" ./cannsim*${test_name}*; then
            echo "$test_name passed."
        else
            echo "$test_name failed: 'Accuracy verification passed' not found."
            exit 1
        fi
    done
    echo "All system tests passed!"    
fi