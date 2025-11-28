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

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

Atvoss_HOME_DIR=$CURRENT_DIR/../
TEST_CASE_LIST=$(ls $Atvoss_HOME_DIR/examples | grep -v '^run_examples.sh$' | grep -v '^ops_*' | grep -v '^common*' | grep -v '^README*'  | xargs)
if [ $# -lt 1 ]; then
    echo "This script requires an input as the test case name. Execution example: 'bash run_examples.sh [$TEST_CASE_LIST]'"
    exit 1
fi
TEST_NAME=$1

# 调试模式设置
RUN_MODE=""
function parse_run_mode(){
    for arg in "$@"; do
        if [[ "$@" =~ --run-mode=([a-zA-Z0-9_]+) ]]; then
            RUN_MODE="${BASH_REMATCH[1]}"
            return
        fi
    done
}

# 根据不同run-mode执行不同的操作
function compile_operator(){
    if [ -n "$ASCEND_INSTALL_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
            _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
        else
            _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
        fi
    fi
    cd $Atvoss_HOME_DIR/examples/$TEST_NAME
    if [ -z "$RUN_MODE" ]; then
        echo "Executing with npu mode"
        bisheng -x asc --npu-arch=dav-2201 $TEST_NAME.cpp -o $TEST_NAME -I ${Atvoss_HOME_DIR}/include -I ${CURRENT_DIR}/common -ltiling_api -lplatform -lm -ldl -L${_ASCEND_INSTALL_PATH}/lib64 -w
    elif [ "$RUN_MODE" = "profiling" ]; then
        echo "Executing with profiling mode"
        bisheng -x asc --npu-arch=dav-2201 $TEST_NAME.cpp -o $TEST_NAME -I ${Atvoss_HOME_DIR}/include -I ${CURRENT_DIR}/common  -ltiling_api -lplatform -lm -ldl -L${_ASCEND_INSTALL_PATH}/lib64 -w -DAtvoss_DEBUG_MODE=2 -DASCENDC_DUMP=0
    else
        echo "--run-mode is an optional parameter and can be left unset. If set, the value must be profiling."
        echo "Execution example: 'bash run_examples.sh $TEST_NAME --run-mode=profiling'"
        exit 1
    fi
}

if [[ " $TEST_CASE_LIST " == *" ${TEST_NAME} "* ]]; then
    cd $Atvoss_HOME_DIR/examples/$TEST_NAME
    rm -rf ./$TEST_NAME
    parse_run_mode "$@"
    compile_operator
    if [ ! -f ./$TEST_NAME ]; then
        echo "Error: Cannot find file ./${TEST_NAME} due to compilation error, please check error message."
        exit 1
    fi
    if [ "$RUN_MODE" = "profiling" ]; then
        msprof --ai-core=on --ascendcl=on --model-execution=on --runtime-api=on --task-time=on --application=$TEST_NAME --output=./
    else
        ./$TEST_NAME
    fi
    if [ $? -eq 0 ]; then
        echo "Sample ${TEST_NAME} passed!"
    else
        echo "Sample ${TEST_NAME} failed!"
    fi
    cd ${Atvoss_HOME_DIR}
else
    echo "Error: Cannot find '$TEST_NAME' in ${Atvoss_HOME_DIR}examples. Execution example: 'bash run_examples.sh [$TEST_CASE_LIST]'"
    exit 1
fi