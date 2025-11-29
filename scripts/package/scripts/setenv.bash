#!/usr/bin/env bash
# Perform setenv for atvoss package
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

curpath="$(dirname ${BASH_SOURCE:-$0})"
curfile="$(realpath ${BASH_SOURCE:-$0})"
DEP_HAL_NAME="libascend_hal.so"
DEP_INFO_FILE="/etc/ascend_install.info"
IS_INSTALL_DRIVER="n"
param_mult_ver=$1

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param=""

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    local install_info_key_array="Atvoss_Install_Type Atvoss_Feature_Type Atvoss_UserName Atvoss_UserGroup Atvoss_Install_Path_Param Atvoss_Arch_Linux_Path Atvoss_Hetero_Arch_Flag"
    for key_param in ${install_info_key_array}; do
        if [ "${key_param}" = "${_key}" ]; then
            _param=$(grep -i "${_key}=" "${_file}" | cut --only-delimited -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

get_install_dir() {
    local install_info="$curpath/../ascend_install.info"
    local hetero_arch=$(get_install_param "Atvoss_Hetero_Arch_Flag" "${install_info}")
    if [ "$param_mult_ver" = "multi_version" ]; then
        if [ "$hetero_arch" = "y" ]; then
            echo "$(realpath $curpath/../../../../../cann)/atvoss"
        else
            echo "$(realpath $curpath/../../../cann)/atvoss"
        fi
    else
        echo "$(realpath $curpath/..)"
    fi
}

INSTALL_DIR="$(get_install_dir)"
lib_stub_path="${INSTALL_DIR}/lib64/stub"
lib_path="${INSTALL_DIR}/lib64"
ld_library_path="$LD_LIBRARY_PATH"
if [ ! -z "$ld_library_path" ]; then
    ld_library_path="$(echo "$ld_library_path" | tr ':' ' ')"
    for var in ${ld_library_path}; do
        if [ -d "$var" ]; then
            if echo "$var" | grep -q "driver"; then
                num=$(find "$var" -name ${DEP_HAL_NAME} 2> /dev/null | wc -l)
                if [ "$num" -gt "0" ]; then
                    IS_INSTALL_DRIVER="y"
                fi
            fi
        fi
    done
fi

# 第一种方案判断驱动包是否存在
if [ -f "$DEP_INFO_FILE" ]; then
    driver_install_path_param="$(grep -iw driver_install_path_param $DEP_INFO_FILE | cut --only-delimited -d"=" -f2-)"
    if [ ! -z "${driver_install_path_param}" ]; then
        DEP_PKG_VER_FILE="${driver_install_path_param}/driver"
        if [ -d "${DEP_PKG_VER_FILE}" ]; then
            DEP_HAL_PATH=$(find "${DEP_PKG_VER_FILE}" -name "${DEP_HAL_NAME}" 2> /dev/null)
            if [ ! -z "${DEP_HAL_PATH}" ]; then
                IS_INSTALL_DRIVER="y"
            fi
        fi
    fi
fi

# 第二种方案判断驱动包是否存在
echo ":$PATH:" | grep ":/sbin:" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    export PATH="$PATH:/sbin"
fi

which ldconfig > /dev/null 2>&1
if [ $? -eq 0 ]; then
    ldconfig -p | grep "${DEP_HAL_NAME}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        IS_INSTALL_DRIVER="y"
    fi
fi

if [ "${IS_INSTALL_DRIVER}" = "n" ]; then
    if [ -d "${lib_path}" ]; then
        ld_library_path="${LD_LIBRARY_PATH}"
        num=$(echo ":${ld_library_path}:" | grep ":${lib_stub_path}:" | wc -l)
        if [ "${num}" -eq 0 ]; then
            if [ "-${ld_library_path}" = "-" ]; then
                export LD_LIBRARY_PATH="${lib_stub_path}"
            else
                export LD_LIBRARY_PATH="${ld_library_path}:${lib_stub_path}"
            fi
        fi
    fi
fi

if [ -d "${lib_path}" ]; then
    ld_library_path="${LD_LIBRARY_PATH}"
    num=$(echo ":${ld_library_path}:" | grep ":${lib_path}:" | wc -l)
    if [ "${num}" -eq 0 ]; then
        if [ "-${ld_library_path}" = "-" ]; then
            export LD_LIBRARY_PATH="${lib_path}"
        else
            export LD_LIBRARY_PATH="${lib_path}:${ld_library_path}"
        fi
    fi
fi

custom_path_file="$INSTALL_DIR/../conf/path.cfg"
common_interface="$INSTALL_DIR/script/common_interface.bash"
owner=$(stat -c %U "$curfile")
if [ $(id -u) -ne 0 ] && [ "$owner" != "$(whoami)" ] && [ -f "$custom_path_file" ] && [ -f "$common_interface" ]; then
    . "$common_interface"
    mk_custom_path "$custom_path_file"
fi


prepend_env() {
    local name="$1"
    local value="$2"
    local env_value="$(eval echo "\${${name}}" | tr ':' '\n' | grep -v "^${value}$" | tr '\n' ':' | sed 's/:$/\n/')"
    if [ "$env_value" = "" ]; then
        read $name <<EOF
$value
EOF
    else
        read $name <<EOF
$value:$env_value
EOF
    fi
    export $name
}

asc_opc_tool_path="$(realpath $curpath/../../)/compiler/bin"
if [ -d ${asc_opc_tool_path} ]; then
    prepend_env PATH "$asc_opc_tool_path"
fi
