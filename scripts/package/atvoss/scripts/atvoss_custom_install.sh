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

curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
ATVOSS_COMMON="${curpath}""/atvoss_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${ATVOSS_COMMON}"
. "${common_func_path}"
# init arch 
architecture=$(uname -m)
architecture_dir="${architecture}-linux"

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done
get_version_dir "atvoss_version_dir" "$install_path/$version_dir/atvoss/version.info"

# create atvoss soft link
logandprint "[INFO]: Start create atvoss softlinks."
create_atvoss_include_softlink "${install_path}/${version_dir}"
return_code=$?
if [ ${return_code} -eq 0 ]; then
    logandprint "[INFO]: Create atvoss softlinks successfully!"
elif [ ${return_code} -eq 3 ]; then
    logandprint "[WARNING]: atvoss source file does not exist!"
else
    logandprint "[ERROR]: Create atvoss softlinks failed!"
fi
