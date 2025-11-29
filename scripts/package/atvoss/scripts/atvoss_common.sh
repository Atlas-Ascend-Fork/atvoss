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

if [ "$(id -u)" != "0" ]; then
  _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
else
  _LOG_PATH="/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
fi

# log functions
getdate() {
  _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${_cur_date}"
}

logandprint() {
  is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
  if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
    echo "[ATVOSS] [$(getdate)] ""$1"
  fi
  echo "[ATVOSS] [$(getdate)] ""$1" >>"${_INSTALL_LOG_FILE}"
}

# create soft link
create_relatively_softlink() {
  local src_path_="$1"
  local dst_path_="$2"
  local dst_parent_path_=$(dirname ${dst_path_})
  # echo "dst_parent_path_: ${dst_parent_path_}"
  local relative_path_=$(realpath --relative-to="$dst_parent_path_" "$src_path_")
  # echo "relative_path_: ${relative_path_}"
  if [ -L "$2" ]; then
    return 0
  fi
  ln -s "${relative_path_}" "${dst_path_}" 2>/dev/null
  if [ "$?" != "0" ]; then
    return 1
  else
    return 0
  fi
}

create_atvoss_include_softlink() {
  osName=""
  if [ -f "$1/atvoss/scene.info" ]; then
    . $1/atvoss/scene.info
    osName=${os}
  fi
  atvoss_include_src_path="$1/${architecture_dir}/asc/atvoss/include"

  if [ ! -d ${atvoss_include_src_path} ]; then
    return 3
  fi

  if [ -d $1/${architecture_dir}/ascendc/act ]; then
    atvoss_include_dst_path="$1/${architecture_dir}/ascendc/act/include"
    if [ -d $atvoss_include_dst_path ] || [ -L $atvoss_include_dst_path ]; then
      rm -fr "$atvoss_include_dst_path"
    fi
    create_relatively_softlink ${atvoss_include_src_path} ${atvoss_include_dst_path}
  fi
}

# remove soft link
remove_relatively_softlink() {
  local path="$1"
  if [ -L "$1" ]; then
    [ -n "${path}" ] && rm -fr ${path}
    return 0
  else
    return 1
  fi
}

remove_atvoss_include_softlink() {
  targetdir=$1
  osName=""
  if [ -f "$targetdir/atvoss/scene.info" ]; then
    . $targetdir/atvoss/scene.info
    osName=${os}
  fi

  if [ -d $targetdir/${architecture_dir}/ascendc/act ]; then
    atvoss_include_path="$targetdir/${architecture_dir}/ascendc/act/include"
    remove_relatively_softlink ${atvoss_include_path}
  fi
}
