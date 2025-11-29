#!/bin/sh
# Perform custom create softlink script for atvoss package
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
common_func_path="${curpath}/common_func.inc"

. "${common_func_path}"

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

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

create_stub_softlink() {
    local stub_dir="$1"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local arch_name="$2"
    local pwdbak="$(pwd)"
    cd $stub_dir && [ -d "$arch_name" ] && for so_file in $(find "$arch_name" -type f -o -type l); do
        ln -sf "$so_file" "$(basename $so_file)"
    done
    [ -d "linux/x86_64" ] && ln -snf "linux/x86_64" "x86_64"
    [ -d "linux/aarch64" ] && ln -snf "linux/aarch64" "aarch64"
    cd $pwdbak
}

do_create_stub_softlink() {
    local arch_name="$(get_arch_name $install_path/$version_dir/share/info/atvoss)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
}

do_create_stub_softlink
