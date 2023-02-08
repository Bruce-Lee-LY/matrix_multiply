# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 19:37:54 on Thu, Jun 02, 2022
#
# Description: compile script

#!/bin/bash

set -euo pipefail

echo "========== build enter =========="

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

BUILD_TYPE=Debug # t: (Debug, Release)
VERBOSE_MAKEFILE=OFF # b: (ON, OFF)

while getopts ":t:b:" opt
do
    case $opt in
        t)
        BUILD_TYPE=$OPTARG
        echo "BUILD_TYPE: $BUILD_TYPE"
        ;;
        b)
        VERBOSE_MAKEFILE=$OPTARG
        echo "VERBOSE_MAKEFILE: $VERBOSE_MAKEFILE"
        ;;
        ?)
        echo "invalid param: $OPTARG"
        exit 1
        ;;
    esac
done

echo_cmd() {
    echo $1
    $1
}

echo "========== build matrix_multiply =========="

echo_cmd "rm -rf build output"
echo_cmd "mkdir build"

echo_cmd "cd build"
echo_cmd "cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DMATRIX_VERBOSE_MAKEFILE=$VERBOSE_MAKEFILE -DCMAKE_INSTALL_PREFIX=$WORK_PATH/output -DCMAKE_SKIP_RPATH=ON .."
echo_cmd "make -j"
echo_cmd "make install"

echo "========== build info =========="

BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse HEAD`
GCC_VERSION=`gcc -dumpversion`
COMPILE_TIME=$(date "+%H:%M:%S %Y-%m-%d")

echo "branch: $BRANCH" >> $WORK_PATH/output/mm_version
echo "commit: $COMMIT" >> $WORK_PATH/output/mm_version
echo "gcc_version: $GCC_VERSION" >> $WORK_PATH/output/mm_version
echo "compile_time: $COMPILE_TIME" >> $WORK_PATH/output/mm_version

echo "========== build exit =========="
