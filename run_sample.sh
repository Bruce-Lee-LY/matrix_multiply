# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:56:32 on Thu, Jun 02, 2022
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log && mkdir -p log/cpu log/cuda

# cpu
nohup $WORK_PATH/output/cpu/matrix_multiply_cpu > log/cpu/matrix_multiply_cpu.log 2>&1 &

# nvidia gpu
nohup $WORK_PATH/output/cuda/matrix_multiply_cuda > log/cuda/matrix_multiply_cuda.log 2>&1 &
