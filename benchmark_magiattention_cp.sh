#! /bin/bash

# Copyright (c) 2025 SandAI. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# export CUDA_VISIBLE_DEVICES="0,4"
export WORLD_SIZE=${WORLD_SIZE:-16}
export GPUS_PER_NODE=8
export NNODES=${NNODES:-2}
export NODE_RANK=${PADDLE_TRAINER_ID:-1}
export MAGI_ATTENTION_HIERARCHICAL_COMM=${MAGI_ATTENTION_HIERARCHICAL_COMM:-1}
export MASTER_ADDR=${MASTER_ADDR:-10.54.107.140}
export MASTER_PORT=${MASTER_PORT:-17988}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [ "${MAGI_ATTENTION_HIERARCHICAL_COMM}" == "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=8
    echo "set CUDA_DEVICE_MAX_CONNECTIONS=8"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo "set CUDA_DEVICE_MAX_CONNECTIONS=1"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

echo $DISTRIBUTED_ARGS

# TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"
CMD="/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/torchrun $DISTRIBUTED_ARGS /root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/benchmark_magiattention_cp.py"
# TORCHRUN_CMD="/opt/nvidia/nsight-systems/2023.2.1/bin/nsys profile \
#     --force-overwrite true \
#     -o magi_32kcp4.nsys-rep \
#     --capture-range=cudaProfilerApi \
#     $CMD"
# $TORCHRUN_CMD
$CMD
