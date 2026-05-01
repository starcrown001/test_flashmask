#! /bin/bash

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

# ------------------------------------------------------------------ #
#  MagiAttention-only distributed benchmark (CPU Timing)               #
#  Single-node, 8-GPU                                                  #
# ------------------------------------------------------------------ #

# ---- 分布式基础配置 ---- #
export MASTER_ADDR=${MASTER_ADDR:-10.52.98.148}
export MASTER_PORT=${MASTER_PORT:-16988}
export NNODES=4
export NPROC_PER_NODE=8
export RANK=0
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ---- MagiAttention 性能相关环境变量 ---- #
# 允许更多并发 CUDA kernel，提升 overlap 效果
export CUDA_DEVICE_MAX_CONNECTIONS=8
# NCCL 优化
export NCCL_CGA_CLUSTER_SIZE=1
export TORCH_NCCL_HIGH_PRIORITY=1
# MagiAttention 优化开关（默认关闭，如需开启改为 1）
export MAGI_ATTENTION_CATGQA=0
export MAGI_ATTENTION_AUTO_RANGE_MERGE=0
export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=0
export MAGI_ATTENTION_HIERARCHICAL_COMM=0   # 单机不需要分层通信
export MAGI_ATTENTION_NATIVE_GRPCOLL=0
export MAGI_ATTENTION_QO_COMM=0
export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=0
export MAGI_ATTENTION_FA4_BACKEND=0         # Hopper 上不使用 FA4
# export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# ---- Python 环境 ---- #
source /root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/activate
PYTHON=/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/python

# ---- 项目根目录（包含 magi_attention 包） ---- #
export PYTHONPATH=../../

# ---- 配置文件 & 输出 ---- #
CONFIG_PATH=${CONFIG_PATH:-"magi_benchmark_conf.py"}

# 解析 --config 参数（可覆盖默认值）
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config=*)
            CONFIG_PATH="${1#*=}"
            shift 1
            ;;
        --config)
            if [[ -n "$2" && "$2" != --* ]]; then
                CONFIG_PATH="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  MagiAttention Distributed Benchmark (CPU Timing)"
echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  CONFIG=$CONFIG_PATH"
echo "============================================================"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS benchmark_magiattention_cp_cpu.py --config "$CONFIG_PATH" --fast_eval true
