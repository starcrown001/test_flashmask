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
#  TE (Transformer Engine) CP Attention Benchmark                     #
#  Single-node or Multi-node, 8-GPU per node                          #
# ------------------------------------------------------------------ #

# ---- 分布式基础配置 ---- #
export MASTER_ADDR=${MASTER_ADDR:-10.52.98.148}
export MASTER_PORT=${MASTER_PORT:-16988}
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export RANK=${RANK:-0}
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ---- CUDA/NCCL 优化 ---- #
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_CGA_CLUSTER_SIZE=1
export TORCH_NCCL_HIGH_PRIORITY=1

# ---- Python 环境 ---- #
source /root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/activate
PYTHON=/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/python

# ---- 确保使用 venv 内 pip 安装的 cuDNN/CUDA 库 ---- #
NVIDIA_LIB_DIR="/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}/cudnn/lib:${NVIDIA_LIB_DIR}/cublas/lib:${NVIDIA_LIB_DIR}/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# ---- 项目根目录 ---- #
# 当前测试目录（提供 te_baselines, sparsity_utils 等本地模块）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- 默认参数 ---- #
ATTN_IMPL=${ATTN_IMPL:-"ring"}    # ring, ulysses, or both
DTYPE=${DTYPE:-"bf16"}            # bf16 or fp16

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --attn_impl=*)
            ATTN_IMPL="${1#*=}"
            shift 1
            ;;
        --attn_impl)
            if [[ -n "$2" && "$2" != --* ]]; then
                ATTN_IMPL="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        --dtype=*)
            DTYPE="${1#*=}"
            shift 1
            ;;
        --dtype)
            if [[ -n "$2" && "$2" != --* ]]; then
                DTYPE="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --attn_impl=<impl>   Attention implementation: ring, ulysses, or both (default: ring)"
            echo "  --dtype=<dtype>      Data type: bf16 or fp16 (default: bf16)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MASTER_ADDR          Master node address (default: 10.52.98.148)"
            echo "  MASTER_PORT          Master port (default: 16988)"
            echo "  NNODES               Number of nodes (default: 2)"
            echo "  NPROC_PER_NODE       GPUs per node (default: 8)"
            echo "  RANK                 Node rank (default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  TE CP Attention Benchmark"
echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  ATTN_IMPL=$ATTN_IMPL  DTYPE=$DTYPE"
echo "============================================================"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS benchmark_te_cp.py --attn_impl "$ATTN_IMPL" --dtype "$DTYPE"
