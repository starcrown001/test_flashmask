#!/bin/bash

# ------------------------------------------------------------------ #
#  MagiAttention distributed benchmark with torch.profiler            #
#  Generates traces for Perfetto / TensorBoard timeline analysis     #
# ------------------------------------------------------------------ #

# ---- 分布式基础配置 ---- #
export MASTER_ADDR=${MASTER_ADDR:-10.52.98.148}
export MASTER_PORT=${MASTER_PORT:-16988}
export NNODES=1
export NPROC_PER_NODE=8
export RANK=0
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ---- MagiAttention 性能相关环境变量 ---- #
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_CGA_CLUSTER_SIZE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export MAGI_ATTENTION_CATGQA=0
export MAGI_ATTENTION_AUTO_RANGE_MERGE=0
export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=0
export MAGI_ATTENTION_HIERARCHICAL_COMM=1
export MAGI_ATTENTION_NATIVE_GRPCOLL=0
export MAGI_ATTENTION_QO_COMM=0
export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=0
export MAGI_ATTENTION_FA4_BACKEND=0

# ---- Python 环境 ---- #
source /root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/activate
PYTHON=/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/python

# ---- 项目根目录 ---- #
export PYTHONPATH=../../

# ---- Profiler 参数 ---- #
PROFILER_OUTPUT_DIR=${PROFILER_OUTPUT_DIR:-"./profiler_traces"}
PROFILER_WAIT=${PROFILER_WAIT:-2}
PROFILER_WARMUP=${PROFILER_WARMUP:-3}
PROFILER_ACTIVE=${PROFILER_ACTIVE:-10}

# ---- 配置文件 ---- #
CONFIG_PATH=${CONFIG_PATH:-"magi_benchmark_conf.py"}

# 解析命令行参数
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
        --output_dir=*)
            PROFILER_OUTPUT_DIR="${1#*=}"
            shift 1
            ;;
        --output_dir)
            if [[ -n "$2" && "$2" != --* ]]; then
                PROFILER_OUTPUT_DIR="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        --wait=*)
            PROFILER_WAIT="${1#*=}"
            shift 1
            ;;
        --warmup=*)
            PROFILER_WARMUP="${1#*=}"
            shift 1
            ;;
        --active=*)
            PROFILER_ACTIVE="${1#*=}"
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config=PATH] [--output_dir=DIR] [--wait=N] [--warmup=N] [--active=N]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  MagiAttention Distributed Benchmark (torch.profiler)"
echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  CONFIG=$CONFIG_PATH"
echo "  PROFILER_OUTPUT_DIR=$PROFILER_OUTPUT_DIR"
echo "  PROFILER schedule: wait=$PROFILER_WAIT warmup=$PROFILER_WARMUP active=$PROFILER_ACTIVE"
echo "============================================================"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS benchmark_magiattention_cp_cpu_profiler.py \
    --config "$CONFIG_PATH" \
    --profiler_output_dir "$PROFILER_OUTPUT_DIR" \
    --profiler_wait "$PROFILER_WAIT" \
    --profiler_warmup "$PROFILER_WARMUP" \
    --profiler_active "$PROFILER_ACTIVE" \
    --fast_eval

echo ""
echo "============================================================"
echo "  Profiler finished!"
echo "  Trace files: $PROFILER_OUTPUT_DIR/rank*/"
echo ""
echo "  View with:"
echo "    TensorBoard:  tensorboard --logdir $PROFILER_OUTPUT_DIR"
echo "    Perfetto:     upload $PROFILER_OUTPUT_DIR/rank*/trace.json.gz to https://ui.perfetto.dev"
echo "============================================================"
