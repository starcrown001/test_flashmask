#!/bin/bash
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
export MAGI_ATTENTION_HIERARCHICAL_COMM=0
export MAGI_ATTENTION_NATIVE_GRPCOLL=0
export MAGI_ATTENTION_QO_COMM=0
export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=0
export MAGI_ATTENTION_FA4_BACKEND=0

# ---- Python 环境 ---- #
source /root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/activate
PYTHON=/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/python

# ---- 项目根目录 ---- #
export PYTHONPATH=../../

# ---- 输出目录 ---- #
OUTPUT_BASE=${OUTPUT_BASE:-"./nsys_profiler_reports"}
NSYS_OUTPUT_DIR="${OUTPUT_BASE}/nsys"
TORCH_PROF_OUTPUT_DIR="${OUTPUT_BASE}/torch_profiler"
mkdir -p "$NSYS_OUTPUT_DIR" "$TORCH_PROF_OUTPUT_DIR"

# ---- nsys 配置 ---- #
NSYS_BIN=${NSYS_BIN:-/root/paddlejob/share-storage/gpfs/system-public/wusiming/nsys/bin/nsys}

# ---- torch.profiler 配置 ---- #
PROFILER_WAIT=${PROFILER_WAIT:-2}
PROFILER_WARMUP=${PROFILER_WARMUP:-20}
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
        --output_base=*)
            OUTPUT_BASE="${1#*=}"
            NSYS_OUTPUT_DIR="${OUTPUT_BASE}/nsys"
            TORCH_PROF_OUTPUT_DIR="${OUTPUT_BASE}/torch_profiler"
            mkdir -p "$NSYS_OUTPUT_DIR" "$TORCH_PROF_OUTPUT_DIR"
            shift 1
            ;;
        --output_base)
            if [[ -n "$2" && "$2" != --* ]]; then
                OUTPUT_BASE="$2"
                NSYS_OUTPUT_DIR="${OUTPUT_BASE}/nsys"
                TORCH_PROF_OUTPUT_DIR="${OUTPUT_BASE}/torch_profiler"
                mkdir -p "$NSYS_OUTPUT_DIR" "$TORCH_PROF_OUTPUT_DIR"
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
            echo "Usage: $0 [--config=PATH] [--output_base=DIR] [--wait=N] [--warmup=N] [--active=N]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  MagiAttention Benchmark (nsys + torch.profiler)"
echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  CONFIG=$CONFIG_PATH"
echo ""
echo "  nsys output:   $NSYS_OUTPUT_DIR"
echo "  profiler output: $TORCH_PROF_OUTPUT_DIR"
echo "  profiler schedule: wait=$PROFILER_WAIT warmup=$PROFILER_WARMUP active=$PROFILER_ACTIVE"
echo "============================================================"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS benchmark_magiattention_cp_cpu_new.py \
    --config "$CONFIG_PATH" \
    --fast_eval true

