#!/bin/bash

# ------------------------------------------------------------------ #
#  MagiAttention distributed benchmark with Nsight Systems (nsys)     #
#  Generates .nsys-rep for Perfetto / Nsight UI timeline analysis    #
#                                                                     #
#  Usage:                                                             #
#    ./benchmark_magiattention_cp_cpu_nsys.sh                         #
#                                                                     #
#  View report:                                                       #
#    nsys-ui ./nsys_reports/magiattention_rank0.nsys-rep             #
#    or upload to https://ui.perfetto.dev                             #
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

# ---- nsys 配置 ---- #
NSYS_BIN=${NSYS_BIN:-/root/paddlejob/share-storage/gpfs/system-public/wusiming/nsys/bin/nsys}
OUTPUT_DIR=${NSYS_OUTPUT_DIR:-"./magi_nsys_reports"}
CONFIG_PATH=${CONFIG_PATH:-"magi_benchmark_conf.py"}

mkdir -p "$OUTPUT_DIR"

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
            OUTPUT_DIR="${1#*=}"
            shift 1
            ;;
        --output_dir)
            if [[ -n "$2" && "$2" != --* ]]; then
                OUTPUT_DIR="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config=PATH] [--output_dir=DIR]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  MagiAttention Distributed Benchmark (Nsight Systems)"
echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  CONFIG=$CONFIG_PATH"
echo "  NSYS=$NSYS_BIN"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "============================================================"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- nsys profile 关键参数说明 ---- #
#   --trace=cuda,nvtx,osrt   : 追踪 CUDA API、NVTX 标注、OS Runtime
#   --nvtx-include="fwd*/bwd*/warmup" : 只录制带这些 NVTX 前缀的范围（可选，去掉则录制全部）
#   --cuda-memory-usage=true  : 记录 CUDA 内存使用
#   --gpu-metrics-device=all  : 记录 GPU 利用率/功耗等指标
#   --sample=cpu              : 采样 CPU 线程栈
#   --output                  : 输出文件路径（每个 rank 独立）
#   --force-overwrite=true    : 覆盖已有文件

$NSYS_BIN profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --sample=cpu \
    --output="${OUTPUT_DIR}/magiattention_rank${RANK}" \
    --force-overwrite=true \
    torchrun $DISTRIBUTED_ARGS benchmark_magiattention_cp_cpu_nsys.py \
        --config "$CONFIG_PATH" --fast_eval true

echo ""
echo "============================================================"
echo "  nsys profiling finished!"
echo "  Report files: ${OUTPUT_DIR}/magiattention_rank*.nsys-rep"
echo ""
echo "  View with:"
echo "    Nsight UI:    nsys-ui ${OUTPUT_DIR}/magiattention_rank0.nsys-rep"
echo "    Perfetto:     File > Open in Nsight UI, or export to SQLite:"
echo "                  nsys export -t sqlite ${OUTPUT_DIR}/magiattention_rank0.nsys-rep"
echo "                  then upload to https://ui.perfetto.dev"
echo ""
echo "  Quick stats:"
echo "    nsys stats ${OUTPUT_DIR}/magiattention_rank0.nsys-rep"
echo "============================================================"
