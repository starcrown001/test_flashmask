#!/bin/bash

# ------------------------------------------------------------------ #
#  MagiAttention benchmark with BOTH nsys + torch.profiler           #
#                                                                     #
#  nsys         → CUDA API / NVTX / GPU metrics (进程外采集)          #
#  torch.profiler → Python 算子 / 内存 / FLOPs (进程内采集)            #
#                                                                     #
#  Usage:                                                             #
#    bash ./benchmark_magiattention_cp_cpu_nsys_profiler.sh           #
#                                                                     #
#  View results:                                                      #
#    nsys:  nsys-ui ./nsys_profiler_reports/magiattention_rank0.nsys-rep  #
#    torch.profiler: tensorboard --logdir ./nsys_profiler_reports/torch_profiler  #
# ------------------------------------------------------------------ #

# ---- 分布式基础配置 ---- #
export MASTER_ADDR=${MASTER_ADDR:-10.52.98.148}
export MASTER_PORT=${MASTER_PORT:-16988}
export NNODES=2
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
PROFILER_WARMUP=${PROFILER_WARMUP:-1}
PROFILER_ACTIVE=${PROFILER_ACTIVE:-3}

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

# nsys 包裹 torchrun，运行 profiler 版 Python 文件
# nsys 采集: CUDA API / NVTX / GPU metrics
# torch.profiler 采集: Python 算子 / 内存 / FLOPs (由 Python 代码内部控制)

$NSYS_BIN profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --sample=cpu \
    --output="${NSYS_OUTPUT_DIR}/magiattention_rank${RANK}" \
    --force-overwrite=true \
    torchrun $DISTRIBUTED_ARGS benchmark_magiattention_cp_cpu_profiler.py \
        --config "$CONFIG_PATH" \
        --profiler_output_dir "$TORCH_PROF_OUTPUT_DIR" \
        --profiler_wait "$PROFILER_WAIT" \
        --profiler_warmup "$PROFILER_WARMUP" \
        --profiler_active "$PROFILER_ACTIVE" \
        --fast_eval 

echo ""
echo "============================================================"
echo "  Profiling finished! Both nsys and torch.profiler data saved."
echo ""
echo "  [nsys] .nsys-rep files:"
echo "    $NSYS_OUTPUT_DIR/magiattention_rank*.nsys-rep"
echo "    View:  nsys-ui $NSYS_OUTPUT_DIR/magiattention_rank0.nsys-rep"
echo "    Export SQLite:  nsys export -t sqlite $NSYS_OUTPUT_DIR/magiattention_rank0.nsys-rep"
echo "    → upload to https://ui.perfetto.dev"
echo ""
echo "  [torch.profiler] TensorBoard traces:"
echo "    $TORCH_PROF_OUTPUT_DIR/rank*/"
echo "    View:  tensorboard --logdir $TORCH_PROF_OUTPUT_DIR"
echo "    → or upload trace.json.gz to https://ui.perfetto.dev"
echo "============================================================"
