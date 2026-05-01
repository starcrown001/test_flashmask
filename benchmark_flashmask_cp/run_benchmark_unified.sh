#!/bin/bash

# Unified FlashMask CP Benchmark launcher
# Usage: bash run_benchmark_unified.sh [MODE]
#   MODE: baseline | overlap | balance | balance_overlap  (default: balance_overlap)

MODE="${1:-balance_overlap}"

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

# 屏蔽平台预设的环境变量
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

START_RANK=1
END_RANK=2

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi
rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36678

export FLAGS_flash_attn_version=3

if [ ! -d bf16_dist_test ]; then
    mkdir bf16_dist_test/
fi

# NVSHMEM env vars (needed for overlap modes)
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_IBGDA_NUM_RC_PER_PE=4
export NVSHMEM_DISABLE_GDRCOPY=0
export NVSHMEM_IB_ENABLE_RELAXED_ORDERING=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0
export NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=32
export NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC=2048

# export FLAGS_call_stack_level=3
# export CUDA_LAUNCH_BLOCKING=1

ARGS="--mode $MODE --use_rs"
PYTHON_BIN="/root/paddlejob/share-storage/gpfs/system-public/xiehaoyang/xhy_backup/fm_test_env/bin/python"
BENCH_FILE="benchmark_flashmask_cp_unified.py"

echo "Start benching mode=$MODE with: $PYTHON_BIN $BENCH_FILE"

# num_heads=(4 8 1)
# for num_head in "${num_heads[@]}"; do
#     echo "Number of head: $num_head"

#     $PYTHON_BIN -m paddle.distributed.launch \
#         --log_dir paddle_cp_logs_unified/output_$rank/ \
#         --master $master:$port \
#         --nnodes $nnodes \
#         --rank $rank \
#         --run_mode=collective \
#         $BENCH_FILE --batch 1 --num_heads $num_head $ARGS

#     $PYTHON_BIN -m paddle.distributed.launch \
#         --log_dir paddle_cp_logs_unified/output_$rank/ \
#         --master $master:$port \
#         --nnodes $nnodes \
#         --rank $rank \
#         --run_mode=collective \
#         $BENCH_FILE --batch 2 --num_heads $num_head $ARGS
# done
# echo "End benching mode=$MODE with: $PYTHON_BIN $BENCH_FILE"

num_heads=(1)
# export CUDA_VISIBLE_DEVICES='6,7'
for num_head in "${num_heads[@]}"; do
    echo "Number of head: $num_head"

    # $PYTHON_BIN -m paddle.distributed.launch \
    #     --log_dir paddle_cp_logs_unified/output_$rank/ \
    #     --master $master:$port \
    #     --nnodes $nnodes \
    #     --rank $rank \
    #     --run_mode=collective \
    #     $BENCH_FILE --cp_size 4 --num_heads $num_head $ARGS

    # $PYTHON_BIN -m paddle.distributed.launch \
    #     --log_dir paddle_cp_logs_unified/output_$rank/ \
    #     --master $master:$port \
    #     --nnodes $nnodes \
    #     --rank $rank \
    #     --run_mode=collective \
    #     $BENCH_FILE --cp_size 8 --num_heads $num_head $ARGS

    # $PYTHON_BIN -m paddle.distributed.launch \
    #     --log_dir paddle_cp_logs_unified/output_$rank/ \
    #     --master $master:$port \
    #     --nnodes $nnodes \
    #     --rank $rank \
    #     --run_mode=collective \
    #     $BENCH_FILE --cp_size 16 --num_heads $num_head $ARGS

    # /root/paddlejob/share-storage/gpfs/system-public/wusiming/nsys/bin/nsys profile \
    # --trace=cuda,nvtx,mpi,osrt \
    # --output=./flashmask_doc_cp32_baseline_causal_doc_no_mp \
    # --force-overwrite=true \
    # --cuda-memory-usage=true \
    # --gpu-metrics-device=all \
    $PYTHON_BIN -m paddle.distributed.launch \
        --log_dir paddle_cp_logs_unified/output_$rank/ \
        --master $master:$port \
        --nnodes $nnodes \
        --rank $rank \
        --run_mode=collective \
        $BENCH_FILE --cp_size 8 --num_heads $num_head $ARGS
done
echo "End benching mode=$MODE with: $PYTHON_BIN $BENCH_FILE"
