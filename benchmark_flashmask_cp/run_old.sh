#!/bin/bash

bench_baseline=${1:-0}
mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}
#
echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
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

START_RANK=0
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
port=36677

export FLAGS_flash_attn_version=3

# rm bf16_dist_test/* -f
if [ ! -d bf16_dist_test ]; then
    mkdir bf16_dist_test/
fi

export PYTHONPATH=/root/paddlejob/gpfs/wusiming/xhy-flash-attention/csrc/utils/cp_balance/:$PYTHONPATH

# 1. Activate GPU Direct Async
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu

# 2. Increase Parallelism
export NVSHMEM_IBGDA_NUM_RC_PER_PE=4

# 3. Ensure direct hardware access
export NVSHMEM_DISABLE_GDRCOPY=0
export NVSHMEM_IB_ENABLE_RELAXED_ORDERING=1

# 4. Topology
# Enable mapping logic
export NVSHMEM_ENABLE_NIC_PE_MAPPING=0

# 5. Aggressive request submission and more slots
export NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=32
export NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC=2048

BENCH_FILE="benchmark_flashmask_cp_balance_overlap.py"
PYTHON_BIN="../erniebot/hqy_eb5_env/bin/python"
if [ $bench_baseline -eq 1 ]; then
    BENCH_FILE="benchmark_flashmask_cp.py"
fi

echo "Start benching with: $PYTHON_BIN $BENCH_FILE"
$PYTHON_BIN -m paddle.distributed.launch \
    --log_dir paddle_cp_logs/output_$rank/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    $BENCH_FILE --batch 2

$PYTHON_BIN -m paddle.distributed.launch \
    --log_dir paddle_cp_logs/output_$rank/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    $BENCH_FILE --batch 1
echo "End benching with: $PYTHON_BIN $BENCH_FILE"