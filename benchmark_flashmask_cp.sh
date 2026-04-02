#!/bin/bash

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

/root/paddlejob/workspace/env_run/xiehaoyang/fm_env/bin/python -m paddle.distributed.launch \
    --log_dir paddle_cp_logs/output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    benchmark_flashmask_cp.py