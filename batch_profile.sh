#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: python_env path is not specified (<python_env>/bin/python)"
    echo "Usage: ./batch_profile.sh <python_env> [device] [suffix] [masktype]"
    echo "Example: $0 ../hqy_env/ 0 new global_swin"
    exit 1
fi

python_env=$1
device=${2:-0}
suffix=${3:-"new"}
masktype=${4:-"global_swin"}

seq_len_base=1024
seq_lens=(128 64 32 16 8 4 2)

regex_filter="(.*flashmask.*|.*device_kernel.*)"
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"

output_folder="./outputs/"
if [ ! -d $output_folder ]; then
    echo "Creating a folder for ncu batch prof at '$output_folder'"
    mkdir -p $output_folder
fi

echo "" &> profile_log.log
task_cnt=0
total_task=${#seq_lens[@]}
for seq_len in "${seq_lens[@]}"; do
    task_cnt=$((task_cnt + 1))
    seq_len_used=$((seq_len_base * seq_len))
    length_str="${seq_len}k"
    logging="`date` -- Task ($task_cnt/$total_task) Profiling with sequence length = $length_str (device = $device, python env: $python_env, mask type: $masktype)..."
    echo $logging
    echo $logging >> profile_log.log
    output_name="fwd_${length_str}_${suffix}"
    export CUDA_VISIBLE_DEVICES=$device
    env CUDA_VISIBLE_DEVICES=$device ncu --set "full" --nvtx --nvtx-include "flashmask/" \
        --kernel-name=regex:$regex_filter \
        -o $output_name \
        -f --import-source yes \
        $python_env/bin/python profile_flashmask.py --seqlen_q $seq_len_used --seqlen_k $seq_len_used \
        --d 128 --dv 128 --nheads 32 --backward_prof --mask_type $masktype \
        --warmup_times 10 --profile_times 3 >> profile_log.log
    mv ${output_name}.ncu-rep $output_folder
    sleep 15        # cooldown
done

echo "${total_task} items profiled. Batch profile output folder: $output_folder"
