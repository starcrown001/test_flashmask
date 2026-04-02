source ~/.bashrc
conda activate /root/paddlejob/workspace/env_run/xiehaoyang/xattn
CUDA_VISIBLE_DEVICES=4 /root/paddlejob/workspace/env_run/xiehaoyang/xattn/bin/python benchmark_blockmask.py