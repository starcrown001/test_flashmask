# python benchmark_flashmask.py --fm_version 1 --suffix ""
# sleep 60
export CUDA_VISIBLE_DEVICES=7
/root/paddlejob/workspace/env_run/xiehaoyang/magiattn_env/bin/python benchmark_flashmask.py --fm_version 3  --suffix "gsw"
# sleep 60
# python benchmark_flexattention.py
sleep 60
# python draw.py
# python draw.py --baseline "flexattention"
