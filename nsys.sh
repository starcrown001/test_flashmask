export CUDA_VISIBLE_DEVICES='6'
/opt/nvidia/nsight-systems/2023.2.1/bin/nsys profile \
  --output paddle_nsys_report_base \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  /root/paddlejob/workspace/env_run/xiehaoyang/fm_env_base/bin/python benchmark_flashmask.py --dtype bf16 --fm_version 4