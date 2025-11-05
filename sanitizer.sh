# logsave sanitizer.log /usr/local/cuda-12.9/bin/compute-sanitizer --tool=memcheck python -m pytest --maxfail=1 --timeout=30 --timeout_method=signal test_blockmaskv2.py 
# CUDA_VISIBLE_DEVICES='3' logsave sanitizer.log /usr/local/cuda-12.9/bin/compute-sanitizer --tool=synccheck python single_test.py

CUDA_VISIBLE_DEVICES=4 logsave sanitizer.log /usr/local/cuda-12.9/bin/compute-sanitizer --tool=memcheck /root/paddlejob/workspace/env_run/xiehaoyang/fm_env/bin/python benchmark_flashmask.py --dtype bf16 --fm_version 4