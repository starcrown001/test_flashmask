for i in $(seq 4 5); do
    python -c "import paddle; paddle.device.set_device('gpu:$i'); paddle.randn([16, 1024, 1024, 1024])" &
done
CUDA_VISIBLE_DEVICES=4 /root/paddlejob/workspace/env_run/xiehaoyang/fm_cp_env/bin/python benchmark_flashblockmask.py