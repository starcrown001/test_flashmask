export CUDA_VISIBLE_DEVICES='6'
echo "[Warn] Note that regex is set for filtering kernel names, this could result in missing items. Be careful."
echo "[Warn] regex applied is '$regex_filter'"
/usr/local/cuda-12.9/bin/ncu --set "full" --nvtx --nvtx-include "paddle/"  -o fwd_bwd_128k_shared_1 -f --import-source yes /root/paddlejob/workspace/env_run/xiehaoyang/fm_env/bin/python single_test.py