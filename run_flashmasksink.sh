export PYTHONPATH=/root/paddlejob/workspace/env_run/xiehaoyang/PaddleFleet/src:$PYTHONPATH
python -m pytest -v test_flashmask_sink.py \
  # -k "gen_startend_row_indices1 and d128 and dv128" \
  2>&1 | tee log_sink