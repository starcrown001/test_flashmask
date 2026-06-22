export PYTHONPATH=/root/paddlejob/workspace/env_run/xiehaoyang/PaddleFleet/src:$PYTHONPATH
python -m pytest -v test_flashmask_sink_new.py \
  # -k "gen_startend_row_indices1 and d64 and dv64 and True" \
  2>&1 | tee log_sink_new
