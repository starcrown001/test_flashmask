CUDA_VISIBLE_DEVICES='7' python -m pytest --maxfail=1 --timeout=100 --timeout_method=signal --log-file=pytest.log test_blockmask.py
