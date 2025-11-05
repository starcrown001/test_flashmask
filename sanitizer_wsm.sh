# logsave sanitizer.log compute-sanitizer --tool=memcheck python -m pytest --maxfail=1 --timeout=30 --timeout_method=signal test_blockmaskv2.py 
compute-sanitizer --tool=racecheck /root/paddlejob/workspace/env_run/xiehaoyang/fm_env/bin/python single_test.py
