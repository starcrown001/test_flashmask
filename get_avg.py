import pandas as pd

# 读取 csv 文件
df = pd.read_csv('/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/flashmask-cp/bf16_dist_test_dump/flashmask_2_1_32768_1_128.csv', delim_whitespace=True)

# 计算每一列的平均值
print("每一列的均值如下：")
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        print(f"{col} 的平均值: {df[col].mean()}")