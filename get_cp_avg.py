import pandas as pd
import glob
import os
import numpy as np

folder = '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test/'  # 替换为你的文件夹
all_files = glob.glob(os.path.join(folder, "*131072*.csv"))

dfs = []
for file in all_files:
    df = pd.read_csv(file, sep=None, engine='python')
    dfs.append(df)

# 检查是否所有文件形状一致（行数和列名）
assert all([df.shape == dfs[0].shape for df in dfs]), "所有CSV文件形状应一致"
assert all([all(df.columns == dfs[0].columns) for df in dfs]), "所有CSV文件列名应一致"
# assert all([all(df['Operation'] == dfs[0]['Operation']) for df in dfs]), "所有CSV文件Operation列应一致"

# 堆叠到一个三维数组 (文件数, 行数, 列数)
# 只对数值部分求平均（假定第一列是非数值 Operation）
num_cols = dfs[0].columns[1:]

values = np.stack([df[num_cols].values for df in dfs], axis=0)  # (文件数, 行数, 列数)

mean_values = values.mean(axis=0)  # (行数, 列数)

# 重新组装到DataFrame
result_df = pd.DataFrame(mean_values, columns=num_cols)
# result_df.insert(0, 'Operation', dfs[0]['Operation'])

# 输出结果
print(result_df)