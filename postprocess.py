import pandas as pd
import glob
import os

# 目录路径
dir_path = '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bfloat16'
# 获取所有csv文件
csv_files = glob.glob(os.path.join(dir_path, '*.csv'))

# 需要处理的列
flops_cols = [
    '   FW FLOPs', '   BW FLOPs', '  TOTAL FLOPs',
    '  FW TFLOPs/s', '  BW TFLOPs/s', '  TOTAL TFLOPs/s'
]


for file in csv_files:
    print(f'Processing: {file}')
    # 用tab分隔符读取
    df = pd.read_csv(file, sep='\t')
    print(df.keys())
    # 去除列名和内容的前后空格
    # df.columns = [col.strip() for col in df.columns]
    for col in flops_cols:
        if col in df.columns:
            # 先转成数字型（防止科学计数法被当字符串）
            df[col] = pd.to_numeric(df[col], errors='ignore')
    # 兼容 True/False 和 'True'/'False'
    if 'Causal  ' in df.columns:
        print(df['Causal  '].unique())
        mask = (df['Causal  '] == 'True    ') | (df['Causal  '] == 'True    ')
        print(f"找到 Causal 列, 处理:",mask.sum())
        cols_to_process = [col for col in flops_cols if col in df.columns]
        df.loc[mask, cols_to_process] = df.loc[mask, cols_to_process] / 2
    else:
        print(f"未找到 Causal 列, 跳过: {file}")

    # 保存（覆盖原文件，或保存为新文件都可以）
    # df.to_csv(file, index=False)
    # 或者保存为新文件
    new_file = file.replace('.csv', '_processed.csv')
    df.to_csv(new_file,sep='\t', index=False)

print("全部处理完毕！")