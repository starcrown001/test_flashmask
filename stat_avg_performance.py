"""
统计 bfloat16 目录下不同方法的性能平均值。
按方法名分组，对所有同方法的 CSV 文件的数值列求平均，保持原始表头。
"""

import os
import glob
import re
import pandas as pd


def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')


def extract_method_name(filename):
    """从文件名中提取方法名，例如 blockattention_1_8192_32_128_processed.csv -> blockattention"""
    basename = os.path.basename(filename)
    # 方法名是第一个 _1_ 之前的部分
    match = re.match(r'^(.+?)_1_', basename)
    if match:
        return match.group(1)
    return None


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bfloat16')
    all_csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    # 按方法名分组
    method_files = {}
    for f in all_csv_files:
        method = extract_method_name(f)
        if method is None:
            continue
        method_files.setdefault(method, []).append(f)

    for method in sorted(method_files.keys()):
        files = sorted(method_files[method])
        print(f"\n{'='*80}")
        print(f"方法: {method}  (共 {len(files)} 个文件)")
        print(f"{'='*80}")

        dataframes = []
        for f in files:
            try:
                df = read_tsv(f)
                dataframes.append(df)
            except Exception as e:
                print(f"  [警告] 读取 {os.path.basename(f)} 失败: {e}")

        if not dataframes:
            print("  无有效数据")
            continue

        # 列出所有文件
        print("包含文件:")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        print()

        # 获取原始表头
        header = dataframes[0].columns.tolist()

        # 找出数值列和非数值列
        numeric_cols = []
        non_numeric_cols = []
        for col in header:
            if pd.api.types.is_numeric_dtype(dataframes[0][col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)

        # 拼接所有 DataFrame，对数值列求平均
        combined = pd.concat(dataframes, ignore_index=True)

        # 按非数值列分组（如 Causal 列），对数值列求平均
        if non_numeric_cols:
            avg_df = combined.groupby(non_numeric_cols, sort=False)[numeric_cols].mean().reset_index()
        else:
            avg_df = combined[numeric_cols].mean().to_frame().T

        # 保持原始列顺序
        avg_df = avg_df[header]

        # 输出结果，保持原始 TSV 格式
        print(avg_df.to_csv(sep='\t', index=False))


if __name__ == '__main__':
    main()
