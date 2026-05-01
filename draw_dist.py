import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob
import os
import re
from collections import defaultdict


def read_tsv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"An error occurred reading {file_path}: {e}")
        return None


def parse_filename(filename):
    """
    Parse filename like:
      magiattention_{rank}_{WORLD_SIZE}_{B}_{S}_{H}_{D}_{idx}.csv  (7 numbers)
      magiattention_{rank}_{WORLD_SIZE(=1)}_{S}_{H}_{D}_{idx}.csv  (6 numbers, B omitted)
    Returns dict with rank, WORLD_SIZE, B, S, H, D, idx
    """
    basename = os.path.basename(filename)
    name = basename.replace('.csv', '')
    parts = name.split('_')
    # parts[0] = 'magiattention', rest are numbers
    nums = [int(x) for x in parts[1:]]

    if len(nums) == 7:
        return {
            'rank': nums[0], 'WORLD_SIZE': nums[1], 'B': nums[2],
            'S': nums[3], 'H': nums[4], 'D': nums[5], 'idx': nums[6]
        }
    elif len(nums) == 6:
        # WORLD_SIZE=1, B omitted (treat as B=1)
        return {
            'rank': nums[0], 'WORLD_SIZE': nums[1], 'B': 1,
            'S': nums[2], 'H': nums[3], 'D': nums[4], 'idx': nums[5]
        }
    else:
        print(f"Warning: unexpected filename format: {basename}")
        return None


def plot_bar_subplots(categories, save_path, metric_label):
    colors = ['#39CFC5', '#FF7D5E', '#6C8EBF', '#D4A574', '#82B366']
    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    num_categories = len(categories)
    if num_categories == 0:
        return
    num_cols = min(3, num_categories)
    num_rows = (num_categories + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(7 * num_cols, 5 * num_rows))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    bar_height = 0.6

    for idx, (title, data) in enumerate(categories.items()):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])

        labels = data['labels']
        values = data['values']
        x = np.arange(len(labels))

        ax.barh(x, values, bar_height, color=colors[:len(labels)])

        for j in range(len(labels)):
            ax.text(
                values[j] + max(values) * 0.01, x[j],
                f'{values[j]:.2f}',
                va='center', ha='left', fontsize=11,
                fontproperties=font_prop
            )

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=11, fontproperties=font_prop)
        ax.invert_yaxis()
        ax.set_xlabel(metric_label, fontsize=12, fontproperties=font_prop)
        ax.set_title(title, fontsize=13, fontproperties=font_prop)
        ax.tick_params(axis='x', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path + '.pdf', dpi=300, format='pdf')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    plt.rcParams['font.family'] = "Liberation Mono"

    root_dir = '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test'
    save_dir = '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test'

    filenames = glob.glob(os.path.join(root_dir, 'magiattention_*.csv'))
    if not filenames:
        print("No CSV files found.")
        return

    # Parse all filenames and group by shape (WORLD_SIZE, B, S, H, D)
    shape_to_files = defaultdict(list)
    for f in filenames:
        info = parse_filename(f)
        if info is None:
            continue
        shape_key = (info['WORLD_SIZE'], info['B'], info['S'], info['H'], info['D'])
        shape_to_files[shape_key].append(f)

    print(f"Found {len(shape_to_files)} unique shapes:")
    for k, v in sorted(shape_to_files.items()):
        print(f"  WS={k[0]}, B={k[1]}, S={k[2]}, H={k[3]}, D={k[4]} -> {len(v)} files")

    # Determine which metrics are available
    # WORLD_SIZE=1 files only have Time columns; others also have TFLOPs
    time_metrics = {
        'fwd': 'FW Time (ms)',
        'bwd': 'BW Time (ms)',
        'total': 'TOTAL Time (ms)',
    }
    tflops_metrics = {
        'fwd': 'FW TFLOPs/s',
        'bwd': 'BW TFLOPs/s',
        'total': 'TOTAL TFLOPs/s',
    }

    # For each WORLD_SIZE, generate plots
    ws_groups = defaultdict(dict)
    for shape_key, files in shape_to_files.items():
        ws = shape_key[0]
        ws_groups[ws][shape_key] = files

    for ws, shapes in sorted(ws_groups.items()):
        # Read one file to check available columns
        sample_df = read_tsv_to_dataframe(list(shapes.values())[0][0])
        has_tflops = 'FW TFLOPs/s' in sample_df.columns

        if has_tflops:
            metric_sets = {
                'time': (time_metrics, 'Time (ms)'),
                'tflops': (tflops_metrics, 'TFLOPs/s'),
            }
        else:
            metric_sets = {
                'time': (time_metrics, 'Time (ms)'),
            }

        for metric_type, (metrics, unit_label) in metric_sets.items():
            for kernel_name, metric_col in metrics.items():
                categories = {}
                for shape_key in sorted(shapes.keys()):
                    files = shapes[shape_key]
                    ws_val, B, S, H, D = shape_key

                    dataframes = []
                    for fp in files:
                        df = read_tsv_to_dataframe(fp)
                        if df is not None and metric_col in df.columns:
                            dataframes.append(df)

                    if not dataframes:
                        continue

                    # Average over rank and idx
                    non_numeric_col = 'Operation'
                    columns_to_avg = [metric_col]
                    if 'Sparsity' in dataframes[0].columns and metric_type == 'tflops':
                        columns_to_avg.append('Sparsity')

                    aligned = [df[columns_to_avg] for df in dataframes]
                    combined = pd.concat(aligned, axis=0, keys=range(len(dataframes)))
                    mean_df = combined.groupby(level=1).mean()
                    mean_df[non_numeric_col] = dataframes[0][non_numeric_col]

                    labels = [op.strip() for op in mean_df[non_numeric_col].tolist()]
                    values = mean_df[metric_col].tolist()

                    title = f'WS={ws_val}, B={B}, S={S//1024}K, H={H}, D={D}'
                    categories[title] = {
                        'labels': labels,
                        'values': values,
                    }

                if not categories:
                    continue

                if metric_type == 'tflops':
                    label = f'{kernel_name.upper()} Speed (TFLOPs/s)'
                else:
                    label = f'{kernel_name.upper()} Time (ms)'

                save_name = f'magiattention_ws{ws}_{metric_type}_{kernel_name}'
                save_path = os.path.join(save_dir, save_name)
                plot_bar_subplots(categories, save_path, label)


if __name__ == "__main__":
    main()
