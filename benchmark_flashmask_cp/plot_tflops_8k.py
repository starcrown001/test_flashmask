#!/usr/bin/env python3
"""
Plot TFLOPs/s line charts for single-card seqlen=8k equivalent experiments.
Filter: WS=1→S=16384, WS=8→S=131072, WS=16→S=262144
Average over rank and idx; plot independently for other params.
"""

import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot TFLOPs/s line charts for single-card seqlen=8k experiments.")
parser.add_argument('--data_dir', type=str, required=True,
                    help='Directory containing CSV result files.')
args = parser.parse_args()
DATA_DIR = args.data_dir

# Target seqlen per WORLD_SIZE (single-card seqlen = 8k)
# TARGET_S = {1: 16384,2:32768,4:65536, 8: 131072, 16: 262144, 32: 524288}
TARGET_S = {4:32768, 8: 65536, 16: 131072, 32:262144}

TFLOPS_COLS = ['FW TFLOPs/s', 'BW TFLOPs/s', 'TOTAL TFLOPs/s']
TFLOPS_LABELS = {'FW TFLOPs/s': 'FW', 'BW TFLOPs/s': 'BW', 'TOTAL TFLOPs/s': 'TOTAL'}


def parse_filename(fname):
    """Parse CSV filename into parameter dict.
    Format with B:    {method}_{rank}_{WS}_{B}_{S}_{H}_{D}_{idx}.csv  (8 fields)
    Format without B: {method}_{rank}_{WS}_{S}_{H}_{D}_{idx}.csv      (7 fields)
    """
    stem = fname.replace('.csv', '')
    parts = stem.split('_')
    print(stem)

    # Identify method by prefix (method name may contain underscores)
    if stem.startswith('flashmask_unified_balance_overlap_'):
        method = 'flashmask_overlap'
        rest = parts[4:]  # remove 'flashmask' and 'overlap'
        print(rest)
    elif stem.startswith('magiattention_'):
        method = 'magiattention'
        rest = parts[1:]  # remove 'magiattention'
    elif stem.startswith('flashmask_balancecomm_balance_overlap_'):
        method = 'flashmask_balancecomm_overlap'
        rest = parts[4:]  # remove 'flashmask' and 'balancecomm'
    elif stem.startswith('flashmask_'):
        method = 'flashmask'
        rest = parts[1:]  # remove 'flashmask'
    elif stem.startswith('te_ring_'):
        method = 'te_ring'
        rest = parts[2:]  # remove 'te_ring'
    else:
        return None

    if len(rest) == 6:
        # No B: rank, WS, S, H, D, idx
        b, s, h, d, idx, rank = int(rest[0]), int(rest[1]), int(rest[2]), int(rest[3]), int(rest[4]), int(rest[5])
        ws = s // 8192
    elif len(rest) == 7:
        # With B: rank, WS, B, S, H, D, idx
        rank, ws, b, s, h, d, idx = int(rest[0]), int(rest[1]), int(rest[2]), int(rest[3]), int(rest[4]), int(rest[5]), int(rest[6])
    else:
        return None

    return {
        'method': method, 'rank': rank, 'ws': ws, 'B': b,
        'S': s, 'H': h, 'D': d, 'idx': idx
    }


def read_csv_tflops(filepath):
    """Read CSV and return rows with TFLOPs/s data."""
    try:
        df = pd.read_csv(filepath, sep='\t', skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        # Check if TFLOPs columns exist
        if 'TOTAL TFLOPs/s' not in df.columns:
            return None
        df['Operation'] = df['Operation'].str.strip()
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

    # Collect all data: key=(method, ws, B, H, D, operation), value=list of tflops dicts
    records = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        params = parse_filename(fname)
        if params is None:
            continue

        ws = params['ws']
        s = params['S']

        # Filter by target seqlen
        if ws not in TARGET_S or TARGET_S[ws] != s:
            continue

        df = read_csv_tflops(fpath)
        if df is None:
            continue

        print(df)

        for _, row in df.iterrows():
            rec = {
                'method': params['method'],
                'ws': ws,
                'B': params['B'],
                'H': params['H'],
                'D': params['D'],
                'rank': params['rank'],
                'idx': params['idx'],
                'Operation': row['Operation'],
            }
            for col in TFLOPS_COLS:
                if col in df.columns:
                    rec[col] = float(row[col])
            records.append(rec)

    if not records:
        print("No matching data found!")
        return

    df_all = pd.DataFrame(records)
    print(f"Total records: {len(df_all)}")
    print(f"Methods: {df_all['method'].unique()}")
    print(f"WS values: {sorted(df_all['ws'].unique())}")
    print(f"H values: {sorted(df_all['H'].unique())}")
    print(f"D values: {sorted(df_all['D'].unique())}")
    print(f"B values: {sorted(df_all['B'].dropna().unique())}")
    print(f"Operations: {df_all['Operation'].unique()}")

    # Average over rank and idx
    group_cols = ['method', 'ws', 'B', 'H', 'D', 'Operation']
    df_avg = df_all.groupby(group_cols, dropna=False)[TFLOPS_COLS].mean().reset_index()

    # Determine unique (method, H, D) combinations for separate figures
    # B is usually 1 or None, group it in
    plot_keys = df_avg.groupby(['method', 'B', 'H', 'D'], dropna=False).size().reset_index()[['method', 'B', 'H', 'D']]

    for _, pk in plot_keys.iterrows():
        method = pk['method']
        b_val = pk['B']
        h_val = pk['H']
        d_val = pk['D']

        mask = (df_avg['method'] == method) & (df_avg['H'] == h_val) & (df_avg['D'] == d_val)
        if pd.isna(b_val):
            mask = mask & (df_avg['B'].isna())
            b_str = 'noB'
        else:
            mask = mask & (df_avg['B'] == b_val)
            b_str = f'B{int(b_val)}'

        subset = df_avg[mask]
        if subset.empty:
            continue

        operations = sorted(subset['Operation'].unique())
        ws_values = sorted(subset['ws'].unique())

        # Create figure with 3 subplots: FW, BW, TOTAL
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{method}  H={h_val} D={d_val} {b_str}\n(single-card seqlen=8k)',
                     fontsize=14, fontweight='bold')

        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

        for ax_idx, tflops_col in enumerate(TFLOPS_COLS):
            ax = axes[ax_idx]
            label_name = TFLOPS_LABELS[tflops_col]

            for op_idx, op in enumerate(operations):
                op_data = subset[subset['Operation'] == op]
                xs = []
                ys = []
                for ws_val in ws_values:
                    row = op_data[op_data['ws'] == ws_val]
                    if not row.empty:
                        xs.append(ws_val)
                        ys.append(row[tflops_col].values[0])

                color = colors[op_idx % len(colors)]
                marker = markers[op_idx % len(markers)]
                ax.plot(xs, ys, marker=marker, color=color, label=op,
                        linewidth=2, markersize=8)

            ax.set_xlabel('WORLD_SIZE', fontsize=12)
            ax.set_ylabel('TFLOPs/s', fontsize=12)
            ax.set_title(f'{label_name} TFLOPs/s', fontsize=13)
            ax.set_xticks(ws_values)
            ax.set_xticklabels([str(w) for w in ws_values])
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out_name = f'tflops_8k_{method}_{b_str}_H{h_val}_D{d_val}'
        for ext in ['png', 'pdf']:
            out_path = os.path.join(DATA_DIR, f'{out_name}.{ext}')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_path}")
        plt.close(fig)

    print("\nDone!")


if __name__ == '__main__':
    main()
