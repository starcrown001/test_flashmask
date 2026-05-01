#!/usr/bin/env python3
"""
Plot TFLOPs/s line charts for single-card seqlen=8k equivalent experiments.
Filter: WS=1→S=16384, WS=8→S=131072, WS=16→S=262144
Average over rank and idx; plot independently for other params.

Supports multiple DATA_DIRs — curves from each directory are drawn on the
same figure, distinguished by the LABELS prefix.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Edit here =====
DATA_DIR = [
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0424-no-balance-cpu',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0424',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0424-new-balance-eps0.1-cpu',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0424-new-balance-eps0.2-cpu',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0424-no-balance-multi-stream-cpu',
    # '/root/paddlejob/share-storage/gpfs/system-public/xiehaoyang/xhy_backup/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_te_ring_cpu_test_0',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0428',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_balance_overlap_locswap_False_intrabalance_False_eps_0.2_8192',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_balance_overlap_locswap_False_intrabalance_True_eps_0.2_8192',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_balance_overlap_locswap_False_intrabalance_True_eps_0.2_8192_partially',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_balance_overlap_locswap_False_intrabalance_True_eps_0.2_8192_ipo',
    '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0501',
    '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_te_ring_cpu_test',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/gp/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_magi_cpu',
    '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test_cpu'
    # '/root/paddlejob/share-storage/gpfs/system-public/xiehaoyang/xhy_backup/flashmask/test_flashmask/benchmark_flashmask_cp/bf16_dist_test_magi_cpu_02',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test_cpu_01',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test_cpu_02',
    # '/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/bf16_dist_test_cpu',
    # '/root/paddlejob/share-storage/gpfs/system-public/heqianyue/cp_flashmask_bench/flashmask_bench/bf16_dist_test_0427-per-work-ld-cg',
]
LABELS = [
    # 'Overlap (CPU)',
    # 'Balance Comm Overlap (EPS = 0.1)',
    # 'Balance Comm Overlap intra balance (EPS = 0.8)',
    # 'Balance Comm Overlap Orin(EPS = 0.2)',
    # 'Balance Comm Overlap (EPS = 0.2)',
    # 'Balance Comm Overlap Intra Balance (EPS = 0.2)',
    'Balance Comm Overlap (Hierarchical)',
    # 'Overlap', 
    # 'Balance Overlap (EPS=2)',
    # 'Baseline',
    # 'Magi Old version',
    # 'Te Ring old',
    'Te Ring',
    # 'Magi old',
    'Magi',
    # 'Magi Cpu dist sync1',
    # 'Magi Cpu dist sync2',
    # 'Magi Cpu sync dist',
    # 'Magi Cpu sync dist',
    # 'Magi Cpu sync dist',
    # 'Balance Comm Overlap (EPS = 0.3)'
]  # same length as DATA_DIR, or empty (auto: "dir0", "dir1", ...)
# ======================

TARGET_S = {4: 32768, 8: 65536, 16: 131072, 32: 262144}

TFLOPS_COLS = ['FW TFLOPs/s', 'BW TFLOPs/s', 'TOTAL TFLOPs/s']
TFLOPS_LABELS = {'FW TFLOPs/s': 'FW', 'BW TFLOPs/s': 'BW', 'TOTAL TFLOPs/s': 'TOTAL'}


def parse_filename(fname):
    """Parse CSV filename into parameter dict.
    Format with B:    {method}_{rank}_{WS}_{B}_{S}_{H}_{D}_{idx}.csv  (8 fields)
    Format without B: {method}_{rank}_{WS}_{S}_{H}_{D}_{idx}.csv      (7 fields)
    """
    stem = fname.replace('.csv', '')
    parts = stem.split('_')

    if stem.startswith('flashmask_unified_balance_overlap_'):
        method = 'flashmask_balance_overlap'
        rest = parts[4:]
    elif stem.startswith('flashmask_unified_overlap_'):
        method = 'flashmask_overlap'
        rest = parts[3:]
    elif stem.startswith('flashmask_unified_baseline_'):
        method = 'flashmask_baseline'
        rest = parts[3:]
    elif stem.startswith('flashmask_unified_balance_comm_'):
        method = 'flashmask_balance_comm'
        rest = parts[4:]
    elif stem.startswith('flashmask_intra_balance_balance_overlap_'):
        method = 'flashmask'
        rest = parts[5:]
    elif stem.startswith('magiattention_'):
        method = 'magiattention'
        rest = parts[1:]
    elif stem.startswith('flashmask_'):
        method = 'flashmask'
        rest = parts[1:]
    elif stem.startswith('te_ring_'):
        method = 'te_ring'
        rest = parts[2:]
    else:
        return None

    if len(rest) == 6:
        b, s, h, d, idx, rank = int(rest[0]), int(rest[1]), int(rest[2]), int(rest[3]), int(rest[4]), int(rest[5])
        ws = s // 8192
    elif len(rest) == 7:
        rank, ws, b, s, h, d, idx = (int(x) for x in rest)
    else:
        return None

    return {'method': method, 'rank': rank, 'ws': ws, 'B': b,
            'S': s, 'H': h, 'D': d, 'idx': idx}


def read_csv_tflops(filepath):
    try:
        df = pd.read_csv(filepath, sep='\t', skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        if 'TOTAL TFLOPs/s' not in df.columns:
            return None
        df['Operation'] = df['Operation'].str.strip()
        # Normalize: strip method suffixes so the same mask groups together
        import re as _re
        df['Operation'] = df['Operation'].apply(
            lambda s: _re.sub(r'\s*\(.*?\)\s*$', '', s).strip())
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def load_records(data_dir, dir_label):
    """Load all matching CSV records from one directory, tagging with dir_label."""
    records = []
    for fpath in glob.glob(os.path.join(data_dir, '*.csv')):
        params = parse_filename(os.path.basename(fpath))
        if params is None:
            continue
        ws, s = params['ws'], params['S']
        if ws not in TARGET_S or TARGET_S[ws] != s:
            continue
        df = read_csv_tflops(fpath)
        if df is None:
            continue
        for _, row in df.iterrows():
            rec = {
                'dir_label': dir_label,
                'method': params['method'],
                'ws': ws, 'B': params['B'], 'H': params['H'],
                'D': params['D'], 'rank': params['rank'],
                'idx': params['idx'], 'Operation': row['Operation'],
            }
            for col in TFLOPS_COLS:
                if col in df.columns:
                    rec[col] = float(row[col])
            records.append(rec)
    return records


def main():
    # Resolve labels
    labels = list(LABELS) if LABELS else [f"dir{i}" for i in range(len(DATA_DIR))]
    print(labels)
    print(DATA_DIR)
    assert len(labels) == len(DATA_DIR), "LABELS must be empty or same length as DATA_DIR but not {}".format(len(DATA_DIR))
    multi = len(DATA_DIR) > 1

    # Load from all directories
    all_records = []
    for data_dir, label in zip(DATA_DIR, labels):
        recs = load_records(data_dir, label)
        print(f"[{label}] loaded {len(recs)} records from {data_dir}")
        all_records.extend(recs)

    if not all_records:
        print("No matching data found!")
        return

    df_all = pd.DataFrame(all_records)
    print(f"Total records: {len(df_all)}")

    # Average over rank and idx
    group_cols = ['dir_label', 'method', 'ws', 'B', 'H', 'D', 'Operation']
    df_avg = df_all.groupby(group_cols, dropna=False)[TFLOPS_COLS].mean().reset_index()

    # Figures grouped by (B, H, D) — in multi-dir mode, different methods
    # from different directories are plotted together.
    plot_keys = df_avg.groupby(['B', 'H', 'D'], dropna=False).size().reset_index()[['B', 'H', 'D']]

    for _, pk in plot_keys.iterrows():
        b_val, h_val, d_val = pk['B'], pk['H'], pk['D']

        mask = (df_avg['H'] == h_val) & (df_avg['D'] == d_val)
        if pd.isna(b_val):
            mask = mask & (df_avg['B'].isna())
            b_str = 'noB'
        else:
            mask = mask & (df_avg['B'] == b_val)
            b_str = f'B{int(b_val)}'

        subset = df_avg[mask]
        if subset.empty:
            continue

        dir_labels_in_subset = sorted(subset['dir_label'].unique())
        operations = sorted(subset['Operation'].unique())
        ws_values = sorted(subset['ws'].unique())
        methods_in_subset = sorted(subset['method'].unique())
        title_method = ' vs '.join(methods_in_subset)

        # Visual encoding:
        #   Each directory (method) gets a fixed color — same across all subplots.
        #   Each operation gets a distinct marker shape.
        n_ops = len(operations)
        op_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

        # One fixed color per dir_label
        n_dirs = len(dir_labels_in_subset)
        base_colors = list(plt.cm.tab10.colors[:n_dirs])

        # Rows = operations, Columns = FW/BW/TOTAL
        # Within each subplot, different dir_labels (methods) are separate lines
        n_rows = len(operations)

        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows), squeeze=False)
        fig.suptitle(f'{title_method}  H={h_val} D={d_val} {b_str}  (single-card seqlen=8k)',
                     fontsize=14, fontweight='bold')

        for row_idx, op in enumerate(operations):
            op_idx = row_idx
            marker = op_markers[op_idx % len(op_markers)]

            for col_idx, tflops_col in enumerate(TFLOPS_COLS):
                ax = axes[row_idx][col_idx]

                for dl_idx, dl in enumerate(dir_labels_in_subset):
                    op_data = subset[(subset['dir_label'] == dl) & (subset['Operation'] == op)]
                    xs, ys = [], []
                    for ws_val in ws_values:
                        row = op_data[op_data['ws'] == ws_val]
                        if not row.empty:
                            xs.append(ws_val)
                            ys.append(row[tflops_col].values[0])
                    if not xs:
                        continue

                    color = base_colors[dl_idx]
                    label = dl
                    ax.plot(xs, ys, color=color, marker=marker, markersize=8,
                            linewidth=2.2, linestyle='-', label=label)

                    # Annotate each point with its value
                    for x, y in zip(xs, ys):
                        ax.annotate(f'{y:.1f}', (x, y),
                                    textcoords='offset points', xytext=(0, 8),
                                    ha='center', fontsize=8, color=color)

                ax.set_xticks(ws_values)
                ax.set_xticklabels([str(w) for w in ws_values])
                ax.grid(True, alpha=0.25, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(fontsize=7, loc='best', framealpha=0.9, edgecolor='#ccc')

                # Column title (only top row)
                if row_idx == 0:
                    ax.set_title(f'{TFLOPS_LABELS[tflops_col]} TFLOPs/s', fontsize=13)

                # Row label (only left column)
                if col_idx == 0:
                    ax.set_ylabel(f'{op}\nTFLOPs/s', fontsize=11)
                else:
                    ax.set_ylabel('TFLOPs/s', fontsize=11)

                # X label (only bottom row)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('WORLD_SIZE', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_name = f'tflops_8k_{"_".join(methods_in_subset)}_{b_str}_H{h_val}_D{d_val}'
        out_dir = DATA_DIR[0]
        out_path = os.path.join(out_dir, f'{out_name}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)

    print("\nDone!")


if __name__ == '__main__':
    main()
