#!/usr/bin/env python3
"""
Plot TFLOPs/s line charts for single-card FlashMask CP benchmark results.

Layout: rows = mask types, columns = FW / BW / TOTAL.
Each subplot is one mask x one metric, X axis = sequence length S.
All masks appear in a single figure as separate rows.

Usage:
    python draw_singlecard.py --input_dir bf16_singlecard_test
    python draw_singlecard.py --input_dir bf16_singlecard_test --global_density
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot singlecard CP benchmark TFLOPs/s")
    parser.add_argument("--input_dir", type=str, default="bf16_singlecard_test",
                        help="Directory containing the CSV/TSV result files")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save plots (default: same as input_dir)")
    parser.add_argument("--global_density", action="store_true",
                        help="Use global-density TFLOPs (G.FW/G.BW/G.TOTAL) instead of local")
    return parser.parse_args()


def parse_filename(fname):
    """
    Parse filename to extract (mode, mask, B, S, H, D, idx, cp_size).

    Patterns:
      flashmask_cp_singlecard_{mode}_{mask}_{B}_{S}_{H}_{D}_{idx}_cp{cp_size}.csv
      flashmask_cp_singlecard_{mode}_{B}_{S}_{H}_{D}_{idx}_cp{cp_size}.csv
    """
    base = os.path.basename(fname).replace('.csv', '')

    # New format with mask tag (mask name starts with uppercase, contains underscores)
    m = re.match(
        r'flashmask_cp_singlecard_([a-z_]+?)_([A-Z][A-Za-z_]+Mask)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_cp(\d+)$',
        base)
    if m:
        mode, mask, B, S, H, D, idx, cp = m.groups()
        return mode, mask, int(B), int(S), int(H), int(D), int(idx), int(cp)

    # Old format without mask tag
    m = re.match(
        r'flashmask_cp_singlecard_([a-z_]+?)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_cp(\d+)$',
        base)
    if m:
        mode, B, S, H, D, idx, cp = m.groups()
        return mode, "Unknown", int(B), int(S), int(H), int(D), int(idx), int(cp)

    return None


def read_tsv(filepath):
    """Read TSV, return (headers, per-rank rows) excluding MAX/MIN row."""
    with open(filepath, 'r') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    if not lines:
        return None, []
    headers = [h.strip() for h in lines[0].split('\t')]
    rows = []
    for line in lines[1:]:
        cols = [c.strip() for c in line.split('\t')]
        if cols[0].startswith("MAX") or cols[0].startswith("MIN"):
            continue
        rows.append(cols)
    return headers, rows


def fmt_s(s):
    if s >= 1024 * 1024:
        return f"{s // (1024*1024)}M"
    elif s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else args.input_dir

    if args.global_density:
        tflops_cols = ['G.FW TFLOPs/s', 'G.BW TFLOPs/s', 'G.TOTAL TFLOPs/s']
        tflops_labels = {'G.FW TFLOPs/s': 'G.FW', 'G.BW TFLOPs/s': 'G.BW', 'G.TOTAL TFLOPs/s': 'G.TOTAL'}
        density_tag = "global_density"
    else:
        tflops_cols = ['FW TFLOPs/s', 'BW TFLOPs/s', 'TOTAL TFLOPs/s']
        tflops_labels = {'FW TFLOPs/s': 'FW', 'BW TFLOPs/s': 'BW', 'TOTAL TFLOPs/s': 'TOTAL'}
        density_tag = "local_density"

    # Collect all CSV files
    pattern = os.path.join(args.input_dir, "flashmask_cp_singlecard_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching {pattern}")
        return

    # Structure: plot_key = (mode, B, H, D, cp) -> mask -> S -> [per-file avg tflops dict]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for fpath in files:
        parsed = parse_filename(fpath)
        if parsed is None:
            continue
        mode, mask, B, S, H, D, idx, cp = parsed
        plot_key = (mode, B, H, D, cp)

        headers, rows = read_tsv(fpath)
        if headers is None or not rows:
            continue

        # Find column indices
        col_indices = {}
        for col in tflops_cols:
            try:
                col_indices[col] = headers.index(col)
            except ValueError:
                pass

        if not col_indices:
            continue

        # Average across all ranks in this file
        rank_vals = {col: [] for col in tflops_cols}
        for row in rows:
            for col in tflops_cols:
                if col in col_indices:
                    try:
                        rank_vals[col].append(float(row[col_indices[col]]))
                    except (ValueError, IndexError):
                        pass

        avg = {}
        for col in tflops_cols:
            if rank_vals[col]:
                avg[col] = np.mean(rank_vals[col])
        if avg:
            groups[plot_key][mask][S].append(avg)

    if not groups:
        print("No valid data parsed.")
        return

    os.makedirs(output_dir, exist_ok=True)

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    for plot_key, mask_data in sorted(groups.items()):
        mode, B, H, D, cp = plot_key
        masks = sorted(mask_data.keys())
        n_masks = len(masks)

        # Collect all S values across masks
        all_s = sorted(set(s for m in masks for s in mask_data[m]))

        # Create figure: n_masks rows x 3 columns
        fig, axes = plt.subplots(n_masks, 3, figsize=(18, 5 * n_masks),
                                 squeeze=False)
        fig.suptitle(
            f'mode={mode}  B={B}  H={H}  D={D}  cp_size={cp}\n({density_tag})',
            fontsize=14, fontweight='bold')

        for row_idx, mask in enumerate(masks):
            for col_idx, tflops_col in enumerate(tflops_cols):
                ax = axes[row_idx][col_idx]
                label_name = tflops_labels[tflops_col]

                xs = []
                ys = []
                for s in all_s:
                    if s in mask_data[mask] and mask_data[mask][s]:
                        vals = [d[tflops_col] for d in mask_data[mask][s] if tflops_col in d]
                        if vals:
                            xs.append(s)
                            ys.append(np.mean(vals))

                color = colors[row_idx % len(colors)]
                marker = markers[row_idx % len(markers)]
                ax.plot(xs, ys, marker=marker, color=color,
                        linewidth=2, markersize=8)

                # Annotate each point with its value
                for x, y in zip(xs, ys):
                    ax.annotate(f'{y:.1f}', (x, y),
                                textcoords='offset points', xytext=(0, 8),
                                ha='center', fontsize=8, color=color)

                ax.set_xticks(all_s)
                ax.set_xticklabels([fmt_s(s) for s in all_s])
                ax.grid(True, alpha=0.25, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Column title (only top row)
                if row_idx == 0:
                    ax.set_title(f'{label_name} TFLOPs/s', fontsize=13)

                # Row label (only left column)
                if col_idx == 0:
                    ax.set_ylabel(f'{mask}\nTFLOPs/s', fontsize=11)
                else:
                    ax.set_ylabel('TFLOPs/s', fontsize=11)

                # X label (only bottom row)
                if row_idx == n_masks - 1:
                    ax.set_xlabel('Sequence Length (S)', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_name = f'tflops_singlecard_{mode}_B{B}_H{H}_D{D}_cp{cp}_{density_tag}'
        for ext in ['png', 'pdf']:
            out_path = os.path.join(output_dir, f'{out_name}.{ext}')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_path}")
        plt.close(fig)

    print(f"\nDone! {len(groups)} figure(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
