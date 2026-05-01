#!/usr/bin/env python3
"""
Plot TFLOPs/s vs cp_size for fixed local sequence length (S // cp_size).

Reads TSV files from the output directory, filters cases where
S // cp_size == local_seqlen (default 8192), groups by (mode, B, H, D),
uses different masks as separate lines with distinct markers,
averages TFLOPS across ranks and idx, plots vs cp_size.

Usage:
    python draw_singlecard_cp.py --input_dir bf16_singlecard_test
    python draw_singlecard_cp.py --input_dir bf16_singlecard_test --local_seqlen 16384
    python draw_singlecard_cp.py --input_dir bf16_singlecard_test --global_density
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
    parser = argparse.ArgumentParser(
        description="Plot singlecard CP benchmark TFLOPs/s vs cp_size (fixed local seqlen)")
    parser.add_argument("--input_dir", type=str, default="bf16_singlecard_test",
                        help="Directory containing the CSV/TSV result files")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save plots (default: same as input_dir)")
    parser.add_argument("--global_density", action="store_true",
                        help="Use global-density TFLOPs (G.FW/G.BW/G.TOTAL) instead of local")
    parser.add_argument("--local_seqlen", type=int, default=8192,
                        help="Target local sequence length per rank (S // cp_size)")
    return parser.parse_args()


def parse_filename(fname):
    """
    Parse filename to extract (mode, mask, B, S, H, D, idx, cp_size).
    """
    base = os.path.basename(fname).replace('.csv', '')

    m = re.match(
        r'flashmask_cp_singlecard_([a-z_]+?)_([A-Z][A-Za-z_]+Mask)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_cp(\d+)$',
        base)
    if m:
        mode, mask, B, S, H, D, idx, cp = m.groups()
        return mode, mask, int(B), int(S), int(H), int(D), int(idx), int(cp)

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
    local_seqlen = args.local_seqlen

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

    # Structure: plot_key = (mode, B, H, D) -> mask -> cp_size -> [per-file avg tflops dict]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped = 0

    for fpath in files:
        parsed = parse_filename(fpath)
        if parsed is None:
            continue
        mode, mask, B, S, H, D, idx, cp = parsed

        # Filter: only keep cases where S // cp_size == local_seqlen
        if S // cp != local_seqlen:
            skipped += 1
            continue

        plot_key = (mode, B, H, D)

        headers, rows = read_tsv(fpath)
        if headers is None or not rows:
            continue

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
            groups[plot_key][mask][cp].append(avg)

    if not groups:
        print(f"No valid data with S // cp_size == {local_seqlen}. (skipped {skipped} files)")
        return

    os.makedirs(output_dir, exist_ok=True)

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    for plot_key, mask_data in sorted(groups.items()):
        mode, B, H, D = plot_key
        masks = sorted(mask_data.keys())

        all_cp = sorted(set(cp for m in masks for cp in mask_data[m]))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f'mode={mode}  B={B}  H={H}  D={D}  local_seqlen={fmt_s(local_seqlen)}\n({density_tag})',
            fontsize=14, fontweight='bold')

        for ax_idx, tflops_col in enumerate(tflops_cols):
            ax = axes[ax_idx]
            label_name = tflops_labels[tflops_col]

            for mask_idx, mask in enumerate(masks):
                xs = []
                ys = []
                for cp in all_cp:
                    if cp in mask_data[mask] and mask_data[mask][cp]:
                        vals = [d[tflops_col] for d in mask_data[mask][cp] if tflops_col in d]
                        if vals:
                            xs.append(cp)
                            ys.append(np.mean(vals))

                color = colors[mask_idx % len(colors)]
                marker = markers[mask_idx % len(markers)]
                ax.plot(xs, ys, marker=marker, color=color, label=mask,
                        linewidth=2, markersize=8)

            ax.set_xlabel('cp_size', fontsize=12)
            ax.set_ylabel('TFLOPs/s', fontsize=12)
            ax.set_title(f'{label_name} TFLOPs/s', fontsize=13)
            ax.set_xticks(all_cp)
            ax.set_xticklabels([str(c) for c in all_cp])
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out_name = f'tflops_singlecard_cp_{mode}_B{B}_H{H}_D{D}_local{fmt_s(local_seqlen)}_{density_tag}'
        for ext in ['png', 'pdf']:
            out_path = os.path.join(output_dir, f'{out_name}.{ext}')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_path}")
        plt.close(fig)

    print(f"\nDone! {len(groups)} figure(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
