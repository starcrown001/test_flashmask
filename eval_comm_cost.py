#!/usr/bin/env python3
"""
Evaluate CP communication cost from dumped before/after balance masks.

Usage:
    python eval_comm_cost.py --mask_dir record_mask_0421
    python eval_comm_cost.py --mask_dir record_mask_0421 --block_size 256 --inter_node_cost 20
"""

import argparse
import os
import re
import numpy as np
import paddle
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from context_parallel_utils_new import preprocess_index_dual_chunks
from overlap_utils import rearrange_blocks

LOCAL_CHUNK_SIZE = 8192


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate CP communication cost")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing {before,after}_{rank}_mask_{idx}.npy files")
    parser.add_argument("--block_size", type=int, default=512,
                        help="KV block size for masked-or-not check (default: 512)")
    parser.add_argument("--inter_node_cost", type=float, default=15,
                        help="Cost multiplier for inter-node communication (default: 15)")
    parser.add_argument("--gpus_per_node", type=int, default=8,
                        help="GPUs per node, used to determine intra/inter-node (default: 8)")
    parser.add_argument("--markdown", action="store_true",
                        help="Output tables in markdown format for easy copy-paste")
    parser.add_argument("--plot", action="store_true",
                        help="Generate mask pattern visualization")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mask preprocessing (before-mask only: dual_chunk layout reconstruction)
# ---------------------------------------------------------------------------
def preprocess_before_mask(mask_np, cp_size, rank):
    """Convert global before-mask to dual_chunk layout.

    After this, [8192*i : 8192*(i+1)] is rank i's KV data — same semantics
    as the after-mask from balance_flashmask_input.
    """
    mask_t = paddle.to_tensor(mask_np)
    seq_blocksize = mask_t.shape[2] // (2 * cp_size)  # e.g. 131072 / (2 * 16) = 4096
    mask_t = preprocess_index_dual_chunks(
        mask_t,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )
    mask_t = rearrange_blocks(mask_t, cp_size)
    return mask_t.numpy()


# ---------------------------------------------------------------------------
# Core cost computation
# ---------------------------------------------------------------------------
def compute_rank_cost(mask_np, rank, cp_size, block_size, gpus_per_node, inter_node_cost):
    """Compute communication cost for one rank.

    For each remote chunk i (i != rank), count how many sub-blocks of size
    `block_size` need to be transferred (i.e. are NOT fully masked), then
    weight by intra-node (×1) or inter-node (×inter_node_cost).
    """
    cost = 0
    for i in range(cp_size):
        if i == rank:
            continue
        # Count non-masked sub-blocks in chunk i
        chunk_start = i * LOCAL_CHUNK_SIZE
        counter = 0
        for b_start in range(chunk_start, chunk_start + LOCAL_CHUNK_SIZE, block_size):
            block = mask_np[0, 0, b_start:b_start + block_size, :]
            # start <= end for all elements → fully masked → no transfer
            if not (block[:, 0] <= block[:, 1]).all():
                counter += 1
        # Intra-node vs inter-node
        multiplier = 1 if (i // gpus_per_node == rank // gpus_per_node) else inter_node_cost
        cost += counter * multiplier
    return cost


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------
def scan_mask_files(mask_dir):
    """Scan directory and group files by (idx, mask_name).

    Filename format: {before|after}_{rank}_mask_{idx}_{mask_name}.npy
    Returns: {(idx, mask_name): {"before": {rank: path}, "after": {rank: path}}}
    """
    pattern = re.compile(r"^(before|after)_(\d+)_mask_(\d+)_(.+)\.npy$")
    groups = {}
    for fname in os.listdir(mask_dir):
        m = pattern.match(fname)
        if not m:
            continue
        kind, rank, idx, mask_name = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        key = (idx, mask_name)
        groups.setdefault(key, {"before": {}, "after": {}})
        groups[key][kind][rank] = os.path.join(mask_dir, fname)
    return groups


# ---------------------------------------------------------------------------
# Block-reduce: mask [1,1,S,2] → boolean array [S//block_size]
# ---------------------------------------------------------------------------
def reduce_mask(mask_np, block_size):
    """Reduce mask to per-block boolean: True = needs computation (not fully masked)."""
    S = mask_np.shape[2]
    n_blocks = S // block_size
    result = np.zeros(n_blocks, dtype=bool)
    for i in range(n_blocks):
        blk = mask_np[0, 0, i * block_size:(i + 1) * block_size, :]
        # If any element has start > end → block is active (not fully masked)
        result[i] = not (blk[:, 0] <= blk[:, 1]).all()
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _render_panel(ax, grid, ranks, cp_size, blocks_per_chunk,
                  gpus_per_node, cmap, title):
    """Render one heatmap panel on *ax*. Caller ensures ax is square."""
    n_ranks = len(ranks)

    ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=3,
              interpolation='nearest', origin='upper')

    n_nodes = (cp_size + gpus_per_node - 1) // gpus_per_node

    # Chunk boundary lines (thin, subtle)
    for c in range(1, cp_size):
        x = c * blocks_per_chunk - 0.5
        ax.axvline(x, color='#9E9E9E', linewidth=0.4, alpha=0.35)
    for ri in range(1, n_ranks):
        ax.axhline(ri - 0.5, color='#9E9E9E', linewidth=0.4, alpha=0.35)

    # Node boundary lines (dashed, light)
    for nd in range(1, n_nodes):
        x = nd * gpus_per_node * blocks_per_chunk - 0.5
        ax.axvline(x, color='#C62828', linewidth=1.2, linestyle='--', alpha=0.35)
        y = nd * gpus_per_node - 0.5
        if y < n_ranks:
            ax.axhline(y, color='#C62828', linewidth=1.2, linestyle='--', alpha=0.35)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('KV chunk', fontsize=10, labelpad=6)
    ax.set_ylabel('Rank', fontsize=10, labelpad=6)

    ax.set_yticks(range(n_ranks))
    ax.set_yticklabels([str(r) for r in ranks], fontsize=7)

    xtick_pos = [c * blocks_per_chunk + blocks_per_chunk / 2 - 0.5
                 for c in range(cp_size)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([str(c) for c in range(cp_size)], fontsize=7)
    ax.tick_params(axis='both', length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#BDBDBD')


def plot_mask_pattern(before_masks, after_masks, ranks, cp_size, block_size,
                      gpus_per_node, idx, mask_name, out_dir):
    """Plot before/after mask patterns side by side as two square subplots."""
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    S = cp_size * LOCAL_CHUNK_SIZE
    n_blocks = S // block_size
    blocks_per_chunk = LOCAL_CHUNK_SIZE // block_size
    n_ranks = len(ranks)

    cmap = ListedColormap(['#F5F5F5', '#4682B4', '#43A047', '#E53935'])

    def build_grid(masks_dict):
        grid = np.zeros((n_ranks, n_blocks), dtype=int)
        for ri, rank in enumerate(ranks):
            reduced = reduce_mask(masks_dict[rank], block_size)
            for bi in range(n_blocks):
                if not reduced[bi]:
                    continue
                chunk_owner = bi // blocks_per_chunk
                if chunk_owner == rank:
                    grid[ri, bi] = 1
                elif chunk_owner // gpus_per_node == rank // gpus_per_node:
                    grid[ri, bi] = 2
                else:
                    grid[ri, bi] = 3
        return grid

    has_before = len(before_masks) > 0
    n_panels = 2 if has_before else 1

    # Figure: each panel is a square of side `panel_side`
    panel_side = max(6, min(10, max(n_ranks, cp_size) * 0.5 + 2))
    gap = 0.8  # inches between panels
    fig_w = panel_side * n_panels + gap * (n_panels - 1) + 1.0  # extra for labels
    fig_h = panel_side + 1.6  # extra for suptitle + legend

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = gridspec.GridSpec(1, n_panels, figure=fig,
                           wspace=gap / panel_side * 1.2,
                           left=0.06, right=0.97,
                           bottom=0.10, top=0.88)

    panels = []
    if has_before:
        panels.append((gs[0, 0], before_masks, "Before (dual_chunk)"))
    panels.append((gs[0, n_panels - 1], after_masks, "After (balance)"))

    for gs_slot, masks_dict, title in panels:
        ax = fig.add_subplot(gs_slot)
        # Force square axes via adjustable box
        ax.set_box_aspect(1)
        grid = build_grid(masks_dict)
        _render_panel(ax, grid, ranks, cp_size, blocks_per_chunk,
                      gpus_per_node, cmap, title)

    # Legend — centered, clear
    legend_elements = [
        Patch(facecolor='#F5F5F5', edgecolor='#BDBDBD', label='Masked (no comm)'),
        Patch(facecolor='#4682B4', edgecolor='#3A6D96', label='Local (self)'),
        Patch(facecolor='#43A047', edgecolor='#2E7D32', label='Intra-node'),
        Patch(facecolor='#E53935', edgecolor='#B71C1C', label='Inter-node'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, frameon=True, fancybox=True,
               framealpha=0.95, edgecolor='#BDBDBD',
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f'idx={idx}   mask={mask_name}   S={S:,}   CP={cp_size}   block={block_size}',
        fontsize=14, fontweight='bold', color='#333333')

    out_path = os.path.join(out_dir, f'comm_pattern_idx{idx}_{mask_name}.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    tfmt = "pipe" if args.markdown else "grid"

    groups = scan_mask_files(args.mask_dir)
    if not groups:
        print(f"No mask files found in {args.mask_dir}")
        return

    def stats(vals):
        return {
            "Total": vals.sum(), "Mean": vals.mean(),
            "Max": vals.max(), "Min": vals.min(),
            "Max-Min": vals.max() - vals.min(), "Std": vals.std(),
        }

    all_before_vals = []
    all_after_vals = []

    for (idx, mask_name) in sorted(groups.keys()):
        entry = groups[(idx, mask_name)]
        after_files = entry["after"]
        before_files = entry["before"]

        if not after_files:
            print(f"[idx={idx}, mask={mask_name}] No after-mask files, skipping.")
            continue

        ranks = sorted(after_files.keys())
        # Infer cp_size from after mask shape
        sample = np.load(after_files[ranks[0]])
        S = sample.shape[2]
        cp_size = S // LOCAL_CHUNK_SIZE

        print(f"\n{'='*60}")
        print(f"idx={idx}, mask={mask_name}, S={S}, cp_size={cp_size}, "
              f"block_size={args.block_size}, inter_node_cost={args.inter_node_cost}")
        print(f"{'='*60}")

        # -- After-mask cost (per rank, load each rank's own file) --
        after_masks = {}
        after_costs = {}
        for rank in ranks:
            mask_np = np.load(after_files[rank])
            after_masks[rank] = mask_np
            after_costs[rank] = compute_rank_cost(
                mask_np, rank, cp_size,
                args.block_size, args.gpus_per_node, args.inter_node_cost)

        # -- Before-mask cost (global mask → preprocess per rank) --
        before_masks = {}
        before_costs = {}
        if before_files:
            # All ranks share the same global mask; just pick one
            sample_rank = sorted(before_files.keys())[0]
            global_mask = np.load(before_files[sample_rank])
            for rank in ranks:
                processed = preprocess_before_mask(global_mask, cp_size, rank)
                before_masks[rank] = processed
                before_costs[rank] = compute_rank_cost(
                    processed, rank, cp_size,
                    args.block_size, args.gpus_per_node, args.inter_node_cost)

        # Collect for global aggregate
        all_after_vals.extend(after_costs[r] for r in ranks)
        if before_costs:
            all_before_vals.extend(before_costs[r] for r in ranks)

        # -- Per-rank table --
        rows = []
        for rank in ranks:
            bc = before_costs.get(rank, "N/A")
            ac = after_costs[rank]
            delta = ac - bc if isinstance(bc, (int, float)) else "N/A"
            rows.append([rank, bc, ac, delta])

        print(tabulate(rows,
                       headers=["Rank", "Before Cost", "After Cost", "Delta"],
                       tablefmt=tfmt, floatfmt=".1f"))

        # -- Per-mask aggregate --
        agg_rows = []
        after_s = stats(np.array([after_costs[r] for r in ranks], dtype=float))
        if before_costs:
            before_s = stats(np.array([before_costs[r] for r in ranks], dtype=float))
            for key in ["Total", "Mean", "Max", "Min", "Max-Min", "Std"]:
                b, a = before_s[key], after_s[key]
                pct = (a - b) / b * 100 if b != 0 else float('nan')
                agg_rows.append([key, b, a, f"{pct:+.2f}%"])
        else:
            for key in ["Total", "Mean", "Max", "Min", "Max-Min", "Std"]:
                agg_rows.append([key, "N/A", after_s[key], "N/A"])

        print(f"\n--- Aggregate (idx={idx}, mask={mask_name}) ---")
        print(tabulate(agg_rows,
                       headers=["Metric", "Before", "After", "Δ%"],
                       tablefmt=tfmt, floatfmt=".2f"))

        # -- Visualization --
        if args.plot:
            plot_mask_pattern(before_masks, after_masks, ranks, cp_size,
                              args.block_size, args.gpus_per_node,
                              idx, mask_name, args.mask_dir)

    # -- Global aggregate across all masks --
    if len(all_after_vals) > 0 and len(sorted(groups.keys())) > 1:
        print(f"\n{'='*60}")
        print(f"Global Aggregate (all masks, all ranks)")
        print(f"{'='*60}")
        after_all = np.array(all_after_vals, dtype=float)
        global_rows = []
        if all_before_vals:
            before_all = np.array(all_before_vals, dtype=float)
            bs, as_ = stats(before_all), stats(after_all)
            for key in ["Total", "Mean", "Max", "Min", "Max-Min", "Std"]:
                b, a = bs[key], as_[key]
                pct = (a - b) / b * 100 if b != 0 else float('nan')
                global_rows.append([key, b, a, f"{pct:+.2f}%"])
        else:
            as_ = stats(after_all)
            for key in ["Total", "Mean", "Max", "Min", "Max-Min", "Std"]:
                global_rows.append([key, "N/A", as_[key], "N/A"])
        print(tabulate(global_rows,
                       headers=["Metric", "Before", "After", "Δ%"],
                       tablefmt=tfmt, floatfmt=".2f"))


if __name__ == "__main__":
    main()
