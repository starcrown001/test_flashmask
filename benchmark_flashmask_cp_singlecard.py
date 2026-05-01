#!/usr/bin/env python3
"""
Single-card FlashMask CP Benchmark

Simulates context parallelism on a single GPU by emulating multiple
virtual ranks.  Supports 4 modes (matching benchmark_flashmask_cp_unified.py):

  baseline:         DualChunkSwap Q + uniform K/V + global mask
  overlap:          (same data prep as baseline; overlap only matters for comm)
  balance:          balanced_swap Q/K/V via balance_flashmask_input + local mask
  balance_overlap:  (same data prep as balance)

For each simulated rank:
  q_local  x  k_gathered  x  v_gathered  -->  flashmask_attention
Reports per-rank TFLOPS with per-rank density.
"""

import numpy as np
from functools import partial
from tabulate import tabulate
import paddle
import os
import sys

from paddle.nn.functional.flash_attention import flashmask_attention
from sparsity_utils import flashmask_block_sparsity


def flashmask_block_sparsity_nonsquare(
    causal, flashmask, Sq, Sk,
    Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
):
    """
    Compute block-level sparsity for non-square attention matrix (Sq x Sk).

    Same logic as flashmask_block_sparsity but supports Sq != Sk.
    flashmask shape: (B, H, Sk, bounds), 3rd dim is K positions,
    values are Q-row indices into [0, Sq].
    """
    if flashmask is None and not causal:
        return 0.0

    Br = Q_BLOCK_SIZE
    Bc = KV_BLOCK_SIZE
    Tr = Sq // Br
    Tc = Sk // Bc
    print("Br, Bc, Tr, Tc", Br, Bc, Tr, Tc)

    if flashmask is None and causal:
        total_size = Sq * Sk
        num_sparse_blocks = 0
        for i in range(Tr):
            for j in range(Tc):
                if j > i:
                    num_sparse_blocks += 1
        return float((num_sparse_blocks * Bc * Br) / total_size)

    if hasattr(flashmask, 'cpu'):
        fm = flashmask.cpu().detach().numpy()
    elif hasattr(flashmask, 'numpy'):
        fm = flashmask.numpy()
    else:
        fm = np.asarray(flashmask)

    B, H, Sk_from_mask = fm.shape[0], fm.shape[1], fm.shape[2]
    bounds = fm.shape[-1]

    # The 3rd dimension of flashmask is the K dimension (key positions),
    # NOT the Q dimension. Keep Sq from caller parameter; use mask shape for Sk.
    # Tr = Sq // Br stays as-is (from caller's Sq).
    # Tc = Sk // Bc stays as-is (from caller's Sk).

    LTS = LTE = UTS = UTE = None
    if bounds == 4:
        LTS, LTE, UTS, UTE = fm[..., 0], fm[..., 1], fm[..., 2], fm[..., 3]
    elif bounds == 2 and causal:
        LTS, LTE = fm[..., 0], fm[..., 1]
    elif bounds == 2 and not causal:
        LTS, UTE = fm[..., 0], fm[..., 1]
    else:
        LTS = fm[..., 0]

    if LTS is None:
        LTS = np.full((B, H, Sk), Sq, dtype=np.int32)
    if LTE is None:
        LTE = np.full((B, H, Sk), Sq, dtype=np.int32)
    if UTS is None:
        UTS = np.full((B, H, Sk), 0, dtype=np.int32)
    if UTE is None:
        UTE = np.tile(np.arange(Sk, dtype=np.int32).reshape(1, 1, Sk), (B, H, 1))

    # Per KV-block stats: reshape along K dim with Bc, reduce to per-block min/max
    # Shape after reshape+reduce: (B, H, Tc)
    LTStartMax = LTS.reshape(B, H, Tc, Bc).max(axis=-1)
    LTStartMin = LTS.reshape(B, H, Tc, Bc).min(axis=-1)
    LTEndMax   = LTE.reshape(B, H, Tc, Bc).max(axis=-1)
    LTEndMin   = LTE.reshape(B, H, Tc, Bc).min(axis=-1)
    UTStartMax = UTS.reshape(B, H, Tc, Bc).max(axis=-1)
    UTStartMin = UTS.reshape(B, H, Tc, Bc).min(axis=-1)
    UTEndMax   = UTE.reshape(B, H, Tc, Bc).max(axis=-1)
    UTEndMin   = UTE.reshape(B, H, Tc, Bc).min(axis=-1)

    num_dense_blocks = 0
    for bsz in range(B):
        for head in range(H):
            for i in range(Tr):
                for j in range(Tc):
                    if causal and j > i:
                        continue
                    # Fully inside lower-triangle mask -> sparse
                    if (i * Br >= LTStartMax[bsz, head, j]
                            and (i + 1) * Br <= LTEndMin[bsz, head, j]):
                        continue
                    # Fully inside upper-triangle mask -> sparse
                    if (i * Br >= UTStartMax[bsz, head, j]
                            and (i + 1) * Br <= UTEndMin[bsz, head, j]):
                        continue
                    num_dense_blocks += 1

    total_blocks = Tr * Tc
    num_sparse_blocks = B * H * total_blocks - num_dense_blocks
    total_size = B * H * Sq * Sk
    return float((num_sparse_blocks * Bc * Br) / total_size)


# Add benchmark_flashmask_cp to sys.path for balance imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmark_flashmask_cp'))
from flash_mask.cp_balance import balance_flashmask_input
from flash_mask.cp_balance import balance_flashmask_input_inter_machine

from argparse import ArgumentParser


# ---------------------------------------------------------------------------
# FLOPS helpers (from benchmark_flashmask_cp_unified.py)
# ---------------------------------------------------------------------------
def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def cal_tflops(flops, time_ms):
    return flops * (1e3 / time_ms) / 1e12


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    parser = ArgumentParser(description="Single-card FlashMask CP Benchmark")
    parser.add_argument("--mode", type=str, default="balance",
                        choices=["baseline", "overlap", "balance", "balance_overlap"],
                        help="CP mode: baseline (DualChunkSwap), overlap, balance, balance_overlap")
    parser.add_argument("--examples", type=str, nargs="+", default=["all"])
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=8,
                        help="Simulated CP size (number of virtual ranks)")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--input_file", type=str, default="kernel_test_dist_seq_info.txt")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    return parser.parse_args()


args = get_args()
WARM_UP = args.warmup
BENCH_TIME = args.rep


# ---------------------------------------------------------------------------
# Local scatter_balance simulation (no distributed communication)
# ---------------------------------------------------------------------------
def dual_chunk_local(tensor, cp_size, rank, axis=1):
    """Simulate scatter_balance(mode="dual_chunk") on single card.

    Each rank gets two chunks of size interval = seq_len // (cp_size * 2):
      - chunk_start: [rank * interval : (rank+1) * interval]
      - chunk_end:   [total - (rank+1) * interval : total - rank * interval]
    """
    seq_len = tensor.shape[axis]
    assert seq_len % (cp_size * 2) == 0, \
        f"seq_len={seq_len} not divisible by cp_size*2={cp_size * 2}"
    interval = seq_len // (cp_size * 2)
    total_len = seq_len

    chunk_start = paddle.slice(tensor, axes=[axis],
                               starts=[interval * rank],
                               ends=[interval * (rank + 1)])
    chunk_end = paddle.slice(tensor, axes=[axis],
                             starts=[total_len - interval * (rank + 1)],
                             ends=[total_len - interval * rank])
    return paddle.concat([chunk_start, chunk_end], axis=axis)


def scatter_balance_local(tensor, buckets, cp_size, rank, axis=1):
    """Simulate scatter_balance(mode="balanced_swap") on single card.

    Uses buckets[rank] to determine which chunks this rank gets,
    slices them from the full tensor and concatenates.
    """
    seq_len = tensor.shape[axis]
    n_chunks_per_rank = len(buckets[rank])
    assert seq_len % (cp_size * n_chunks_per_rank) == 0, \
        f"seq_len={seq_len} not divisible by cp_size*n_chunks={cp_size * n_chunks_per_rank}"
    balance_chunksize = seq_len // (cp_size * n_chunks_per_rank)

    local_chunks = []
    for (_, idx) in buckets[rank]:
        chunk_start = idx * balance_chunksize
        chunk_end = (idx + 1) * balance_chunksize
        chunk = paddle.slice(tensor, axes=[axis], starts=[chunk_start], ends=[chunk_end])
        local_chunks.append(chunk)
    return paddle.concat(local_chunks, axis=axis)


# ---------------------------------------------------------------------------
# Single-card benchmark core (adapted from benchmark_flashmask_cp_unified.py)
# ---------------------------------------------------------------------------
def do_bench_singlecard(attn_fn, warmup=WARM_UP, rep=BENCH_TIME, fast_flush=True):
    """
    Benchmark using CUDA event timing on a single card.
    1 fwd + 1 bwd per iteration.
    """
    # Dry run
    out = attn_fn()
    paddle.device.synchronize()
    o_grad = paddle.ones_like(out)
    out.backward(o_grad)
    paddle.device.synchronize()

    # L2 cache flush buffer
    if fast_flush:
        cache = paddle.empty([int(256e6 // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(256e6)], dtype=paddle.int8)

    # Warm-up
    for _ in range(warmup):
        out = attn_fn()
        out.backward(o_grad)
    paddle.device.synchronize()

    # Create CUDA events
    fwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(rep)]
    fwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(rep)]
    bwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(rep)]
    bwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(rep)]

    # Benchmark
    for i in range(rep):
        cache.zero_()

        fwd_start_event[i].record()
        out = attn_fn()
        fwd_end_event[i].record()

        bwd_start_event[i].record()
        out.backward(o_grad)
        bwd_end_event[i].record()

    paddle.device.synchronize()

    fwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(fwd_start_event, fwd_end_event)],
        dtype=paddle.float32,
    )
    bwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(bwd_start_event, bwd_end_event)],
        dtype=paddle.float32,
    )

    fwd_time = paddle.mean(fwd_times).item()
    bwd_time = paddle.mean(bwd_times).item()
    return fwd_time, bwd_time


# ---------------------------------------------------------------------------
# Prepare inputs for a simulated rank
# ---------------------------------------------------------------------------
def prepare_inputs_singlecard(query, key, value, startend_row_indices,
                               cp_size, rank, mode):
    """
    Prepare local Q/K/V and mask for a simulated rank.

    baseline/overlap:
      Q via DualChunkSwap, K/V uniform split, global mask.
    balance/balance_overlap:
      Q/K/V via balanced_swap (balance_flashmask_input), local mask.
    """
    if mode in ["baseline", "overlap"]:
        # Q: DualChunkSwap
        q_local = dual_chunk_local(query, cp_size, rank, axis=1).detach().contiguous()
        # K/V: Uniform split
        k_blocksize = key.shape[1] // cp_size
        k_local = key[:, rank * k_blocksize:(rank + 1) * k_blocksize, :, :].detach().contiguous()
        v_local = value[:, rank * k_blocksize:(rank + 1) * k_blocksize, :, :].detach().contiguous()
        # Global mask (unchanged)
        local_mask = startend_row_indices

    elif mode in ["balance", "balance_overlap"]:
        # Get balanced mask and bucket assignment
        local_mask, buckets = balance_flashmask_input(
            startend_row_indices, cp_size, rank)
        print(buckets, local_mask)
        # Q/K/V split via scatter_balance(mode="balanced_swap")
        q_local = scatter_balance_local(query, buckets, cp_size, rank, axis=1).detach().contiguous()
        k_local = scatter_balance_local(key, buckets, cp_size, rank, axis=1).detach().contiguous()
        v_local = scatter_balance_local(value, buckets, cp_size, rank, axis=1).detach().contiguous()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Enable gradients
    q_local.stop_gradient = False
    k_local.stop_gradient = False
    v_local.stop_gradient = False

    return q_local, k_local, v_local, local_mask


# ---------------------------------------------------------------------------
# Test entry: benchmark one mask type across all simulated ranks
# ---------------------------------------------------------------------------
def test_cp_flashmask_singlecard(generate_mask_fn, B=1, S=8192, H=1, D=128,
                                  dtype='bf16', cp_size=8, mode='balance'):
    paddle.seed(2024)

    num_head = H
    num_head_q = 8 * H
    head_size = D

    # Generate global mask
    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        startend_row_indices, causal = generate_mask_fn(B, S, num_head, head_size)

    # Create global Q/K/V
    query = paddle.randn([B, S, num_head_q, head_size], dtype=paddle.bfloat16)
    key = paddle.randn([B, S, num_head, head_size], dtype=paddle.bfloat16)
    value = paddle.randn([B, S, num_head, head_size], dtype=paddle.bfloat16)

    # Global density (full S x S attention matrix, same as unified.py)
    global_sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, num_head, S)
    global_density = 1.0 - global_sparsity
    print(f"Global density: {global_density:.4f}")

    per_rank_results = []

    # Pre-compute all ranks' local Q/K/V and masks
    all_rank_data = []
    for rank in range(cp_size):
        q_local, k_local, v_local, local_mask = prepare_inputs_singlecard(
            query, key, value, startend_row_indices,
            cp_size, rank, mode)
        all_rank_data.append((q_local, k_local, v_local, local_mask))

    # Build gathered K/V: concatenate all ranks' k_local/v_local along seqlen (axis=1)
    # This simulates the all_gather_kv in real CP
    k_gathered = paddle.concat([d[1] for d in all_rank_data], axis=1).detach().contiguous()
    v_gathered = paddle.concat([d[2] for d in all_rank_data], axis=1).detach().contiguous()
    k_gathered.stop_gradient = False
    v_gathered.stop_gradient = False

    for rank in range(cp_size):
        q_local, k_local, v_local, local_mask = all_rank_data[rank]

        Sq_local = q_local.shape[1]
        Sk_gathered = k_gathered.shape[1]

        # Per-rank density: compute sparsity for this rank's Sq_local x Sk_gathered attention
        sparsity = flashmask_block_sparsity_nonsquare(
            False, local_mask, Sq_local, Sk_gathered)
        density = 1.0 - sparsity

        print(local_mask.shape, q_local.shape, k_gathered.shape)
        attn_fn = partial(
            flashmask_attention,
            q_local, k_gathered, v_gathered,
            startend_row_indices=local_mask,
            causal=False,  # mask handles causal pattern already
        )

        print(f"  Simulated rank {rank}: q_shape={q_local.shape}, "
              f"kv_gathered_shape={k_gathered.shape}")

        fwd_time, bwd_time = do_bench_singlecard(
            attn_fn, warmup=WARM_UP, rep=BENCH_TIME)

        total_time = fwd_time + bwd_time

        # TFLOPS per rank: q_local x k_gathered (simulates allgather_kv)
        fwd_tflops = cal_tflops(
            density * cal_flops(B, num_head_q, Sq_local, Sk_gathered, D, mode='fwd'), fwd_time)
        bwd_tflops = cal_tflops(
            density * cal_flops(B, num_head_q, Sq_local, Sk_gathered, D, mode='bwd'), bwd_time)
        total_tflops = cal_tflops(
            density * cal_flops(B, num_head_q, Sq_local, Sk_gathered, D, mode='fwd_bwd'), total_time)

        # TFLOPS using global density (full S x S / cp_size, same formula as unified.py)
        global_fwd_tflops = cal_tflops(
            global_density * cal_flops(B, num_head_q, S, S, D, mode='fwd') / cp_size, fwd_time)
        global_bwd_tflops = cal_tflops(
            global_density * cal_flops(B, num_head_q, S, S, D, mode='bwd') / cp_size, bwd_time)
        global_total_tflops = cal_tflops(
            global_density * cal_flops(B, num_head_q, S, S, D, mode='fwd_bwd') / cp_size, total_time)

        per_rank_results.append({
            'rank': rank,
            'fwd_time': fwd_time,
            'bwd_time': bwd_time,
            'total_time': total_time,
            'fwd_tflops': fwd_tflops,
            'bwd_tflops': bwd_tflops,
            'total_tflops': total_tflops,
            'density': density,
            'global_density': global_density,
            'global_fwd_tflops': global_fwd_tflops,
            'global_bwd_tflops': global_bwd_tflops,
            'global_total_tflops': global_total_tflops,
        })

    return per_rank_results


# ---------------------------------------------------------------------------
# Mask generators (same as benchmark_flashmask_cp_unified.py)
# ---------------------------------------------------------------------------
def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"sequence_length must be >= {num_answers + 1}")
    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    return [base + (1 if i < extra else 0) for i in range(num_answers + 1)]


def generate_causal_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S, f"{total_seq_len=}, {S=}"
    padding = S - total_seq_len
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    lts = np.repeat(seq_cusums, doc_seq_lens)
    lts = paddle.to_tensor(lts, dtype=paddle.int32).reshape((1, 1, S, 1))
    ute = paddle.arange(S, dtype='int32').reshape((1, 1, S, 1))
    startend_row_indices = paddle.concat([lts, ute], axis=-1)
    startend_row_indices = startend_row_indices.repeat_interleave(B, 0)
    return startend_row_indices, False


def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    down_left_row_indices = paddle.to_tensor(
        down_left_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(
        up_right_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    return startend_row_indices, False


def generate_prefix_lm_document_mask(B, S, H, D,
                                     doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    assert len(doc_seq_lens) >= 2
    total_seq_len = sum(seq_length for _, seq_length in doc_seq_lens)
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(
        down_left_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend(
            [cur_len_so_far] * prefix_length +
            list(range(cur_len_so_far + prefix_length, cur_len_so_far + seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(
        up_right_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    return startend_row_indices, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    cp_size = args.cp_size
    mode = args.mode

    print(f"Single-card FlashMask CP Benchmark — mode={mode}, cp_size={cp_size}, "
          f"dtype={args.dtype}, warmup={WARM_UP}, rep={BENCH_TIME}")
    if mode in ["balance", "balance_overlap"]:
        print(f"Balance params: balance_flashmask_input (from unified.py)")

    # Read input file
    total_length = 0
    doc_seq_lens_list = []
    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'Total length' in line:
                total_length = int(line.split(":")[1].split(',')[0].strip())
            else:
                doc_list = eval(line.split(":")[-1].split("#")[0].strip())
                qksparse_mask = eval(line.split(":")[-1].split("#")[1].strip())
                doc_seq_lens_list.append((total_length, doc_list, qksparse_mask))

    for H in [args.num_heads]:
        for D in [128]:
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                if((S // cp_size )<= 4096):
                    continue
                B = args.batch
                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]

                print(f"\n{'='*80}")
                print(f"Config: B={B}, S={S}, H={H}, D={D}, idx={idx}, cp_size={cp_size}")
                print(f"{'='*80}")

                available_examples = {
                    "Causal Document Mask": lambda: test_cp_flashmask_singlecard(
                        generate_mask_fn=partial(
                            generate_causal_document_mask, doc_seq_lens=doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype, cp_size=cp_size, mode=mode),
                    "Document Mask": lambda: test_cp_flashmask_singlecard(
                        generate_mask_fn=partial(
                            generate_document_mask, doc_seq_lens=doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype, cp_size=cp_size, mode=mode),
                    "Prefix LM Document Mask": lambda: test_cp_flashmask_singlecard(
                        generate_mask_fn=partial(
                            generate_prefix_lm_document_mask,
                            doc_seq_lens=prefix_doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype, cp_size=cp_size, mode=mode),
                }

                if "all" in args.examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = args.examples

                for ex in ex_to_run:
                    if ex not in available_examples:
                        print(f"Warning: Unknown example '{ex}'. Skipping.")
                        continue

                    print(f"\n--- {ex} ---")
                    per_rank_results = available_examples[ex]()

                    # Per-rank table
                    rows = []
                    for r in per_rank_results:
                        rows.append([
                            f"Rank {r['rank']}",
                            f"{r['fwd_time']:.4f}",
                            f"{r['bwd_time']:.4f}",
                            f"{r['total_time']:.4f}",
                            f"{r['fwd_tflops']:.4f}",
                            f"{r['bwd_tflops']:.4f}",
                            f"{r['total_tflops']:.4f}",
                            f"{r['density']:.4f}",
                            f"{r['global_density']:.4f}",
                            f"{r['global_fwd_tflops']:.4f}",
                            f"{r['global_bwd_tflops']:.4f}",
                            f"{r['global_total_tflops']:.4f}",
                        ])

                    # Summary row (max time across ranks, min tflops)
                    max_fwd = max(r['fwd_time'] for r in per_rank_results)
                    max_bwd = max(r['bwd_time'] for r in per_rank_results)
                    max_total = max(r['total_time'] for r in per_rank_results)
                    min_fwd_tflops = min(r['fwd_tflops'] for r in per_rank_results)
                    min_bwd_tflops = min(r['bwd_tflops'] for r in per_rank_results)
                    min_total_tflops = min(r['total_tflops'] for r in per_rank_results)
                    density = per_rank_results[0]['density']
                    g_density = per_rank_results[0]['global_density']
                    min_g_fwd_tflops = min(r['global_fwd_tflops'] for r in per_rank_results)
                    min_g_bwd_tflops = min(r['global_bwd_tflops'] for r in per_rank_results)
                    min_g_total_tflops = min(r['global_total_tflops'] for r in per_rank_results)
                    rows.append([
                        "MAX/MIN",
                        f"{max_fwd:.4f}",
                        f"{max_bwd:.4f}",
                        f"{max_total:.4f}",
                        f"{min_fwd_tflops:.4f}",
                        f"{min_bwd_tflops:.4f}",
                        f"{min_total_tflops:.4f}",
                        f"{density:.4f}",
                        f"{g_density:.4f}",
                        f"{min_g_fwd_tflops:.4f}",
                        f"{min_g_bwd_tflops:.4f}",
                        f"{min_g_total_tflops:.4f}",
                    ])

                    headers = ["Rank", "FW Time (ms)", "BW Time (ms)", "TOTAL Time (ms)",
                               "FW TFLOPs/s", "BW TFLOPs/s", "TOTAL TFLOPs/s", "Density",
                               "Global Density", "G.FW TFLOPs/s", "G.BW TFLOPs/s", "G.TOTAL TFLOPs/s"]
                    print(tabulate(rows, headers=headers, tablefmt="grid"))

                    # Save results — each mask type in its own file
                    out_dir = f"{args.dtype}_singlecard_test"
                    os.makedirs(out_dir, exist_ok=True)
                    content = tabulate(rows, headers=headers, tablefmt="tsv")
                    ex_tag = ex.replace(" ", "_")
                    out_path = os.path.join(
                        out_dir,
                        f"flashmask_cp_singlecard_{mode}_{ex_tag}_{B}_{S}_{H}_{D}_{idx}_cp{cp_size}.csv")
                    with open(out_path, "w") as f:
                        f.write(content)


if __name__ == "__main__":
    main()
