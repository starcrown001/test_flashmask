#!/usr/bin/env python3
"""
FlashMask CP Benchmark — Inter-Machine Communication-Aware Balance

Uses balance_flashmask_input_inter_machine (two-phase: computation balance + inter-machine comm swap).

Supports 4 modes via --mode:
  baseline:         DualChunkSwap + flashmask_attention_cp (allgather_kv)
  overlap:          DualChunkSwap + overlap_flashmask_attention (overlap)
  balance:          inter_machine_balance + flashmask_attention_cp (balance_q)
  balance_overlap:  inter_machine_balance + overlap_flashmask_attention (balance_q)

Benchmarking uses CUDA event timing, 1 fwd + 1 bwd per iteration.
Cross-rank timing aggregated via all_reduce(MAX).
"""

import numpy as np
from functools import partial
from tabulate import tabulate
import paddle
import os

from context_parallel_utils_new import scatter_balance, flashmask_attention_cp
from flash_mask.cp_balance import balance_flashmask_input_inter_machine
from overlap_utils import overlap_flashmask_attention
from sparsity_utils import flashmask_block_sparsity

import paddle.distributed.fleet as fleet
from argparse import ArgumentParser


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
    parser = ArgumentParser(description="FlashMask CP Benchmark — Inter-Machine Comm-Aware Balance")
    parser.add_argument("--mode", type=str, default="balance",
                        choices=["baseline", "overlap", "balance", "balance_overlap"])
    parser.add_argument("--examples", type=str, nargs="+", default=["all"])
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=0, help="Specify the CP size. 0 means hard-coded.")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--use_rs", action="store_true")
    parser.add_argument("--use_ipo", action="store_true")
    # Inter-machine balance hyperparameters
    parser.add_argument("--buckets_per_machine", type=int, default=8,
                        help="Number of buckets per machine (inter-machine scheduling granularity).")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Computation balance tolerance for inter-machine swap phase.")
    parser.add_argument("--max_swap_iterations", type=int, default=100,
                        help="Max iterations for inter-machine swap phase.")
    parser.add_argument("--use_locality_swap", action="store_true",
                        help="Enable locality-aware swap after inter-machine balance.")
    parser.add_argument("--max_locality_iterations", type=int, default=100,
                        help="Max iterations for locality swap phase.")
    return parser.parse_args()


args = get_args()

# ---------------------------------------------------------------------------
# Config & Fleet init (module-level, same as existing benchmarks)
# ---------------------------------------------------------------------------
output_prefix = f"flashmask_balancecomm_{args.mode}"
FIXED_LOCAL_LENGTH = 8192

if args.cp_size == 0:
    if args.batch == 1:
        input_file = 'kernel_test_dist_seq_info-128k.txt'
        cp_size = 16
        mp_size = 1
        sd_size = 16
    else:
        input_file = 'kernel_test_dist_seq_info-32k.txt'
        cp_size = 4
        mp_size = 4
        sd_size = 4
else:
    # make sure the CP group is an inter-node group
    input_file = 'kernel_test_dist_seq_info.txt'
    cp_size = args.cp_size
    mp_size = 1
    sd_size = args.cp_size
    print(f"Manual cp_size={cp_size}, mp_size={mp_size}, sd_size={sd_size}")

if args.profile:
    WARM_UP = 5
    BENCH_TIME = 20
else:
    WARM_UP = 50
    BENCH_TIME = 100

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": mp_size,
    "pp_degree": 1,
    "sharding_degree": sd_size,
    "sep_degree": 1,
    "ep_degree": cp_size,
    "moe_sharding_degree": 1,
    "cp_degree": cp_size,
    "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"]
}
print(strategy.hybrid_configs)
fleet.init(is_collective=True, strategy=strategy)
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------
def do_bench_flashmaskcp(attn_fn, cp_group, warmup=WARM_UP, rep=BENCH_TIME,
                         profile=False, rank=0, fast_flush=True):
    """
    Benchmark using CUDA event timing, 1 fwd + 1 bwd per iteration.

    Adapted from benchmark_flashmask_cp_overlap.py's do_bench_flashmaskcp.
    Uses paddle.device.Event for precise GPU timing, all_reduce as barrier,
    and takes max across ranks for synchronized measurement.
    """
    # Dry run (kernel compilation + o_grad creation)
    out = attn_fn()
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()
    print("FWD completed.")
    o_grad = paddle.ones_like(out)
    out.backward(o_grad)
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()
    print("BWD completed.")

    # L2 cache flush buffer
    if fast_flush:
        cache = paddle.empty([int(256e6 // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(256e6)], dtype=paddle.int8)

    n_warmup = warmup
    n_repeat = rep

    # Create CUDA events for per-iteration fwd/bwd timing
    fwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    fwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]

    # Warm-up (1 fwd + 1 bwd per iteration)
    for _ in range(n_warmup):
        out = attn_fn()
        out.backward(o_grad)
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()

    # Barrier before benchmark
    paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)

    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)

        if profile:
            paddle.base.core.nvprof_nvtx_push(f"flashmask_cp_fwd_{rank}")
        fwd_start_event[i].record()
        out = attn_fn()
        fwd_end_event[i].record()
        if profile:
            paddle.base.core.nvprof_nvtx_pop()

        if profile:
            paddle.base.core.nvprof_nvtx_push(f"flashmask_cp_bwd_{rank}")
        bwd_start_event[i].record()
        out.backward(o_grad)
        bwd_end_event[i].record()
        if profile:
            paddle.base.core.nvprof_nvtx_pop()

    # Final barrier
    paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)
    paddle.device.synchronize()

    # Compute times from CUDA events
    fwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(fwd_start_event, fwd_end_event)],
        dtype=paddle.float32,
    )
    bwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(bwd_start_event, bwd_end_event)],
        dtype=paddle.float32,
    )

    # Synchronize times across ranks (take max)
    paddle.distributed.all_reduce(fwd_times, op=paddle.distributed.ReduceOp.MAX, group=cp_group)
    paddle.distributed.all_reduce(bwd_times, op=paddle.distributed.ReduceOp.MAX, group=cp_group)

    fwd_time = paddle.mean(fwd_times).item()
    bwd_time = paddle.mean(bwd_times).item()
    return fwd_time, bwd_time


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_inputs(query, key, value, startend_row_indices,
                   cp_size, rank, cp_group, mode, use_ipo):
    """
    Prepare local Q/K/V and mask based on mode.

    All modes start with the same global Q/K/V and startend_row_indices.
    - baseline/overlap: Q via DualChunkSwap, K/V uniform, global mask
    - balance/balance_overlap: Q/K/V via balanced_swap, local mask
      (uses balance_flashmask_input_inter_machine with two-phase inter-machine optimization)
    """
    if mode in ["baseline", "overlap"]:
        # Q: DualChunkSwap via scatter_balance
        q_local = scatter_balance(query, mode="dual_chunk", axis = 1, group=cp_group).detach().contiguous()
        # K/V: Uniform splitting
        k_blocksize = key.shape[1] // cp_size
        k_local = key[:, rank * k_blocksize:(rank + 1) * k_blocksize, :, :].detach().contiguous()
        v_local = value[:, rank * k_blocksize:(rank + 1) * k_blocksize, :, :].detach().contiguous()
        local_mask = startend_row_indices

    elif mode in ["balance", "balance_overlap"]:
        local_mask, buckets = balance_flashmask_input_inter_machine(
            startend_row_indices, cp_size, rank,
            balance_chunk_size=2048,
            buckets_per_machine=args.buckets_per_machine,
            epsilon=args.epsilon,
            max_swap_iterations=args.max_swap_iterations,
            use_locality_swap=args.use_locality_swap,
            max_locality_iterations=args.max_locality_iterations)
        print(buckets,local_mask)
        q_local = scatter_balance(query, mode="balanced_swap", axis = 1,
                                  buckets=buckets, group=cp_group).detach().contiguous()
        k_local = scatter_balance(key, mode="balanced_swap", axis = 1,
                                  buckets=buckets, group=cp_group).detach().contiguous()
        v_local = scatter_balance(value, mode="balanced_swap", axis = 1,
                                  buckets=buckets, group=cp_group).detach().contiguous()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Enable gradients
    q_local.stop_gradient = False
    k_local.stop_gradient = False
    v_local.stop_gradient = False

    return q_local, k_local, v_local, local_mask


def make_attn_fn(mode, q_local, k_local, v_local, local_mask, use_rs=False):
    """Create zero-arg attention function based on mode."""
    if mode == "baseline":
        return partial(flashmask_attention_cp, q_local, k_local, v_local,
                       local_mask, causal=False, mode="allgather_kv")
    elif mode == "overlap":
        return partial(overlap_flashmask_attention, q_local, k_local, v_local,
                       local_mask, causal=False, mode="overlap", use_rs=use_rs)
    elif mode == "balance":
        return partial(flashmask_attention_cp, q_local, k_local, v_local,
                       local_mask, causal=False, mode="balance_q")
    elif mode == "balance_overlap":
        return partial(overlap_flashmask_attention, q_local, k_local, v_local,
                       local_mask, causal=False, mode="balance_q", use_rs=use_rs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------
def test_cp_flashmask(generate_mask_fn, B=1, S=8192, H=1, D=128, dtype='bf16'):
    """Run benchmark for a given mask type with the selected mode."""
    paddle.seed(2024)

    total_q = S
    total_k = S
    batch_size = B
    num_head = H
    num_head_q = 8 * H
    head_size = D
    rank = cp_group.rank
    mode = args.mode

    # Generate mask (same for all modes)
    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        startend_row_indices, causal = generate_mask_fn(batch_size, total_q, num_head, head_size)

    # Create data on rank 0, broadcast to all
    if rank == 0:
        query = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    else:
        query = paddle.empty([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)

    paddle.distributed.broadcast(query, src=cp_group.ranks[0], group=cp_group)
    paddle.distributed.broadcast(key, src=cp_group.ranks[0], group=cp_group)
    paddle.distributed.broadcast(value, src=cp_group.ranks[0], group=cp_group)
    paddle.device.synchronize()
    paddle.distributed.barrier(group=cp_group)

    # Prepare local inputs and attention function
    q_local, k_local, v_local, local_mask = prepare_inputs(
        query, key, value, startend_row_indices,
        cp_size, rank, cp_group, mode, args.use_ipo)

    attn_fn = make_attn_fn(mode, q_local, k_local, v_local, local_mask, args.use_rs)

    print(f"Rank: {rank}, group members: {cp_group.ranks}, q shape: {q_local.shape}, kv shape: {k_local.shape}, use_rs: {args.use_rs}")
    # Benchmark
    fwd_time, bwd_time = do_bench_flashmaskcp(
        attn_fn, cp_group,
        warmup=WARM_UP, rep=BENCH_TIME, profile=args.profile, rank=rank)

    total_time = fwd_time + bwd_time

    # Compute TFLOPs/s
    sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, num_head, S)
    density = 1.0 - sparsity
    fwd_tflops = cal_tflops(density * cal_flops(B, num_head_q, S, S, D, mode='fwd') / cp_size, fwd_time)
    bwd_tflops = cal_tflops(density * cal_flops(B, num_head_q, S, S, D, mode='bwd') / cp_size, bwd_time)
    total_tflops = cal_tflops(density * cal_flops(B, num_head_q, S, S, D, mode='fwd_bwd') / cp_size, total_time)

    return fwd_time, bwd_time, total_time, fwd_tflops, bwd_tflops, total_tflops, density


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------
def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"sequence_length must be >= {num_answers + 1}")
    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    return [base + (1 if i < extra else 0) for i in range(num_answers + 1)]


def generate_none_mask(B, S, H, D, causal=True):
    return None, causal


def generate_ones_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(shape=(B, H, S, 2), dtype="int32")
    startend_row_indices[:, :, :, 0] = S
    return startend_row_indices, False


def generate_causal_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(shape=(B, H, S, 1), dtype="int32")
    startend_row_indices[:, :, :, 0] = S
    return startend_row_indices, True


def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)
    return startend_row_indices, True


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


def generate_upper_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213], padding_size=256):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - total_seq_len

    up_right_row_indices = []
    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    up_right_row_indices = paddle.to_tensor(
        up_right_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    down_left_row_indices = paddle.ones_like(up_right_row_indices) * (S - padding_size)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
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


def generate_share_question_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - total_seq_len

    startend_row_indices = [S] * doc_seq_lens[0]
    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])
    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)

    startend_row_indices = paddle.to_tensor(
        startend_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    return startend_row_indices, True


def generate_global_sliding_window_mask(B, S, H, D, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start[:global_token] = S
    down_left_start = down_left_start.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_start = paddle.full([S], global_token, dtype="int32")
    up_right_start[:global_token + right_window_size + 1] = 0
    up_right_start = up_right_start.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_end = paddle.arange(-right_window_size, S - right_window_size, dtype="int32")
    up_right_end[:global_token + right_window_size + 1] = 0
    up_right_end = up_right_end.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat(
        [down_left_start, down_left_end, up_right_start, up_right_end], axis=-1)
    return startend_row_indices, False


def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - total_seq_len

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(
        start_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + \
                      [seq_cusums[-1]] * doc_seq_lens[-1] + \
                      [S] * padding
    end_row_indices = paddle.to_tensor(
        end_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)
    return startend_row_indices, True


def generate_prefix_lm_document_mask(B, S, H, D,
                                     doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    """doc_seq_lens: list of (prefix_length, seq_length) tuples."""
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


def generate_prefix_lm_causal_mask(B, S, H, D, prefix_length=1024):
    assert prefix_length <= S
    down_left = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right = paddle.to_tensor(
        [0] * prefix_length + list(range(prefix_length, S)),
        dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left, up_right], axis=-1)
    return startend_row_indices, False


def generate_qk_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
    """maskout_pair: list of (offset, maskout_len) tuples."""
    start_row_indices = []
    end_row_indices = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset > last_offset
        start_row_indices.extend([S] * (offset - last_offset))
        end_row_indices.extend([S] * (offset - last_offset))
        start_row_indices.extend(list(range(offset, offset + maskout_len)))
        end_row_indices.extend([offset + maskout_len] * maskout_len)
        last_offset = offset + maskout_len

    assert last_offset <= S
    start_row_indices.extend([S] * (S - last_offset))
    end_row_indices.extend([S] * (S - last_offset))

    start_row_indices = paddle.to_tensor(
        start_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(
        end_row_indices, dtype=paddle.int32
    ).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)
    return startend_row_indices, True


def generate_random_eviction_mask(B, S, H, D, start_row=4096):
    np.random.seed(0)
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S + 1] * S)
            mask_pos = np.random.choice(S - 1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate(
                [mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            min_index = np.arange(1, S + 1)
            start_rows = np.maximum(start_rows, min_index)
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(
        start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    return startend_row_indices, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    rank = paddle.distributed.get_rank()

    print(f"[Rank {rank}] FlashMask CP Benchmark (Inter-Machine Comm Balance) — mode={args.mode}, "
          f"profile={args.profile}, use_rs={args.use_rs}, "
          f"buckets_per_machine={args.buckets_per_machine}, epsilon={args.epsilon}, "
          f"max_swap_iterations={args.max_swap_iterations}, "
          f"use_locality_swap={args.use_locality_swap}, "
          f"max_locality_iterations={args.max_locality_iterations}")

    # Read input file
    total_length = 0
    doc_seq_lens_list = []
    with open(input_file, 'r') as f:
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
                if args.cp_size > 0:
                    B = 1
                    if FIXED_LOCAL_LENGTH * args.cp_size != S:
                        print(f"Skipping {S}, since the specified CP size is: {args.cp_size}")
                        continue
                else:
                    B = 1 if S == 131072 else 2
                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{D}_{idx}_{args.dtype}")

                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]

                available_examples = {
                    "Causal Document Mask": lambda: test_cp_flashmask(
                        generate_mask_fn=partial(
                            generate_causal_document_mask, doc_seq_lens=doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype),
                    "Document Mask": lambda: test_cp_flashmask(
                        generate_mask_fn=partial(
                            generate_document_mask, doc_seq_lens=doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype),
                    "Prefix LM Document Mask": lambda: test_cp_flashmask(
                        generate_mask_fn=partial(
                            generate_prefix_lm_document_mask,
                            doc_seq_lens=prefix_doc_seq_lens),
                        B=B, S=S, H=H, D=D, dtype=args.dtype),
                }

                if "all" in args.examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = args.examples

                results = []
                for ex in ex_to_run:
                    if ex in available_examples:
                        print(ex)
                        fw_time, bw_time, total_time, fw_tflops, bw_tflops, total_tflops, density = available_examples[ex]()
                        results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}",
                                        f"{total_time:.4f}", f"{fw_tflops:.4f}",
                                        f"{bw_tflops:.4f}", f"{total_tflops:.4f}",
                                        f"{density:.4f}"])
                    else:
                        print(f"Warning: Unknown example '{ex}'. Skipping.")

                headers = ["Operation", "FW Time (ms)", "BW Time (ms)", "TOTAL Time (ms)",
                           "FW TFLOPs/s", "BW TFLOPs/s", "TOTAL TFLOPs/s", "Density"]
                print(tabulate(results, headers=headers, tablefmt="grid"))

                # Save results
                content = tabulate(results, headers=headers, tablefmt="tsv")
                out_dir = f"{args.dtype}_dist_test_locswap_{args.use_locality_swap}_eps_{args.epsilon}_{FIXED_LOCAL_LENGTH}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(
                    out_dir, f"{output_prefix}_{B}_{S}_{H}_{D}_{idx}_{rank}.csv")
                with open(out_path, "w") as f:
                    f.write(content)


if __name__ == "__main__":
    main()
