import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import paddle
import os
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flashmask_attention
from context_parallel_utils_new import scatter_balance, all_gather_balance
from flash_mask.cp_balance import balance_flashmask_input, get_q_workload, assign_tasks_heap
from overlap_utils import overlap_flashmask_attention
from sparsity_utils import flashmask_block_sparsity

from jsonargparse import ArgumentParser
import paddle.distributed.fleet as fleet
import time

import numpy as np

def get_args():
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: causal, alibi, sliding_window, prefix_lm, "
        "document, softcap, softcap_approx, or 'all' to run all examples.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16"
    )
    parser.add_argument(
        "--profile",
        action="store_true"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1
    )
    parser.add_argument(
        "--use_rs",
        action="store_true"
    )

    return parser.parse_args()

args = get_args()
output_prefix = "flashmask_overlap"
if args.batch == 0:
    input_file = 'kernel_test_dist_seq_info.txt'
    cp_size = (int)(os.getenv("CP_SIZE", "8"))
    mp_size = 1
    sd_size = cp_size
else:
    input_file = 'kernel_test_dist_seq_info.txt'
    cp_size = 8
    mp_size = 2
    sd_size = 8

if cp_size == 32:
    input_file = 'kernel_test_dist_seq_info_cp32.txt'

if args.profile:
    BENCH_TIME = 20
    WARM_UP = 5
else:
    BENCH_TIME = 1500
    WARM_UP = 100

cp_use_ipo = False
strategy = fleet.DistributedStrategy()

strategy.hybrid_configs = {
  "dp_degree": 1,
  "mp_degree": mp_size,
  "pp_degree": 1,
  "sharding_degree": sd_size,
  "sep_degree": 1,
  "ep_degree":  cp_size * mp_size,
  "moe_sharding_degree": 1,
  "cp_degree": cp_size,
  "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"]
}

fleet.init(is_collective=True, strategy=strategy)
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"序列长度必须至少为 {num_answers + 1}")

    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    # 前extra个部分多加1
    lengths = [base + (1 if i < extra else 0) for i in range(num_answers + 1)]

    return lengths

def do_bench_dist(fn, cp_group, warmup=1, rep=300, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of the provided function for distributed operations (balance/scatter/gather).
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    paddle.base.core.nvprof_nvtx_push("paddle")

    fn()
    paddle.device.synchronize()
    paddle.distributed.barrier(group=cp_group)

    n_warmup = 3
    n_repeat = 5
    # Warm-up
    for _ in range(n_warmup):
        fn()
        paddle.device.synchronize()
        paddle.distributed.barrier(group=cp_group)
    # Benchmark
    dist_time = []
    for i in range(n_repeat):
        paddle.device.synchronize()
        paddle.distributed.barrier(group=cp_group)
        time0 = time.perf_counter()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        fn()
        paddle.device.synchronize()
        paddle.distributed.barrier(group=cp_group)
        time1 = time.perf_counter()
        dist_time.append((time1 - time0) * 1000)
    paddle.base.core.nvprof_nvtx_pop()
    return sum(dist_time) / n_repeat

def do_bench_flashmaskcp(q_local, k_local, v_local, o_grad_local, startend_row_indices, group, is_causal, mode="balance", warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of overlap_flashmask_attention using CUDA event-based timing.

    :param warmup: Number of warmup iterations
    :type warmup: int
    :param rep: Number of repetition iterations
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: paddle.Tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median"]

    rank = paddle.distributed.get_rank()
    print(f"overlap debug {q_local.shape=}, {k_local.shape=}, {v_local.shape=}, {startend_row_indices.shape=}")

    # Initial run to ensure correctness and warm up CUDA context
    out_local = overlap_flashmask_attention(q_local, k_local, v_local, startend_row_indices, causal=is_causal, mode=mode, use_rs=args.use_rs)
    out_local.backward(o_grad_local)
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    if fast_flush:
        cache = paddle.empty([int(256e6 // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(256e6)], dtype=paddle.int8)

    n_warmup = warmup
    n_repeat = rep

    # Create CUDA events for fwd and bwd timing per iteration
    fwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    fwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_start_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_end_event = [paddle.device.Event(enable_timing=True) for _ in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        out_local = overlap_flashmask_attention(q_local, k_local, v_local, startend_row_indices, causal=is_causal, mode=mode, use_rs=args.use_rs)
        out_local.backward(o_grad_local)
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()

    # Use all_reduce as barrier before benchmark
    paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)

    # Benchmark
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # Clear L2 cache
        cache.zero_()

        # Barrier before each iteration
        paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)

        # Forward timing
        fwd_start_event[i].record()
        out_local = overlap_flashmask_attention(q_local, k_local, v_local, startend_row_indices, causal=is_causal, mode=mode, use_rs=args.use_rs)
        fwd_end_event[i].record()

        # Backward timing
        bwd_start_event[i].record()
        out_local.backward(o_grad_local)
        bwd_end_event[i].record()

    # Final barrier
    paddle.distributed.all_reduce(cache, op=paddle.distributed.ReduceOp.SUM, group=cp_group)

    # Synchronize and compute times from CUDA events
    paddle.device.synchronize()
    fwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(fwd_start_event, fwd_end_event)],
        dtype=paddle.float32,
    )
    bwd_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(bwd_start_event, bwd_end_event)],
        dtype=paddle.float32,
    )

    # Synchronize times across ranks (take max across all ranks)
    paddle.distributed.all_reduce(fwd_times, op=paddle.distributed.ReduceOp.MAX, group=cp_group)
    paddle.distributed.all_reduce(bwd_times, op=paddle.distributed.ReduceOp.MAX, group=cp_group)

    fwd_times = fwd_times.cpu()
    bwd_times = bwd_times.cpu()

    if quantiles is not None:
        fwd_ret = paddle.quantile(fwd_times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        bwd_ret = paddle.quantile(bwd_times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(fwd_ret) == 1:
            fwd_ret = fwd_ret[0]
        if len(bwd_ret) == 1:
            bwd_ret = bwd_ret[0]
        return fwd_ret, bwd_ret

    fwd_stat = getattr(paddle, return_mode)(fwd_times).item()
    bwd_stat = getattr(paddle, return_mode)(bwd_times).item()
    return fwd_stat, bwd_stat
    
def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

def cp_flashmask_balance_bench(query, key, value, startend_row_indices, is_causal, o_grad, mode):
    B, S, H, D = query.shape
    group = cp_group
    rank = group.rank
    local_qs = []
    local_ks = []
    local_vs = []
    local_ograds = []
    balance_q_chunksize = 2048
    workload = get_q_workload(startend_row_indices, balance_q_chunksize, 128, 128)

    total_workload = paddle.sum(workload, axis=1)
    if cp_use_ipo:
        buckets, bucket_weights, cuts = assign_tasks_ipo(workload.reshape(-1, 2), cp_size)
    else:
        buckets, bucket_weights, cuts = assign_tasks_heap(workload.reshape(-1, 2), cp_size)
    hcg = fleet.get_hybrid_communicate_group()
    # print(buckets)
    for (_, idx) in buckets[rank]:
        local_qs.append(query[:, idx * balance_q_chunksize:(idx + 1) * balance_q_chunksize, :, :])
        local_ks.append(key[:, idx * balance_q_chunksize:(idx + 1) * balance_q_chunksize, :, :])
        local_vs.append(value[:, idx * balance_q_chunksize:(idx + 1) * balance_q_chunksize, :, :])
        local_ograds.append(o_grad[:, idx * balance_q_chunksize:(idx + 1) * balance_q_chunksize, :, :])
    local_q = paddle.concat(local_qs, axis=1).detach().contiguous()
    local_k = paddle.concat(local_ks, axis=1).detach().contiguous()
    local_v = paddle.concat(local_vs, axis=1).detach().contiguous()
    local_o_grad = paddle.concat(local_ograds, axis=1)
    local_startend_row_indices, buckets = balance_flashmask_input(startend_row_indices, cp_size, rank)
    local_q = scatter_balance(query, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets).detach().contiguous()
    print("pass0")
    balancex = lambda: balance_flashmask_input(startend_row_indices, cp_size, rank)
    balance_time = do_bench_dist(balancex, cp_group=cp_group)
    print("pass1")

    local_k.stop_gradient = False
    local_v.stop_gradient = False
    local_q.stop_gradient = False
    x = query.detach().reshape(B, S, -1).contiguous()

    local_startend_row_indices, buckets = balance_flashmask_input(startend_row_indices, cp_size, rank, balance_chunk_size=balance_q_chunksize)

    scatter_x = lambda: scatter_balance(x, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets)
    scatter_x_time = do_bench_dist(scatter_x, cp_group=cp_group)
    local_x = scatter_balance(x, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets)

    gather_x = lambda: all_gather_balance(local_x, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets)
    gather_x_time = do_bench_dist(gather_x, cp_group=cp_group)

    cp_fwd_time, cp_bwd_time = do_bench_flashmaskcp(local_q, local_k, local_v, local_o_grad, local_startend_row_indices, group, is_causal, mode)
    return balance_time, scatter_x_time, gather_x_time, cp_fwd_time, cp_bwd_time

def test_cp_famask(
    generate_mask_fn,
    B: int = 16,
    S: int = 8192,
    H: int = 16,
    D: int = 64,
    dtype = 'bf16',
):
    """
    测试上下文并行FlashMask注意力机制的性能基准 (overlap版本)
    
    该函数用于测试在分布式并行环境中使用overlap策略的FlashMask注意力机制的
    前向传播和后向传播性能，支持不同类型的注意力掩码生成策略。

    Args:
        generate_mask_fn: 注意力掩码生成函数，用于生成startend_row_indices和因果关系标记
        B: 批次大小，默认16
        S: 序列长度，默认8192
        H: 注意力头数，默认16
        D: 每个注意力头的维度，默认64
        dtype: 数据类型，默认'bf16'

    Returns:
        tuple: 包含前向传播时间、后向传播时间、FLOPS、TFLOPS、稀疏度以及balance/scatter/gather开销
    """
    paddle.seed(2024)
    total_q = S
    total_k = S
    batch_size = B
    num_head = H
    num_head_q = 8 * H
    head_size = D
    rank = cp_group.rank

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        print("enter", generate_mask_fn)
        startend_row_indices, causal = generate_mask_fn(batch_size, total_q, num_head, head_size)

    if rank == 0:
        query = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        o_grad = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
    else:
        query = paddle.empty([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        o_grad = paddle.empty([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)

    print(f"wsm debug {query.shape=}, {key.shape=}, {value.shape=}, {startend_row_indices.shape=}")

    # 广播到所有 rank
    paddle.distributed.broadcast(query, src=cp_group.ranks[0], group=cp_group)
    paddle.distributed.broadcast(key, src=cp_group.ranks[0], group=cp_group)
    paddle.distributed.broadcast(value, src=cp_group.ranks[0], group=cp_group)
    paddle.distributed.broadcast(o_grad, src=cp_group.ranks[0], group=cp_group)
    paddle.device.synchronize()
    paddle.distributed.barrier(group=cp_group)
    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False
    causal = False

    balance_time, scatter_x_time, gather_x_time, fwd_time, bwd_time = cp_flashmask_balance_bench(query, key, value, startend_row_indices, causal, o_grad, "balance_q")
    paddle.device.synchronize()

    total_time = fwd_time + bwd_time

    sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, H, S)
    density = 1.0 - sparsity

    fwd_flops = density * cal_flops(B, num_head_q, S, S, D, mode='fwd') / cp_size
    bwd_flops = density * cal_flops(B, num_head_q, S, S, D, mode='bwd') / cp_size
    total_flops = density * cal_flops(B, num_head_q, S, S, D, mode='fwd_bwd') / cp_size

    fwd_tflops = cal_tflops(fwd_flops, fwd_time)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time)
    total_tflops = cal_tflops(total_flops, total_time)

    return fwd_time, bwd_time, total_time, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity, balance_time, scatter_x_time, gather_x_time

def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_ones_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(
        shape=(B, 1, S, 2), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = False
    return startend_row_indices, causal

def generate_causal_mask(B,S,H,D):
    startend_row_indices = paddle.zeros(
        shape=(B, 1, S, 1), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = True
    return startend_row_indices, causal

def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)

    causal=True
    return startend_row_indices, causal

def generate_causal_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S, f"{total_seq_len=}, {S=}"
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    lts = np.repeat(seq_cusums, doc_seq_lens)
    lts = paddle.to_tensor(lts, dtype=paddle.int32).reshape((1, 1, S, 1))
    ute = paddle.arange(S, dtype='int32').reshape((1, 1, S, 1))
    startend_row_indices = paddle.concat([lts, ute], axis=-1)
    startend_row_indices = startend_row_indices.repeat_interleave(B, 0)
    
    causal = False
    return startend_row_indices, causal

def generate_upper_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213],padding_size = 256):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    up_right_row_indices = []

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    down_left_row_indices =  paddle.ones_like(up_right_row_indices) * (S - padding_size)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

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
        
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_global_sliding_window_mask(B, S, H, D, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start_row_indices[:global_token] = S
    down_left_start_row_indices =  down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_start_row_indices = paddle.full([S], global_token, dtype="int32")
    up_right_start_row_indices[:global_token+right_window_size+1] = 0
    up_right_start_row_indices = up_right_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_end_row_indices = paddle.arange(
        -right_window_size, S - right_window_size, dtype="int32"
    )
    up_right_end_row_indices[:global_token+right_window_size+1] = 0
    up_right_end_row_indices = up_right_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_start_row_indices, down_left_end_row_indices, up_right_start_row_indices, up_right_end_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seq_lens[-1] + [S] * padding
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_prefix_lm_document_mask(B, S, H, D, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    """
    tuple(prefix_length, seq_length)
    """
    assert len(doc_seq_lens) >= 2
    total_seq_len = 0
    for prefix_length, seq_length in doc_seq_lens:
        total_seq_len += seq_length
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_causal_mask(B, S, H, D, prefix_length=1024):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S
    down_left_row_indices = paddle.full([S], S, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, S)), dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
    """
    tuple(offset, maskout_len)
    """
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset > last_offset
        start_row_indices.extend([S]*(offset-last_offset))
        end_row_indices.extend([S]*(offset-last_offset))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend([S]*(S-last_offset))
    end_row_indices.extend([S]*(S-last_offset))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_random_eviction_mask(B, S, H, D, start_row=4096):
    np.random.seed(0)
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            min_index = np.arange(1,S+1)
            start_rows = np.maximum(start_rows, min_index)
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    causal = True
    return startend_row_indices, causal

def main(examples: List[str] = ["all"], dtype='bf16', profile=False, batch=1, num_heads=8, use_rs=False):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    total_length = 0
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    doc_seq_lens_list = []
    rank = paddle.distributed.get_rank()
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
            
        for D in [ 128]:
            H = 4
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                if(S // cp_size < 4096):
                    print(f"Skipped {S}")
                    continue
                B = 1

                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                maskout_pair = []
                offset = 0
                print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")
                if sum(qksparse_mask) == 0:
                    maskout_pair = [(1024, 538), (2358, 1700)]
                else:
                    for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                        if is_maskout:
                            maskout_pair.append((offset, doc_seq))
                        offset += doc_seq

                share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]
                print(share_qa_docs)

                available_examples = {
                    # "Full": lambda: test_cp_famask(generate_mask_fn=partial(generate_ones_mask), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Causal": lambda: test_cp_famask(generate_mask_fn=partial(generate_causal_mask), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Sliding Window": lambda: test_cp_famask(generate_mask_fn=partial(generate_sliding_window_mask, window_size=int(S*0.0625)), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal Document Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Document Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Share Question Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_share_question_mask, doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Global Sliding Window": lambda: test_cp_famask(generate_mask_fn=partial(generate_global_sliding_window_mask, global_token=16, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Causal Blockwise Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_causal_blockwise_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Prefix LM Document Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Prefix LM Causal Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, prefix_length=int(S*0.5)), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "QK-sparse Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_qk_sparse_mask, maskout_pair=maskout_pair), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Random Eviction Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_random_eviction_mask, start_row=S//2), B=B, S=S, H=H, D=D, dtype=dtype),
                }

                global total_num
                total_num = len(available_examples)

                if "all" in examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = examples

                results = []
                for ex in ex_to_run:
                    if ex in available_examples:
                        print(ex)
                        fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity, balance_time, scatter_x_time, gather_x_time = available_examples[ex]()
                        results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}", f"{balance_time:.4f}", f"{scatter_x_time:.4f}", f"{gather_x_time:.4f}"])
                    else:
                        print(f"Warning: Unknown example key '{ex}'. Skipping.")

                # Usage in your results formatting:
                headers = [
                    "Operation",
                    "FW Time (ms)",
                    "BW Time (ms)",
                    "TOTAL Time (ms)",
                    "FW FLOPs",
                    "BW FLOPs",
                    "TOTAL FLOPs",
                    "FW TFLOPs/s",
                    "BW TFLOPs/s",
                    "TOTAL TFLOPs/s",
                    "Sparsity",
                    "Balance Time (ms)",
                    "Scatter Time (ms)",
                    "Gather Time (ms)",
                ]
                print(
                    tabulate(
                        results,
                        headers=headers,
                        tablefmt="grid",
                    )
                )
                
                content2=tabulate(results, headers=headers, tablefmt="tsv")
                os.makedirs(f"{dtype}_dist_test", exist_ok=True)
                text_file = open(f"{dtype}_dist_test/{output_prefix}_{rank}_{cp_size}_{B}_{S}_{H}_{D}_{idx}.csv","w")
                text_file.write(content2)
                text_file.close()

if __name__ == "__main__":
    print("New FlashMask Balance + Overlap!")
    main(**vars(args))
