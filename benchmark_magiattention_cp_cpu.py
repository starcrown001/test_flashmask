import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random
import time
import gc
import argparse
import os
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.distributed as dist
from pydantic import TypeAdapter
from torch.distributed.device_mesh import init_device_mesh
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from datetime import timedelta

from sparsity_utils import ranges_block_sparsity

from tabulate import tabulate
import magi_attention
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.mask import AttnMask
from magi_attention.common.range import AttnRange
from magi_attention.functional import flex_flash_attn_func as ffa_func
from magi_attention.api.functools import infer_attn_mask_from_sliding_window
from magi_attention.meta import make_global_bucket_from_qk_ranges
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.api import (
    calc_attn,
    compute_pad_size,
    magi_attn_flex_dispatch,
    undispatch,
)
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    MinHeapDispatchAlg,
)
from magi_attention.api import calc_attn, compute_pad_size, dispatch, magi_attn_flex_key
from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg


torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

DISPATCH_ALG = MinHeapDispatchAlg()
CHUNK_SIZE = 2048
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 16))
CP_SIZE = WORLD_SIZE  # may differ from WORLD_SIZE when MP is enabled
ITERATION = 40
WARMUP = 5

BENCH_MODE: Any = None
ATTN_CONFIG: Any = None
BENCH_CONFIG: Any = None
DATA_CONFIG: Any = None
SAMPLE_CONFIG: Any = None
SEED: Any = None

def flush_cache(fast_flush=True):
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = torch.empty([int(cache_size // 4)], dtype=torch.int32, device="cuda")
    else:
        cache = torch.empty([int(cache_size)], dtype=torch.int8, device="cuda")
    cache.zero_()  # optional, if you需要
    del cache
    torch.cuda.synchronize()
    
def init_dist_environment(
    world_size: int
):
    rank = int(os.environ.get("RANK", 0))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if magi_attention.comm.is_hierarchical_comm_enable():
        cp_group = None
    else:
        cp_group = dist.new_group(list(range(world_size)), backend="nccl")

    return cp_group

def init_hierarchical_mesh(world_size: int,use_mp: bool):
    if magi_attention.comm.is_hierarchical_comm_enable() and world_size in (
        4,
        8,
        16,
        32,
        64,
    ):
        world_size_inter_node, world_size_intra_node = {
            4: (2, 2),
            8: (2, 4),
            16: (2, 8),
            32: (4, 8),
            64: (8, 8),
        }[world_size]
        if(use_mp):
            assert world_size == 16
            global_mesh = init_device_mesh(
                    device_type='cuda',
                    mesh_shape=(2, 2, 4),
                    mesh_dim_names=("inter", "intra", "exp_group")
                )
            device_mesh = global_mesh["inter", "intra"]

            cp_group = device_mesh._flatten().get_group()
            cp_ranks = dist.get_process_group_ranks(cp_group)
            # print(f"[Rank {dist.get_rank()}] Mode: Normal Seq (CP2 Strided), CP group ranks: {cp_ranks}")
            return device_mesh

        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )
    else:
        device_mesh = None

    return device_mesh

def do_bench_cpu(
    fn,
    warmup=5,
    rep=40,
    grad_to_none=None,
    fast_flush=True,
    return_mode="mean",
):
    """Benchmark the provided function with CPU timing (time.perf_counter).

    Similar interface to do_bench but uses CPU wall-clock timing with
    CUDA synchronize + dist.barrier between iterations, and all_reduce MAX
    across ranks.

    Args:
        fn (Callable): Function to benchmark
        warmup (int): Number of warmup iterations
        rep (int): Number of repeat iterations
        grad_to_none (list[torch.Tensor], optional): Reset gradients of these tensors to None
        fast_flush (bool): Whether to use faster kernel to flush L2 between measurements
        return_mode (str): Statistics mode, one of ["min", "max", "mean", "median"]

    Returns:
        float: Time in milliseconds (statistics according to return_mode)
    """
    assert return_mode in ["min", "max", "mean", "median"]

    rank = int(os.environ.get("RANK", 0))

    # L2 cache flush buffer
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    warm_buf = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    del warm_buf
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        if grad_to_none is not None:
            for x in grad_to_none:
                if x.grad is not None:
                    x.grad = None
        fn()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    # Benchmark
    times = []
    for i in range(rep):
        if grad_to_none is not None:
            for x in grad_to_none:
                if x.grad is not None:
                    x.grad = None

        # Flush L2 cache
        cache.zero_()

        # Synchronize before timing
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Time with CPU wall-clock
        torch.cuda.nvtx.range_push(f"bench_iter{i}")
        t0 = time.perf_counter()
        fn()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.nvtx.range_pop()

        times.append(1000 * (t1 - t0))

    # Final synchronization
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # Reduce-max across ranks
    times_tensor = torch.tensor(times, dtype=torch.float32, device="cuda")
    # print(f'cpu rank{rank}:{times_tensor}')
    # print("-" * 50)
    if dist.is_initialized():
        dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)
    times_tensor = times_tensor.to(device="cpu")
    if(rank == 0):
        print(f'cpu rank{rank}:{times_tensor}')
    print("-" * 50)
    # torch.cuda.empty_cache()
    # gc.collect()

    return getattr(torch, return_mode)(times_tensor).item()


def do_bench_gpu(
    fn,
    warmup=5,
    rep=40,
    grad_to_none=None,
    fast_flush=True,
    return_mode="mean",
):
    """Benchmark the provided function with GPU timing (CUDA events).

    Uses torch.cuda.Event for precise GPU-side timing, with L2 cache flush,
    dist.barrier synchronization, and all_reduce MAX across ranks.

    Args:
        fn (Callable): Function to benchmark
        warmup (int): Number of warmup iterations
        rep (int): Number of repeat iterations
        grad_to_none (list[torch.Tensor], optional): Reset gradients of these tensors to None
        fast_flush (bool): Whether to use faster kernel to flush L2 between measurements
        return_mode (str): Statistics mode, one of ["min", "max", "mean", "median"]

    Returns:
        float: Time in milliseconds (statistics according to return_mode)
    """
    assert return_mode in ["min", "max", "mean", "median"]

    # L2 cache flush buffer
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Warmup
    for _ in range(warmup):
        if grad_to_none is not None:
            for x in grad_to_none:
                if x.grad is not None:
                    x.grad = None
        fn()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    # Pre-allocate CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    # Benchmark
    for i in range(rep):
        if grad_to_none is not None:
            for x in grad_to_none:
                if x.grad is not None:
                    x.grad = None

        # Flush L2 cache
        cache.zero_()

        # Synchronize before timing
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Time with CUDA events
        torch.cuda.nvtx.range_push(f"bench_iter{i}")
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.cuda.nvtx.range_pop()

    # Wait for all events to complete
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # Compute GPU elapsed times
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]

    # Reduce-max across ranks
    rank = int(os.environ.get("RANK", 0))
    times_tensor = torch.tensor(times, dtype=torch.float32, device="cuda")
    # print(f'gpu rank{rank}:{times_tensor}')
    # print("-" * 50)
    if dist.is_initialized():
        dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)
    times_tensor = times_tensor.to(device="cpu")
    # print(f'gpu rank{rank}:{times_tensor}')
    # print("-" * 50)
    # torch.cuda.empty_cache()
    # gc.collect()

    return getattr(torch, return_mode)(times_tensor).item()


def run_magi_attn(
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    cp_size: int,
    chunk_size: int,
    attn_mask_type: list[AttnMaskType],
    cp_group,
    cp_mesh,
    iteration: int,
):
    """Run MagiAttention distributed attention benchmark with CPU timing.

    Uses time.perf_counter() for timing instead of do_bench.
    """

    rank = int(os.environ.get("RANK", 0))
    device = torch.cuda.current_device()
    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)

    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    # -----   init dist attn config ----- #

    pad_size = compute_pad_size(
        total_seqlen_q=total_seqlen,
        cp_size=cp_size,
        chunk_size=chunk_size,
    )
    num_sms = int(getattr(ATTN_CONFIG, "num_sms", 24))
    nvl_chunk_size = int(getattr(ATTN_CONFIG, "nvl_chunk_size", 8))
    nvl_buffer_size = int(getattr(ATTN_CONFIG, "nvl_buffer_size", 256))
    rdma_chunk_size = int(getattr(ATTN_CONFIG, "rdma_chunk_size", 16))
    rdma_buffer_size = int(getattr(ATTN_CONFIG, "rdma_buffer_size", 128))
    num_nvl_bytes = int(getattr(ATTN_CONFIG, "num_nvl_bytes", int(3e9)))
    num_rdma_bytes = int(getattr(ATTN_CONFIG, "num_rdma_bytes", int(1e9)))

    if cp_size <= 8:  # single node
        num_rdma_bytes = 0
        min_num_nvl_bytes = GrpCollConfig.get_min_num_bytes_intranode(
            num_sms=num_sms,
            num_ranks=cp_size,
            hidden_size=q_heads * hidden_size,
            nvl_buffer_size=nvl_buffer_size,
            dtype=torch.float32,
            transfer_lse=True,
            num_heads=q_heads,
            num_groups=3,
        )
        min_num_rdma_bytes = 0
    else:  # multi node
        assert cp_size % 8 == 0, (
            "cp_size must be a multiple of 8 for internode native grpcoll."
        )
        assert num_rdma_bytes > 0, (
            "num_rdma_bytes must be positive for internode native grpcoll."
        )
        (min_num_rdma_bytes, min_num_nvl_bytes) = (
            GrpCollConfig.get_min_num_bytes_internode(
                num_sms=num_sms,
                num_rdma_ranks=cp_size // 8,
                num_nvl_ranks=8,
                hidden_size=q_heads * hidden_size,
                rdma_buffer_size=rdma_buffer_size,
                nvl_buffer_size=nvl_buffer_size,
                dtype=torch.float32,
                transfer_lse=True,
                num_heads=q_heads,
                num_groups=3,
            )
        )

    assert num_nvl_bytes >= min_num_nvl_bytes, (
        f"{num_nvl_bytes=} ({num_nvl_bytes / 1024**3:.2f} GB) "
        "is insufficient for native grpcoll, "
        f"since {min_num_nvl_bytes=} ({min_num_nvl_bytes / 1024**3:.2f} GB)."
    )
    assert num_rdma_bytes >= min_num_rdma_bytes, (
        f"{num_rdma_bytes=} ({num_rdma_bytes / 1024**3:.2f} GB) "
        "is insufficient for native grpcoll, "
        f"since {min_num_rdma_bytes=} ({min_num_rdma_bytes / 1024**3:.2f} GB)."
    )

    grpcoll_config = GrpCollConfig(
        num_sms=num_sms,
        nvl_chunk_size=nvl_chunk_size,
        nvl_buffer_size=nvl_buffer_size,
        rdma_chunk_size=rdma_chunk_size,
        rdma_buffer_size=rdma_buffer_size,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
    )
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=ATTN_CONFIG.dispatch_alg()),  # type: ignore[arg-type]
        overlap_config=OverlapConfig(
            enable=ATTN_CONFIG.enable_overlap,
            mode=ATTN_CONFIG.overlap_mode,
            degree=ATTN_CONFIG.degree,
            min_chunk_size=ATTN_CONFIG.min_chunk_size,
            max_num_chunks=ATTN_CONFIG.max_num_chunks,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
        grpcoll_config=grpcoll_config,
    )

    # -----    dispatch   ---- #

    magi_attn_runtime_key = magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        num_heads_q=q_heads,
        num_heads_kv=kv_heads,
        head_dim=hidden_size,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_mesh
        if magi_attention.comm.is_hierarchical_comm_enable()
        else cp_group,
        dist_attn_config=dist_attn_config,
    )
    x_local = dispatch(x, key=magi_attn_runtime_key)

    # -----   projection  ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

    q_local.requires_grad_(True)
    k_local.requires_grad_(True)
    v_local.requires_grad_(True)
    
    if(rank == 0) :
        print('q_local_shape:', q_local.shape)
        print(f'q_local_shape:{q_local.shape}, k_local_shape:{k_local.shape}, v_local_shape:{v_local.shape}, dout_local_shape:{dout_local.shape}')

    # -----    forward benchmark (CPU + GPU timing)   ---- #
    def fwd_fn():
        return calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    fwd_time_cpu = do_bench_cpu(fwd_fn, warmup=5, rep=iteration)
    # fwd_time_gpu = do_bench_gpu(fwd_fn, warmup=5, rep=iteration)

    # -----    backward benchmark (CPU + GPU timing)   ---- #
    out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    def bwd_fn():
        out_local.backward(dout_local, retain_graph=True)

    bwd_time_cpu = do_bench_cpu(bwd_fn, grad_to_none=[q_local, k_local, v_local], warmup=5, rep=iteration)
    # bwd_time_gpu = do_bench_gpu(bwd_fn, grad_to_none=[q_local, k_local, v_local], warmup=5, rep=iteration)

    # # -----    Print CPU vs GPU timing comparison   ---- #
    # if rank == 0:
    #     print(f"  fwd: CPU={fwd_time_cpu:.3f}ms  GPU={fwd_time_gpu:.3f}ms  "
    #           f"overhead={fwd_time_cpu - fwd_time_gpu:.3f}ms ({(fwd_time_cpu / fwd_time_gpu - 1) * 100:.1f}%)")
    #     print(f"  bwd: CPU={bwd_time_cpu:.3f}ms  GPU={bwd_time_gpu:.3f}ms  "
    #           f"overhead={bwd_time_cpu - bwd_time_gpu:.3f}ms ({(bwd_time_cpu / bwd_time_gpu - 1) * 100:.1f}%)")

    # Use CPU timing as the primary result (includes sync/barrier overhead)
    fwd_time_ms = fwd_time_cpu
    bwd_time_ms = bwd_time_cpu

    # ----- undispatch ----- #
    _ = undispatch(out_local, magi_attn_runtime_key)
    return fwd_time_ms, bwd_time_ms

def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

def calculate_sparsity(q_ranges, k_ranges, attn_mask_type, seq_len_q, seq_len_k):
    total_possible = seq_len_q * seq_len_k
    active_positions = 0
    attn_area = make_global_bucket_from_qk_ranges(
        q_ranges,
        k_ranges,
        attn_mask_type,
        num_chunks=1,
        chunk_size=seq_len_q,
    ).area
                
    return 1 - (attn_area / total_possible)

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")
    
def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    """transfer seqlens list to cu_seqlens, do not have check"""
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens

def copy_mask_for_batchs(q_ranges, k_ranges, is_causal_mapping, seqlen_qkv, bs):
    q_ranges_multi = q_ranges.copy()
    k_ranges_multi = k_ranges.copy()
    is_causal_mapping_multi = is_causal_mapping.copy()
    for i in range(1,bs):
        q_ranges_i = [[x + seqlen_qkv,y + seqlen_qkv] for [x, y] in q_ranges]
        k_ranges_i = [[x + seqlen_qkv,y + seqlen_qkv] for [x, y] in k_ranges]
        is_causal_mapping_i = is_causal_mapping
        q_ranges_multi.extend(q_ranges_i)
        k_ranges_multi.extend(k_ranges_i)
        is_causal_mapping_multi.extend(is_causal_mapping_i)
    return q_ranges_multi, k_ranges_multi, is_causal_mapping_multi


def test_mask(
    mask_mod: Optional[tuple[list[int], list[int], bool]] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    dtype = 'bf16',
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
    disable_fwd_atomic_reduction: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_mesh = None,
):
    if dtype == 'bf16':
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    GQA_fac = 8    
    q_ranges ,k_ranges, attn_mask_type = mask_mod

    # Compute block sparsity from original (pre-batch-copy) ranges
    sparsity = ranges_block_sparsity(q_ranges, k_ranges, attn_mask_type, S, S)
    density = 1.0 - sparsity

    q_ranges ,k_ranges, attn_mask_type = copy_mask_for_batchs(q_ranges, k_ranges, attn_mask_type, S, B)
    q_ranges_tensor = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges_tensor = torch.tensor(k_ranges, device=device, dtype=torch.int32)
    attn_mask_type_tensor = torch.tensor(attn_mask_type, device=device, dtype=torch.int32)
    
    magi_attention_call = lambda: ffa_func(q, k, v, q_ranges_tensor, k_ranges_tensor, attn_mask_type_tensor, disable_fwd_atomic_reduction = disable_fwd_atomic_reduction)

    q_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=q_ranges)
    k_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=k_ranges)
    attn_mask_type_: list[AttnMaskType] = [
        AttnMaskType.FULL if mask_type == 0 else
        AttnMaskType.CAUSAL if mask_type == 1 else
        AttnMaskType.INVCAUSAL if mask_type == 2 else
        AttnMaskType.BICAUSAL if mask_type == 3 else
        AttnMaskType.FULL  
        for mask_type in attn_mask_type
    ]
    fwd_time_ms ,bwd_time_ms = run_magi_attn(
        total_seqlen=B * S,
        embed_dim= H * D,
        q_heads=H * GQA_fac,
        kv_heads=H,
        hidden_size=D,
        dtype=data_type,
        q_ranges=q_ranges_,
        k_ranges=k_ranges_,
        cp_size=CP_SIZE,
        chunk_size=CHUNK_SIZE,
        attn_mask_type=attn_mask_type_,
        cp_group=cp_group,
        cp_mesh = cp_mesh,
        iteration=ITERATION,
    )
    
    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd') * GQA_fac / CP_SIZE
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd') * GQA_fac / CP_SIZE
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd') * GQA_fac / CP_SIZE

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity


    
def generate_prefix_lm_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """generate PREFIX LM DOCUMENT mask (prefix lm varlen)"""
        seqlens = [x[1] for x in doc_seq_lens]
        full_seqlens = [x[0] for x in doc_seq_lens]
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_mask_type: list[int] = []
        for i in range(len(seqlens)):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            full_seqlen = full_seqlens[i]
            if full_seqlen < seqlens[i]:
                q_ranges.append([start, start + full_seqlen])
                k_ranges.append([start, start + full_seqlen])
                attn_mask_type.append(0)

                q_ranges.append([start + full_seqlen, end])
                k_ranges.append([start, end])
                attn_mask_type.append(1)
            else:
                q_ranges.append([start, end])
                k_ranges.append([start, end])
                attn_mask_type.append(0)

        return (q_ranges, k_ranges, attn_mask_type)

def generate_causal_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate CAUSAL DOCUMENT mask (varlen causal)"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

    attn_mask_type = [1] * len(seqlens)

    return (ranges, ranges, attn_mask_type)

def generate_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate FULL DOCUMENT maks (varlen full)"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

    attn_mask_type = [0] * len(seqlens)

    return (ranges, ranges, attn_mask_type)

def generate_global_sliding_window_mask(global_token = 16, window_size = 4096, total_seqlen = 8192)-> tuple[list[list[int]], list[list[int]], list[int]]:
        """generate GLOBAL SLIDING WINDOW mask"""
        if window_size + 1 >= total_seqlen:
            return generate_full_mask(total_seqlen=total_seqlen)[:3]
        window_size_single = window_size

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_type_map: list[int] = []

        q_ranges.append([0, total_seqlen])
        k_ranges.append([0, window_size_single])
        attn_type_map.append(0)

        q_ranges.append([0, window_size_single])
        k_ranges.append([window_size_single, total_seqlen])
        attn_type_map.append(0)

        (
            sw_q_ranges,
            sw_k_ranges,
            sw_attn_mask_type,
        ) = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(start=window_size_single, end=total_seqlen),
            k_range=AttnRange(start=window_size_single, end=total_seqlen),
            window_size=[window_size_single, window_size_single],
        )

        sw_attn_type_map = [
            {
                AttnMaskType.FULL: 0,
                AttnMaskType.CAUSAL: 1,
                AttnMaskType.INVCAUSAL: 2,
                AttnMaskType.BICAUSAL: 3,
            }[mask_type]
            for mask_type in sw_attn_mask_type
        ]

        q_ranges.extend(sw_q_ranges.to_naive_ranges())  # type: ignore
        k_ranges.extend(sw_k_ranges.to_naive_ranges())  # type: ignore
        attn_type_map.extend(sw_attn_type_map)

        return (q_ranges, k_ranges, attn_type_map)

def generate_sliding_window_mask( window_size = 4096, total_seqlen = 8192)-> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate SLIDING WINDOW FULL mask"""
    if window_size + 1 >= total_seqlen:
        return generate_full_mask(total_seqlen=total_seqlen)[:3]

    q_ranges, k_ranges, attn_mask_type = infer_attn_mask_from_sliding_window(
        q_range=AttnRange(start=0, end=total_seqlen),
        k_range=AttnRange(start=0, end=total_seqlen),
        window_size=[window_size, 0],
    )
    attn_type_map = [
        {
            AttnMaskType.FULL: 0,
            AttnMaskType.CAUSAL: 1,
            AttnMaskType.INVCAUSAL: 2,
            AttnMaskType.BICAUSAL: 3,
        }[mask_type]
        for mask_type in attn_mask_type
    ]

    return (
        q_ranges.to_naive_ranges(),  # type: ignore
        k_ranges.to_naive_ranges(),
        attn_type_map,
    )
    
def generate_causal_mask(total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate CAUSAL mask"""
    ranges = [[0, total_seqlen]]
    attn_mask_type = [1]

    return (ranges, ranges, attn_mask_type)

def generate_full_mask(total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate FULL mask"""
    ranges = [[0, total_seqlen]]
    attn_mask_type = [0]

    return (ranges, ranges, attn_mask_type)

def _process_sequence_block(seqlens: list[int], cu_seqlens: list[int], cu_seqlens_offset: int, 
                           q_ranges: list[list[int]], k_ranges: list[list[int]], is_causal_mapping: list[bool]) -> None:
    """处理单个文档序列块，生成对应的注意力掩码范围"""
    total_seqlen = sum(seqlens)
    for j in range(len(seqlens)):
        if j == 1:
            q_ranges[-1] = [cu_seqlens[cu_seqlens_offset] , cu_seqlens[cu_seqlens_offset + j + 1]]
            k_ranges[-1] = [cu_seqlens[cu_seqlens_offset], cu_seqlens[cu_seqlens_offset + j + 1]]

            q_ranges.append([cu_seqlens[cu_seqlens_offset + j + 1], cu_seqlens[cu_seqlens_offset] +total_seqlen])
            k_ranges.append([cu_seqlens[cu_seqlens_offset], cu_seqlens[cu_seqlens_offset + j]])
            is_causal_mapping.append(False)
        else:
            q_ranges.append([cu_seqlens[cu_seqlens_offset + j] , cu_seqlens[cu_seqlens_offset + j + 1]])
            k_ranges.append([cu_seqlens[cu_seqlens_offset + j], cu_seqlens[cu_seqlens_offset + j + 1]])
            is_causal_mapping.append(True)

def _flatten_seqlens_and_compute_cu_seqlens(doc_seq_lens: list[list[int]]) -> tuple[list[int], list[int]]:
    """扁平化文档序列长度并计算累计序列长度"""
    seqlens_flatten = [num for sublist in doc_seq_lens for num in sublist]
    cu_seqlens = seqlens2cu_seqlens(seqlens_flatten)
    return seqlens_flatten, cu_seqlens


def generate_share_question_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """生成共享问题注意力掩码"""
    seqlens_flatten, cu_seqlens = _flatten_seqlens_and_compute_cu_seqlens(doc_seq_lens)

    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    is_causal_mapping: list[bool] = []
    cu_seqlens_offset = 0
    
    for i in range(len(doc_seq_lens)):
        _process_sequence_block(doc_seq_lens[i], cu_seqlens, cu_seqlens_offset, q_ranges, k_ranges, is_causal_mapping)
        cu_seqlens_offset += len(doc_seq_lens[i])

    return (q_ranges, k_ranges, is_causal_mapping)

def generate_causal_blockwise_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate CAUSAL BLOCKWISE mask"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    total_seqlen = sum(seqlens)

    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    for i in range(len(seqlens)):
        q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    k_ranges[-1] = [0, total_seqlen]

    attn_mask_type = [1] * len(seqlens)

    return (q_ranges, k_ranges, attn_mask_type)

def generate_prefix_lm_causal_mask(seqlen=3746, total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate PREFIX LM CAUSAL mask"""

    if seqlen < total_seqlen:
        q_ranges = [[0, total_seqlen], [seqlen, total_seqlen]]
        k_ranges = [[0, seqlen], [seqlen, total_seqlen]]
        attn_mask_type = [0, 1]
    else:
        q_ranges = [[0, total_seqlen]]
        k_ranges = [[0, total_seqlen]]
        attn_mask_type = [0]

    return (q_ranges, k_ranges, attn_mask_type)

def generate_qk_sparse_mask( maskout_pair=[(1024, 538), (2358, 1700)],total_seqlen=8192) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """generate QK SPARSE mask"""

    offsets = [x[0] for x in maskout_pair]
    mask_offset_seqlens = [x[1] for x in maskout_pair]

    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    attn_mask_type: list[int] = []
    last_offset = 0

    for i in range(len(offsets)):
        offset = offsets[i]
        mask_offset = mask_offset_seqlens[i]
        assert offset >= last_offset
        if mask_offset != 0:
            q_ranges.append([last_offset, offset])
            k_ranges.append([0, offset])
            attn_mask_type.append(1)

            q_ranges.append([offset,offset + mask_offset ])
            k_ranges.append([0, offset])
            attn_mask_type.append(0)

            last_offset =  offset + mask_offset
        else :
            assert False

    if(last_offset < total_seqlen):
        q_ranges.append([last_offset, total_seqlen])
        k_ranges.append([0, total_seqlen])
        attn_mask_type.append(1)

    return (q_ranges, k_ranges, attn_mask_type)

def generate_random_eviction_mask(start_row = 4096, total_seqlen = 8192):
    """generate random eviction mask"""
    q_ranges: list[list[int]] = [[0, start_row]]
    k_ranges: list[list[int]] = [[0, start_row]]
    attn_mask_type: list[int] = [1]
    S = total_seqlen

    start_rows = np.array([S+1] * S)
    mask_pos = np.random.choice(S-1, S - start_row, replace=False)
    index = np.arange(start_row, S)
    mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
    start_rows[mask_pos] = index
    causal_mask = np.arange(0, total_seqlen)
    start_rows = np.maximum(start_rows, causal_mask)

    q_ranges += [[start_row, int(start_rows[id])] for id in range(S)]
    k_ranges += [[id, id+1] for id in range(S)]
    attn_mask_type += [0 for _ in range(S)]

    return (q_ranges, k_ranges, attn_mask_type)

def split_sequence(sequence_length):
    if sequence_length < 3:
        raise ValueError("序列长度必须至少为 3，以保证能够分配给一个 Question 和两个 Answer。")
    
    # 确定 Answer 的数量
    num_answers = random.randint(2, 6)
    
    # 初始化分配的长度
    lengths = [1] * (num_answers + 1)  # 至少给每个部分分配一个长度，确保为正整数
    
    # 剩余的长度需要分配
    remaining_length = sequence_length - sum(lengths)
    
    # 随机分配剩余的长度
    for _ in range(remaining_length):
        # 随机选择一个位置增加长度
        index = random.randint(0, num_answers)
        lengths[index] += 1

    return lengths

def load_py_as_dict(config_path: str) -> dict[str, Any]:
    """Load a Python file as a module and extract uppercase-named variables."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_path = os.path.abspath(config_path)
    spec = spec_from_file_location("conf", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config file: {config_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_config = {
        k: v
        for k, v in module.__dict__.items()
        if not k.startswith("__") and k[0].isupper()
    }
    try:
        return TypeAdapter(dict[str, Any]).validate_python(raw_config)
    except Exception as e:
        raise ValueError(f"Failed to validate config: {str(e)}")
    
def load_bench_config(config_file):
    config_dict = load_py_as_dict(config_file)

    global BENCH_MODE, BENCH_CONFIG, ATTN_CONFIG, DATA_CONFIG, SAMPLE_CONFIG
    global SEED, TOTAL_SEQLENS, CP_SIZE
    BENCH_MODE = config_dict["BENCH_MODE"]
    BENCH_CONFIG = config_dict["BENCH_CONFIG"]
    ATTN_CONFIG = config_dict["ATTN_CONFIG"]
    DATA_CONFIG = config_dict["DATA_CONFIG"]
    SAMPLE_CONFIG = config_dict["SAMPLE_CONFIG"]
    SEED = config_dict["SEED"]

    # Build total seqlen list from per-rank seqlens * cp_size
    TOTAL_SEQLENS = [s * CP_SIZE for s in DATA_CONFIG.seqlens_per_rank]

    if BENCH_CONFIG.output_path is not None:
        os.makedirs(BENCH_CONFIG.output_path, exist_ok=True)
        
def main(examples: List[str] = ["all"], dtype='bf16',config = "none", fast_eval=False):
    """Run the benchmark with the given examples using CPU timing.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    total_length = 0
    doc_seq_lens_list = []
    rank = int(os.environ.get("RANK", 0))
    cp_group = init_dist_environment(
        world_size=WORLD_SIZE
    )
    use_mp = WORLD_SIZE != CP_SIZE
    cp_mesh  = init_hierarchical_mesh(WORLD_SIZE, use_mp = use_mp)
    input_file = 'kernel_test_dist_seq_info.txt'
    # if(use_mp == True and fast_eval):
    #     input_file = 'kernel_test_dist_seq_info-32k.txt'
    # elif fast_eval :
    #     input_file = 'kernel_test_dist_seq_info-128k.txt'
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
            
        for H in [1]:
            D = 128
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                B = 2 if use_mp else 1
                if(fast_eval):
                    if(S // CP_SIZE != 16384):
                        continue
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

                available_examples = {
                    "Causal Document Mask": lambda: test_mask(mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Document Mask": lambda: test_mask(mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Prefix LM Document Mask": lambda: test_mask(mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
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
                        fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = available_examples[ex]()
                        results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"])
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
                ]
                print(
                    tabulate(
                        results,
                        headers=headers,
                        tablefmt="grid",
                    )
                )
                
                content2=tabulate(results, headers=headers, tablefmt="tsv")
                os.makedirs(f"{dtype}_dist_test_cpu", exist_ok=True)
                text_file = open(f"{dtype}_dist_test_cpu/magiattention_{rank}_{CP_SIZE}_{B}_{S}_{H}_{D}_{idx}.csv","w")
                text_file.write(content2)
                text_file.close()


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    parser = ArgumentParser(description="Run specific examples or all examples with CPU timing.")
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
        "--config",
        type=str,
        default="magi_benchmark_conf.py"
    )
    
    parser.add_argument(
        "--fast_eval",
        type=bool,
        default=False
    )

    args = parser.parse_args()
    load_bench_config(args.config)
    main(**vars(args))
