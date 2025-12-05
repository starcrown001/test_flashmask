import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random
import time
import gc

import torch
import torch.nn.functional as F

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from datetime import timedelta

from triton.testing import do_bench

from tabulate import tabulate
import magi_attention
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
from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg


from triton.testing import do_bench

torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

DISPATCH_ALG = MinHeapDispatchAlg()
CHUNK_SIZE = 512
WORLD_SIZE = 4
CP_size = 4
ITERATION = 10

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

    # cp_groups = [
    #     [0,4,8,12],
    #     [1,5,9,13],
    #     [2,6,10,14],
    #     [3,7,11,15]
    # ]
    # for group in cp_groups:
    #     if rank in group:
    #         return group
        
    return cp_group

def init_hierarchical_mesh(world_size: int):
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
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )
    else:
        device_mesh = None

    return device_mesh

def run_magi_attn(
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    world_size: int,
    chunk_size: int,
    attn_mask_type: list[AttnMaskType],
    cp_group,
    cp_mesh,
    iteration: int,
):

    rank = int(os.environ.get("RANK", 0))
    device = torch.cuda.current_device()

    # -----    init test data   ---- #

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

    x.requires_grad_(True)

    # -----   init dispatch mata ----- #

    pad_size = compute_pad_size(
        total_seqlen_q=total_seqlen,
        cp_size=CP_size,
        chunk_size=chunk_size,
    )

    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=DISPATCH_ALG),
        overlap_config=OverlapConfig(
            enable=True,
            mode=AttnOverlapMode.STATIC,
            degree=2,
            min_chunk_size=512,
            max_num_chunks=64,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
    )

    # print(f"rank: {rank} | device: {device} pt1")

    # -----    dispatch   ---- #

    (
        x_local,
        magi_attn_runtime_key,
    ) = magi_attn_flex_dispatch(  # local_x with shape (total_seqlen_q + pad_size) / cp_size, h)
        x,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_mesh
        if magi_attention.comm.is_hierarchical_comm_enable()
        else cp_group,
        dist_attn_config=dist_attn_config,
    )
    
    # print(f"rank: {rank} | device: {device} pt2")

    # -----   projection  ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

    # -----    forward   ---- #
    
    fwd_time_ms = []
    bwd_time_ms = []
    
    flush_cache()

    for i in range(5):
        # if rank == 0 and i == 2:

        # if rank == 0 and i == 9:
        #     torch.cuda.profiler.stop()
        # print(f"rank: {rank} | device: {device} pt3")

        # -----    barrier at the beginning of each iteration   ---- #

        dist.barrier()
        torch.cuda.synchronize()

        out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)
        
        dist.barrier()
        torch.cuda.synchronize()
        out_local.backward(dout_local, retain_graph=True)
        dist.barrier()
        torch.cuda.synchronize()

    # torch.cuda.profiler.start()
    # torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()    
    for i in range(iteration):
        dist.barrier()
        torch.cuda.synchronize()
        time0 = time.perf_counter()

        torch.cuda.nvtx.range_push("MAGI_ATTENTION_FWD")
        out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        time1 = time.perf_counter()

        torch.cuda.nvtx.range_push("MAGI_ATTENTION_BWD")
        out_local.backward(dout_local, retain_graph=True)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        time2 = time.perf_counter()

        if(rank == 0):
            print(f"{i}: {1000 * (time1 - time0)}, {time2 - time1}")
        fwd_time_ms.append(1000 * (time1 - time0))
        bwd_time_ms.append(1000 * (time2 - time1))
    # torch.cuda.profiler.stop()
    # ----- undispatch ----- #
    out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    _ = undispatch(out_local, magi_attn_runtime_key)
    return sum(fwd_time_ms) / iteration, sum(bwd_time_ms) / iteration

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
    if(max(attn_mask_type) < 2):
        active_positions = sum((q_end - q_start) * (k_end - k_start) * (2 - attntype) * 0.5
                            for (q_start, q_end), (k_start, k_end), attntype in zip(q_ranges, k_ranges, attn_mask_type))
    else:
        for (q_start, q_end), (k_start, k_end), attntype in zip(q_ranges, k_ranges, attn_mask_type):
            if (attntype == 0):
                active_positions += (q_end - q_start) * (k_end - k_start)
            elif (attntype == 1 or attntype == 2):
                max_len = max(q_end - q_start, k_end - k_start)
                min_len = min(q_end - q_start, k_end - k_start)
                active_positions += (max_len - min_len) * min_len + min_len * min_len * 0.5
            elif (attntype == 3):
                active_positions += (q_end - q_start) * (k_end - k_start - (q_end - q_start))
                
    return 1 - (active_positions / total_possible)

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

    #assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    # if mask_mod is not None:
    #     block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    # else:
    #     block_mask = None

    q,k,v = [
        torch.randn(B * S, H , D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    gradOut = torch.randn(B * S, H, D, device=device, dtype=data_type)
    
    q_ranges ,k_ranges, attn_mask_type = mask_mod
    q_ranges_tensor = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges_tensor = torch.tensor(k_ranges, device=device, dtype=torch.int32)
    attn_mask_type_tensor = torch.tensor(attn_mask_type, device=device, dtype=torch.int32)
    # print(q.shape)
    # print(k.shape)
    # print(v.shape)
    
    magi_attention_call = lambda: ffa_func(q, k, v, q_ranges_tensor, k_ranges_tensor, attn_mask_type_tensor, disable_fwd_atomic_reduction = disable_fwd_atomic_reduction)

    results = []
    sparsity = calculate_sparsity(q_ranges, k_ranges, attn_mask_type, S, S)
    if mask_mod is not None:
        density = 1.0 - sparsity
    else:
        density = 1.0

    ç = 1.0 - density
    
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
        total_seqlen=S,
        embed_dim= H * D,
        q_heads=H,
        kv_heads=H,
        hidden_size=D,
        dtype=data_type,
        q_ranges=q_ranges_,
        k_ranges=k_ranges_,
        world_size=CP_size,
        chunk_size=CHUNK_SIZE,
        attn_mask_type=attn_mask_type_,
        cp_group=cp_group,
        cp_mesh = cp_mesh,
        iteration=ITERATION,
    )
    
    total_time_ms = fwd_time_ms + bwd_time_ms
        
    # np.save(f"tmp_res/q_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", q.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/k_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", k.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/v_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", v.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/gradOut_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", gradOut.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/magi_out_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", magi_out.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/q_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", q.grad.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/k_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", k.grad.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/v_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", v.grad.type(torch.float32).detach().cpu().numpy())

    return fwd_time_ms, bwd_time_ms, total_time_ms
    
def generate_prefix_lm_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate prefix lm document mask"""
        seqlens = [x[1] for x in doc_seq_lens]
        full_seqlens = [x[0] for x in doc_seq_lens]
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[bool] = []
        for i in range(len(seqlens)):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            full_seqlen = full_seqlens[i]
            if full_seqlen < seqlens[i]:
                q_ranges.append([start, start + full_seqlen])
                k_ranges.append([start, start + full_seqlen])
                is_causal_mapping.append(False)

                q_ranges.append([start + full_seqlen, end])
                k_ranges.append([start, end])
                is_causal_mapping.append(True)
            else:
                q_ranges.append([start, end])
                k_ranges.append([start, end])
                is_causal_mapping.append(False)

        return (q_ranges, k_ranges, is_causal_mapping)

def generate_causal_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate document full maks (varlen full mask)"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)

    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

    is_causal_mapping = [True] * len(seqlens)

    return (ranges, ranges, is_causal_mapping)

def generate_document_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate document full maks (varlen full mask)"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)

    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

    is_causal_mapping = [False] * len(seqlens)

    return (ranges, ranges, is_causal_mapping)

def generate_global_sliding_window_mask(global_token = 16, window_size = 4096, total_seqlen = 8192)-> tuple[list[list[int]], list[list[int]], list[int]]:
    q_ranges = [[0, global_token],[global_token, total_seqlen],[global_token, global_token + window_size],
                [global_token + window_size, total_seqlen - window_size],[total_seqlen - window_size,total_seqlen]]
    k_ranges = [[0,total_seqlen], [0,global_token],[global_token, global_token +  2 * window_size],
                [global_token, total_seqlen], [total_seqlen - 2 * window_size, total_seqlen]]
    attntype_map = [1,1,1,3,2]
    return (q_ranges, k_ranges, attntype_map)

def generate_sliding_window_mask( window_size = 4096, total_seqlen = 8192)-> tuple[list[list[int]], list[list[int]], list[int]]:
    q_ranges = [[0, window_size],[window_size, total_seqlen]]
    k_ranges = [[0,window_size],[0,total_seqlen]]
    attntype_map = [1,3]
    return (q_ranges, k_ranges, attntype_map)
    
def generate_causal_mask(total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate causal mask"""
    ranges = [[0, total_seqlen]]
    is_causal_mapping = [True]

    return (ranges, ranges, is_causal_mapping)

def generate_full_mask(total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate full mask"""
    ranges = [[0, total_seqlen]]
    is_causal_mapping = [False]

    return (ranges, ranges, is_causal_mapping)

def generate_share_question_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate share question mask"""
    seqlens = doc_seq_lens
    seqlens_flatten = [num for sublist in seqlens for num in sublist]
    cu_seqlens = seqlens2cu_seqlens(seqlens_flatten)

    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    is_causal_mapping: list[bool] = []
    cu_seqlens_offset = 0
    for i in range(len(seqlens)):
        total_seqlen = sum(seqlens[i])
        for j in range(len(seqlens[i])):
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
        cu_seqlens_offset += len(seqlens[i])

    return (q_ranges, k_ranges, is_causal_mapping)

def generate_causal_blockwise_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate causal blockwise mask"""
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    total_seqlen = sum(seqlens)

    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    for i in range(len(seqlens)):
        q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    k_ranges[-1] = [0, total_seqlen]

    is_causal_mapping = [True] * len(seqlens)

    return (q_ranges, k_ranges, is_causal_mapping)

def generate_prefix_lm_causal_mask(seqlen=3746, total_seqlen=7493) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate prefix lm causal mask"""
    
    if seqlen < total_seqlen:
        q_ranges = [[0, total_seqlen], [seqlen, total_seqlen]]
        k_ranges = [[0, seqlen], [seqlen, total_seqlen]]
        is_causal_mapping = [False, True]
    else:
        q_ranges = [[0, total_seqlen]]
        k_ranges = [[0, total_seqlen]]
        is_causal_mapping = [False]

    return (q_ranges, k_ranges, is_causal_mapping)

def generate_qk_sparse_mask( maskout_pair=[(1024, 538), (2358, 1700)],total_seqlen=8192) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """generate qk sparse mask"""
    
    offsets = [x[0] for x in maskout_pair]
    mask_offset_seqlens = [x[1] for x in maskout_pair]
    
    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    is_causal_mapping: list[bool] = []
    last_offset = 0
    
    for i in range(len(offsets)):
        offset = offsets[i]
        mask_offset = mask_offset_seqlens[i]
        assert offset >= last_offset
        if mask_offset != 0:
            q_ranges.append([last_offset, offset])
            k_ranges.append([0, offset])
            is_causal_mapping.append(True)
            
            last_offset =  offset + mask_offset    
        else :
            assert False

    if(last_offset < total_seqlen):
        q_ranges.append([last_offset, total_seqlen])
        k_ranges.append([0, total_seqlen])
        is_causal_mapping.append(True)
            
    return (q_ranges, k_ranges, is_causal_mapping)

def generate_random_eviction_mask(start_row = 4096, total_seqlen = 8192):
    """generate random eviction mask"""
    q_ranges: list[list[int]] = [[0, start_row]]
    k_ranges: list[list[int]] = [[0, start_row]]
    is_causal_mapping: list[bool] = [True]
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
    is_causal_mapping += [False for _ in range(S)]
    # print(is_causal_mapping)
    
    return (q_ranges, k_ranges, is_causal_mapping)

def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"序列长度必须至少为 {num_answers + 1}")

    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    # 前extra个部分多加1
    lengths = [base + (1 if i < extra else 0) for i in range(num_answers + 1)]

    return lengths

def main(examples: List[str] = ["all"], dtype='bf16'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    total_length = 0
    doc_seq_lens_list = []
    rank = int(os.environ.get("RANK", 0))
    cp_group = init_dist_environment(
        world_size=WORLD_SIZE
    )
    cp_mesh  = init_hierarchical_mesh(WORLD_SIZE)
    with open('kernel_test_dist_seq_info.txt', 'r') as f:
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
            
        #doc_seq_lens_list = doc_seq_lens_list[::-1]
        for D in [128]:
            H = 4096 // D
            # print(doc_seq_lens_list)
            for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
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
                    # "Full": lambda: test_mask(mask_mod=generate_full_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Causal": lambda: test_mask(mask_mod=generate_causal_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Sliding Window": lambda: test_mask(mask_mod=generate_sliding_window_mask(window_size=int(S*0.0625),total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Causal Document Mask": lambda: test_mask(mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Document Mask": lambda: test_mask(mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Share Question Mask": lambda: test_mask(mask_mod=generate_share_question_mask(doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Global Sliding Window": lambda: test_mask(mask_mod=generate_global_sliding_window_mask(global_token=16, window_size=int(S*0.0625), total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Causal Blockwise Mask": lambda: test_mask(mask_mod=generate_causal_blockwise_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Prefix LM Document Mask": lambda: test_mask(mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Prefix LM Causal Mask": lambda: test_mask(mask_mod=generate_prefix_lm_causal_mask(seqlen=int(S*0.5),total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "QK-sparse Mask": lambda: test_mask(mask_mod=generate_qk_sparse_mask(maskout_pair=maskout_pair, total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    # "Random Eviction Mask": lambda: test_mask(mask_mod=generate_random_eviction_mask(start_row=S//2, total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True, cp_group = cp_group,cp_mesh = cp_mesh),
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
                        fw_time, bw_time, total_time = available_examples[ex]()
                        results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}"])
                        # return
                    else:
                        print(f"Warning: Unknown example key '{ex}'. Skipping.")

                # Usage in your results formatting:
                headers = [
                    "Operation",
                    "FW Time (ms)",
                    "BW Time (ms)",
                    "TOTAL Time (ms)",
                ]
                print(
                    tabulate(
                        results,
                        headers=headers,
                        tablefmt="grid",
                    )
                )
                content2=tabulate(results, headers=headers, tablefmt="tsv")
                os.makedirs(f"{dtype}", exist_ok=True)
                text_file = open(f"{dtype}_dist_test/magiattention_{rank}_{B}_{S}_{H}_{D}_{idx}.csv","w")
                text_file.write(content2)
                text_file.close()


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
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

    args = parser.parse_args()
    main(**vars(args))

