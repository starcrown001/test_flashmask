import os
import sys
import numpy as np
from typing import Optional, List, Dict
import random
import argparse
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from tabulate import tabulate

# Local TE baseline imports (copied from MagiAttention exps)
from te_baselines.interface import AttnImpl
from te_baselines.shard import (
    ParallelMode,
    get_ring_pg,
    get_ulysess_pg,
    init_distributed,
)
from te_baselines.utils_cp import AttnBackend

from magi_attention.benchmarking.bench import do_bench
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType

from sparsity_utils import ranges_block_sparsity

# Import baseline implementations
from te_baselines.ring_attn import RingAttnP2P
from te_baselines.ulysess import Ulysess

torch.set_default_device("cuda")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 8))
CP_SIZE = WORLD_SIZE
ITERATION = 40

# ---- Process Group Cache ---- #
CP_GROUP_CACHE: Dict[str, Any] = {}


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def init_cp_group(attn_impl: str, world_size: int):
    """Initialize CP process group for the given attn_impl."""
    cache_key = f"{attn_impl}_{world_size}"
    if cache_key in CP_GROUP_CACHE:
        return CP_GROUP_CACHE[cache_key]

    if attn_impl == "ring":
        pg_meta = {ParallelMode.RING: world_size}
        device_shard = init_distributed(world_size=world_size, pg_meta=pg_meta)
        cp_group = get_ring_pg(device_shard)
    elif attn_impl == "ulysses":
        pg_meta = {ParallelMode.ULYSESS: world_size}
        device_shard = init_distributed(world_size=world_size, pg_meta=pg_meta)
        cp_group = get_ulysess_pg(device_shard)
    else:
        raise ValueError(f"Unknown attn_impl: {attn_impl}")

    CP_GROUP_CACHE[cache_key] = cp_group
    return cp_group


def run_te_dist_attn(
    total_seqlen: int,
    embed_dim: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: AttnMaskType,
    world_size: int,
    attn_impl: str,
    cp_group,
    iteration: int,
):
    """Run TE distributed attention benchmark (ring or ulysses).

    Follows the same pattern as run_dist_attn in run_benchmark.py.
    """
    device = torch.cuda.current_device()
    rank = dist.get_rank()
    attn_backend = AttnBackend.TE

    # -----    init attn module   ---- #
    if attn_impl == "ring":
        attn = RingAttnP2P(
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == "ulysses":
        attn = Ulysess(
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [device]
    else:
        raise ValueError(f"Unknown attn_impl: {attn_impl}")

    # -----    init test data   ---- #
    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)
    q_proj = torch.nn.Linear(embed_dim, num_heads_q * head_dim, dtype=dtype, device=device)
    k_proj = torch.nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)
    v_proj = torch.nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)
    dout_proj = torch.nn.Linear(embed_dim, num_heads_q * head_dim, dtype=dtype, device=device)

    # -----    dispatch   ---- #
    # NOTE: dispatch only supports (t, h, d)
    x = x.view(total_seqlen, 1, embed_dim)
    x_local = attn.dispatch(x, q_ranges, total_seqlen, ["q", "dout"])
    _ = attn.dispatch(x, k_ranges, total_seqlen, ["k", "v"])

    # -----   qkv projection ----- #
    x_local_samples = x_local
    if isinstance(x_local_samples, torch.Tensor):
        x_local_samples = [x_local_samples]

    q_local_samples = []
    k_local_samples = []
    v_local_samples = []
    dout_local_samples = []
    for xl in x_local_samples:
        xl = xl.view(-1, embed_dim)
        ql = q_proj(xl).view(-1, num_heads_q, head_dim)
        kl = k_proj(xl).view(-1, num_heads_kv, head_dim)
        vl = v_proj(xl).view(-1, num_heads_kv, head_dim)
        dl = dout_proj(xl).view(-1, num_heads_q, head_dim)
        ql.requires_grad_(True)
        kl.requires_grad_(True)
        vl.requires_grad_(True)
        q_local_samples.append(ql)
        k_local_samples.append(kl)
        v_local_samples.append(vl)
        dout_local_samples.append(dl)

    q_local = q_local_samples[0]
    k_local = k_local_samples[0]
    v_local = v_local_samples[0]
    dout_local = dout_local_samples[0]

    # Ulysses GQA handling: repeat kv heads if needed
    if attn_impl == "ulysses":
        assert world_size % num_heads_kv == 0 or num_heads_kv % world_size == 0
        H = world_size // num_heads_kv
        if H > 1:
            k_local = torch.repeat_interleave(k_local, H, dim=1)
            v_local = torch.repeat_interleave(v_local, H, dim=1)

    # -----   pre_compute ---- #
    attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

    if rank == 0:
        print(f"  q_local_shape:{q_local.shape}, k_local_shape:{k_local.shape}, "
              f"v_local_shape:{v_local.shape}, dout_local_shape:{dout_local.shape}")

    # -----    forward benchmark   ---- #
    def fwd_fn():
        return attn.apply_attn(
            q_local, k_local, v_local,
            attn_mask_type, 0.0, None, False,
        )

    fwd_perf = do_bench(fwd_fn, return_flops=True, return_mem=False, warmup=5, rep=iteration)
    fwd_time_ms = fwd_perf["flops"]

    # -----    backward benchmark   ---- #
    out, _ = attn.apply_attn(
        q_local, k_local, v_local,
        attn_mask_type, 0.0, None, False,
    )

    def bwd_fn():
        out.backward(dout_local, retain_graph=True)

    bwd_perf = do_bench(bwd_fn, grad_to_none=[q_local, k_local, v_local],
                        return_flops=True, return_mem=False, warmup=5, rep=iteration)
    bwd_time_ms = bwd_perf["flops"]

    return fwd_time_ms, bwd_time_ms


# ---- Mask generation helpers ---- #

def generate_causal_document_mask_ranges(doc_seq_lens: List[int]) -> tuple:
    """Generate ranges for CAUSAL DOCUMENT mask (varlen causal)"""
    cu_seqlens = seqlens2cu_seqlens(doc_seq_lens)
    ranges = []
    for i in range(len(doc_seq_lens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    attn_mask_type = [1] * len(doc_seq_lens)  # 1 = CAUSAL
    return ranges, ranges, attn_mask_type


def generate_document_mask_ranges(doc_seq_lens: List[int]) -> tuple:
    """Generate ranges for FULL DOCUMENT mask (varlen full)"""
    cu_seqlens = seqlens2cu_seqlens(doc_seq_lens)
    ranges = []
    for i in range(len(doc_seq_lens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    attn_mask_type = [0] * len(doc_seq_lens)  # 0 = FULL
    return ranges, ranges, attn_mask_type


def calculate_sparsity_document_mask(doc_seqlens: List[int], is_causal: bool,
                                     Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128) -> float:
    """Calculate block-level sparsity for document mask using ranges_block_sparsity"""
    total_seqlen = sum(doc_seqlens)
    if is_causal:
        q_ranges, k_ranges, attn_mask_type = generate_causal_document_mask_ranges(doc_seqlens)
    else:
        q_ranges, k_ranges, attn_mask_type = generate_document_mask_ranges(doc_seqlens)

    return ranges_block_sparsity(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        seq_len_q=total_seqlen,
        seq_len_k=total_seqlen,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )


def copy_mask_for_batches(doc_seq_lens, seqlen_qkv, bs):
    """Copy document lengths for multiple batches"""
    result = []
    for i in range(bs):
        result.extend(doc_seq_lens)
    return result


# ---- Flops helpers ---- #

def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def cal_tflops(flops, time_ms):
    return flops * (1e3 / time_ms) / 1e12


# ---- Test entry points ---- #

def test_mask(
    doc_seq_lens: List[int],
    B: int = 1,
    H: int = 1,
    S: int = 8192,
    D: int = 128,
    dtype='bf16',
    attn_impl: str = 'ring',
    cp_group=None,
    is_causal: bool = True,
):
    """Test TE attention with document mask."""
    data_type = torch.bfloat16 if dtype == 'bf16' else torch.float16
    GQA_fac = 8

    # Sparsity (from single-batch doc lens)
    sparsity = calculate_sparsity_document_mask(doc_seq_lens, is_causal)
    density = 1.0 - sparsity

    # Multi-batch doc lens
    doc_seq_lens_batched = copy_mask_for_batches(doc_seq_lens, S, B)

    # Build AttnRanges & AttnMaskType for the batched case
    if is_causal:
        q_ranges_raw, k_ranges_raw, mask_type_raw = generate_causal_document_mask_ranges(doc_seq_lens_batched)
        magi_mask_type = AttnMaskType.CAUSAL
    else:
        q_ranges_raw, k_ranges_raw, mask_type_raw = generate_document_mask_ranges(doc_seq_lens_batched)
        magi_mask_type = AttnMaskType.FULL

    q_ranges = AttnRanges.from_ranges(ranges=q_ranges_raw)
    k_ranges = AttnRanges.from_ranges(ranges=k_ranges_raw)

    fwd_time_ms, bwd_time_ms = run_te_dist_attn(
        total_seqlen=B * S,
        embed_dim=H * D,
        num_heads_q=H * GQA_fac,
        num_heads_kv=H,
        head_dim=D,
        dtype=data_type,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=magi_mask_type,
        world_size=CP_SIZE,
        attn_impl=attn_impl,
        cp_group=cp_group,
        iteration=ITERATION,
    )

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd') * GQA_fac / CP_SIZE
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd') * GQA_fac / CP_SIZE
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd') * GQA_fac / CP_SIZE

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return (fwd_time_ms, bwd_time_ms, total_time_ms,
            fwd_flops, bwd_flops, total_flops,
            fwd_tflops, bwd_tflops, total_tflops, sparsity)


# ---- Main ---- #

def main(examples: List[str] = ["all"], dtype='bf16', attn_impl='ring'):
    """Run the TE CP attention benchmark.

    Args:
        examples: List of examples to run. "all" runs all.
        dtype: 'bf16' or 'fp16'
        attn_impl: 'ring', 'ulysses', or 'both'
    """
    rank = int(os.environ.get("RANK", 0))

    # Initialize CP group(s)
    impls_to_run = ["ring", "ulysses"] if attn_impl == "both" else [attn_impl]
    cp_groups = {}
    for impl in impls_to_run:
        cp_groups[impl] = init_cp_group(impl, WORLD_SIZE)

    input_file = 'kernel_test_dist_seq_info.txt'
    if not os.path.exists(input_file):
        doc_seq_lens_list = [
            (8192, [2048, 2048, 2048, 2048], [0, 0, 0, 0]),
            (16384, [4096, 4096, 4096, 4096], [0, 0, 0, 0]),
            (32768, [8192, 8192, 8192, 8192], [0, 0, 0, 0]),
        ]
    else:
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

    for H in [1,4,8]:
        D = 128
        for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
            B = 1

            if isinstance(prefix_doc_seq_lens[0], (list, tuple)):
                doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
            else:
                doc_seq_lens = prefix_doc_seq_lens

            for impl in impls_to_run:
                cp_group = cp_groups[impl]
                impl_label = "TE Ring" if impl == "ring" else "TE Ulysses"

                print(f"\n{B}_{S}_{H}_{D}_{idx}_{dtype}_{impl}")

                available_examples = {
                    f"Causal Document Mask ({impl_label})": lambda impl=impl: test_mask(
                        doc_seq_lens=doc_seq_lens, B=B, S=S, H=H, D=D,
                        dtype=dtype, attn_impl=impl, cp_group=cp_group, is_causal=True,
                    ),
                    f"Document Mask ({impl_label})": lambda impl=impl: test_mask(
                        doc_seq_lens=doc_seq_lens, B=B, S=S, H=H, D=D,
                        dtype=dtype, attn_impl=impl, cp_group=cp_group, is_causal=False,
                    ),
                }

                if "all" in examples:
                    ex_to_run = list(available_examples.keys())
                else:
                    ex_to_run = [e for e in examples if e in available_examples]

                results = []
                for ex in ex_to_run:
                    if ex in available_examples:
                        print(ex)
                        try:
                            (fw_time, bw_time, total_time, fw_flops, bw_flops,
                             total_flops, fw_tflops, bw_tflops, total_tflops,
                             sparsity) = available_examples[ex]()
                            results.append([
                                ex,
                                f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}",
                                f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}",
                                f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:.4f}",
                                f"{sparsity:.4f}",
                            ])
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            results.append([ex] + ["ERROR"] * 10)
                    else:
                        print(f"Warning: Unknown example key '{ex}'. Skipping.")

                headers = [
                    "Operation",
                    "FW Time (ms)", "BW Time (ms)", "TOTAL Time (ms)",
                    "FW FLOPs", "BW FLOPs", "TOTAL FLOPs",
                    "FW TFLOPs/s", "BW TFLOPs/s", "TOTAL TFLOPs/s",
                    "Sparsity",
                ]
                print(tabulate(results, headers=headers, tablefmt="grid"))

                content2 = tabulate(results, headers=headers, tablefmt="tsv")
                out_dir = f"{dtype}_te_{impl}_test"
                os.makedirs(out_dir, exist_ok=True)
                with open(f"{out_dir}/te_{impl}_{rank}_{CP_SIZE}_{B}_{S}_{H}_{D}_{idx}.csv", "w") as f:
                    f.write(content2)


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")

    parser = ArgumentParser(description="Run TE CP attention benchmark for document masks.")
    parser.add_argument(
        "--examples", type=str, nargs="+", default=["all"],
        help="List of examples to run. 'all' runs all.",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        help="Data type: 'bf16' or 'fp16'",
    )
    parser.add_argument(
        "--attn_impl", type=str, default="ring",
        choices=["ring", "ulysses", "both"],
        help="Attention implementation: 'ring', 'ulysses', or 'both'",
    )

    args = parser.parse_args()
    main(**vars(args))
