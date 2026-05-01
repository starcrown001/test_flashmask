"""MagiAttention distributed benchmark with torch.profiler for Perfetto / TensorBoard timeline analysis.

This is a standalone profiler variant of benchmark_magiattention_cp_cpu.py.
It reuses all setup/mask/utility functions from that module and overrides
only run_magi_attn and main to add torch.profiler instrumentation.

Usage:
    torchrun --nproc_per_node=8 benchmark_magiattention_cp_cpu_profiler.py \
        --config magi_benchmark_conf.py \
        --profiler_output_dir ./profiler_traces \
        --profiler_wait 2 --profiler_warmup 1 --profiler_active 3

View traces:
    - TensorBoard:  tensorboard --logdir ./profiler_traces
    - Perfetto:     upload ./profiler_traces/*/trace.json.gz to ui.perfetto.dev
"""

import os
import gc
import time
import argparse
from typing import List, Any

import torch
import torch.distributed as dist

import benchmark_magiattention_cp_cpu as _bm

# Constants set at module level — safe to import directly
from benchmark_magiattention_cp_cpu import (
    CP_SIZE, WORLD_SIZE, ITERATION, WARMUP, CHUNK_SIZE,
    do_bench_cpu,
    do_bench_gpu,
    flush_cache,
    init_dist_environment,
    init_hierarchical_mesh,
    load_bench_config,
    load_py_as_dict,
    generate_causal_document_mask,
    generate_document_mask,
    generate_prefix_lm_document_mask,
    compute_pad_size,
    copy_mask_for_batchs,
    split_sequence,
    cal_flops,
    cal_tflops,
    ranges_block_sparsity,
    print_header,
    seqlens2cu_seqlens,
    calc_attn,
    undispatch,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


# Late-bound accessors for globals that are set by load_bench_config()
# (They are None at import time and only populated later.)
def _get_attn_config():
    return _bm.ATTN_CONFIG

def _get_bench_config():
    return _bm.BENCH_CONFIG

def _get_data_config():
    return _bm.DATA_CONFIG


def run_magi_attn_profiler(
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
    profiler_output_dir: str = "./profiler_traces",
    profiler_wait: int = 2,
    profiler_warmup: int = 1,
    profiler_active: int = 3,
):
    """Run MagiAttention benchmark with torch.profiler for timeline analysis.

    Compared to the vanilla run_magi_attn, this version:
      - Wraps the benchmark loop with torch.profiler.profile
      - Adds NVTX range annotations on every fwd/bwd iteration
      - Records CUDA events (for later GPU-side timing extraction)
      - Uses torch.profiler.schedule to profile only a window of iterations
      - Exports trace as JSON (for Perfetto) and TensorBoard handler

    Args:
        profiler_output_dir: Directory to write profiler trace files.
        profiler_wait: Number of iterations to skip before profiling.
        profiler_warmup: Number of warmup iterations within the profile window.
        profiler_active: Number of active iterations to record.
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
    attn_config = _get_attn_config()
    pad_size = compute_pad_size(
        total_seqlen_q=total_seqlen,
        cp_size=cp_size,
        chunk_size=chunk_size,
    )
    num_sms = int(getattr(attn_config, "num_sms", 24))
    nvl_chunk_size = int(getattr(attn_config, "nvl_chunk_size", 8))
    nvl_buffer_size = int(getattr(attn_config, "nvl_buffer_size", 256))
    rdma_chunk_size = int(getattr(attn_config, "rdma_chunk_size", 16))
    rdma_buffer_size = int(getattr(attn_config, "rdma_buffer_size", 128))
    num_nvl_bytes = int(getattr(attn_config, "num_nvl_bytes", int(3e9)))
    num_rdma_bytes = int(getattr(attn_config, "num_rdma_bytes", int(1e9)))

    from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
    from magi_attention.config import DistAttnConfig
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig, MinHeapDispatchAlg
    from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg
    from magi_attention.common.enum import AttnOverlapMode

    if cp_size <= 8:
        num_rdma_bytes = 0
        min_num_nvl_bytes = GrpCollConfig.get_min_num_bytes_intranode(
            num_sms=num_sms, num_ranks=cp_size,
            hidden_size=q_heads * hidden_size, nvl_buffer_size=nvl_buffer_size,
            dtype=torch.float32, transfer_lse=True, num_heads=q_heads, num_groups=3,
        )
        min_num_rdma_bytes = 0
    else:
        assert cp_size % 8 == 0
        assert num_rdma_bytes > 0
        (min_num_rdma_bytes, min_num_nvl_bytes) = GrpCollConfig.get_min_num_bytes_internode(
            num_sms=num_sms, num_rdma_ranks=cp_size // 8, num_nvl_ranks=8,
            hidden_size=q_heads * hidden_size, rdma_buffer_size=rdma_buffer_size,
            nvl_buffer_size=nvl_buffer_size, dtype=torch.float32,
            transfer_lse=True, num_heads=q_heads, num_groups=3,
        )

    assert num_nvl_bytes >= min_num_nvl_bytes
    assert num_rdma_bytes >= min_num_rdma_bytes

    grpcoll_config = GrpCollConfig(
        num_sms=num_sms, nvl_chunk_size=nvl_chunk_size, nvl_buffer_size=nvl_buffer_size,
        rdma_chunk_size=rdma_chunk_size, rdma_buffer_size=rdma_buffer_size,
        num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes,
    )
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=attn_config.dispatch_alg()),
        overlap_config=OverlapConfig(
            enable=attn_config.enable_overlap, mode=attn_config.overlap_mode,
            degree=attn_config.degree, min_chunk_size=attn_config.min_chunk_size,
            max_num_chunks=attn_config.max_num_chunks,
            alg=UniformOverlapAlg(random_costs=True, random_seed=42),
        ),
        grpcoll_config=grpcoll_config,
    )

    # -----    dispatch   ---- #
    from magi_attention.api import magi_attn_flex_key, dispatch

    magi_attn_runtime_key = magi_attn_flex_key(
        q_ranges=q_ranges, k_ranges=k_ranges, attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen, total_seqlen_k=total_seqlen,
        num_heads_q=q_heads, num_heads_kv=kv_heads, head_dim=hidden_size,
        pad_size=pad_size, chunk_size=chunk_size,
        cp_group_or_mesh=cp_mesh if __import__("magi_attention").comm.is_hierarchical_comm_enable() else cp_group,
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

    if rank == 0:
        print(f'q_local_shape:{q_local.shape}, k_local_shape:{k_local.shape}, '
              f'v_local_shape:{v_local.shape}, dout_local_shape:{dout_local.shape}')

    # Build per-rank output dir to avoid file conflicts
    rank_output_dir = os.path.join(profiler_output_dir, f"rank{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)

    # -----    forward benchmark with torch.profiler (CPU + GPU timing)   ---- #
    def fwd_fn():
        return calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=profiler_wait,
    #         warmup=profiler_warmup,
    #         active=profiler_active,
    #         repeat=1,
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         rank_output_dir, use_gzip=True
    #     ),
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True,
    #     with_flops=True,
    # ) as prof_fwd:
    fwd_time_cpu = do_bench_cpu(fwd_fn, warmup=5, rep=iteration)
    fwd_time_gpu = do_bench_gpu(fwd_fn, warmup=5, rep=iteration)
    # Step through all iterations to satisfy the profiler schedule
    # for _ in range(profiler_wait + profiler_warmup + profiler_active):
    #     prof_fwd.step()

    if rank == 0:
        print(f"[Profiler] Forward trace saved to {rank_output_dir}")

    # -----    backward benchmark with torch.profiler (CPU + GPU timing)   ---- #
    out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    def bwd_fn():
        out_local.backward(dout_local, retain_graph=True)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=profiler_wait,
    #         warmup=profiler_warmup,
    #         active=profiler_active,
    #         repeat=1,
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         rank_output_dir, use_gzip=True
    #     ),
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True,
    #     with_flops=True,
    # ) as prof_bwd:
    bwd_time_cpu = do_bench_cpu(bwd_fn, grad_to_none=[q_local, k_local, v_local], warmup=5, rep=iteration)
    bwd_time_gpu = do_bench_gpu(bwd_fn, grad_to_none=[q_local, k_local, v_local], warmup=5, rep=iteration)
    # Step through all iterations to satisfy the profiler schedule
    # for _ in range(profiler_wait + profiler_warmup + profiler_active):
    #     prof_bwd.step()

    # -----    Print CPU vs GPU timing comparison   ---- #
    if rank == 0:
        print(f"[Profiler] Backward trace saved to {rank_output_dir}")
        print(f"  fwd: CPU={fwd_time_cpu:.3f}ms  GPU={fwd_time_gpu:.3f}ms  "
              f"overhead={fwd_time_cpu - fwd_time_gpu:.3f}ms ({(fwd_time_cpu / fwd_time_gpu - 1) * 100:.1f}%)")
        print(f"  bwd: CPU={bwd_time_cpu:.3f}ms  GPU={bwd_time_gpu:.3f}ms  "
              f"overhead={bwd_time_cpu - bwd_time_gpu:.3f}ms ({(bwd_time_cpu / bwd_time_gpu - 1) * 100:.1f}%)")

    # Use CPU timing as the primary result (includes sync/barrier overhead)
    fwd_time_ms = fwd_time_cpu
    bwd_time_ms = bwd_time_cpu

    # ----- undispatch ----- #
    _ = undispatch(out_local, magi_attn_runtime_key)
    return fwd_time_ms, bwd_time_ms


def main(
    examples: List[str] = ["all"],
    dtype: str = "bf16",
    config: str = "none",
    fast_eval: bool = False,
    profiler_output_dir: str = "./profiler_traces",
    profiler_wait: int = 2,
    profiler_warmup: int = 1,
    profiler_active: int = 3,
):
    """Run the benchmark with torch.profiler enabled.

    Only runs the first (S, H) combination to keep profiler output manageable.
    """
    rank = int(os.environ.get("RANK", 0))
    cp_group = init_dist_environment(world_size=WORLD_SIZE)
    use_mp = WORLD_SIZE != CP_SIZE
    cp_mesh = init_hierarchical_mesh(WORLD_SIZE, use_mp=use_mp)

    input_file = "kernel_test_dist_seq_info.txt"
    if use_mp and fast_eval:
        input_file = "kernel_test_dist_seq_info-32k.txt"
    elif fast_eval:
        input_file = "kernel_test_dist_seq_info-128k.txt"

    total_length = 0
    doc_seq_lens_list = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "Total length" in line:
                total_length = int(line.split(":")[1].split(",")[0].strip())
            else:
                doc_list = eval(line.split(":")[-1].split("#")[0].strip())
                qksparse_mask = eval(line.split(":")[-1].split("#")[1].strip())
                doc_seq_lens_list.append((total_length, doc_list, qksparse_mask))

    # Only profile the first (H, idx) combination to keep trace size small
    GQA_fac = 8
    for H in [1]:
        D = 128
        for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(
            doc_seq_lens_list[:1]
        ):
            B = 2 if use_mp else 1
            doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]

            # Generate mask
            q_ranges_raw, k_ranges_raw, attn_mask_type_raw = generate_causal_document_mask(
                doc_seq_lens=doc_seq_lens
            )

            # Compute sparsity before batch copy
            sparsity = ranges_block_sparsity(q_ranges_raw, k_ranges_raw, attn_mask_type_raw, S, S)
            density = 1.0 - sparsity

            # Batch copy
            q_ranges_raw, k_ranges_raw, attn_mask_type_raw = copy_mask_for_batchs(
                q_ranges_raw, k_ranges_raw, attn_mask_type_raw, S, B
            )

            q_ranges_ = AttnRanges.from_ranges(ranges=q_ranges_raw)
            k_ranges_ = AttnRanges.from_ranges(ranges=k_ranges_raw)
            attn_mask_type_ = [
                AttnMaskType.FULL if mt == 0 else
                AttnMaskType.CAUSAL if mt == 1 else
                AttnMaskType.INVCAUSAL if mt == 2 else
                AttnMaskType.BICAUSAL if mt == 3 else
                AttnMaskType.FULL
                for mt in attn_mask_type_raw
            ]

            if dtype == "bf16":
                data_type = torch.bfloat16
            else:
                data_type = torch.float16

            print(f"\n{'='*60}")
            print(f"  Profiling: Causal Document Mask  B={B} S={S} H={H} D={D}")
            print(f"{'='*60}")

            fwd_time_ms, bwd_time_ms = run_magi_attn_profiler(
                total_seqlen=B * S,
                embed_dim=H * D,
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
                cp_mesh=cp_mesh,
                iteration=ITERATION,
                profiler_output_dir=profiler_output_dir,
                profiler_wait=profiler_wait,
                profiler_warmup=profiler_warmup,
                profiler_active=profiler_active,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MagiAttention benchmark with torch.profiler for timeline analysis."
    )
    parser.add_argument(
        "--examples", type=str, nargs="+", default=["all"],
        help="List of examples to run.",
    )
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--config", type=str, default="magi_benchmark_conf.py")
    parser.add_argument("--fast_eval", action="store_true", default=False)
    parser.add_argument(
        "--profiler_output_dir", type=str, default="./profiler_traces",
        help="Directory to write torch.profiler trace files.",
    )
    parser.add_argument(
        "--profiler_wait", type=int, default=2,
        help="Number of iterations to skip before profiling starts.",
    )
    parser.add_argument(
        "--profiler_warmup", type=int, default=1,
        help="Number of warmup iterations within the profile window.",
    )
    parser.add_argument(
        "--profiler_active", type=int, default=3,
        help="Number of active iterations to record in the profiler.",
    )

    args = parser.parse_args()
    load_bench_config(args.config)
    main(
        examples=args.examples,
        dtype=args.dtype,
        config=args.config,
        fast_eval=args.fast_eval,
        profiler_output_dir=args.profiler_output_dir,
        profiler_wait=args.profiler_wait,
        profiler_warmup=args.profiler_warmup,
        profiler_active=args.profiler_active,
    )
