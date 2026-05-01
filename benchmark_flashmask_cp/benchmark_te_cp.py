import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import os
import time
import numpy as np

import torch
import torch.distributed as dist
from transformer_engine.pytorch import DotProductAttention
import torch.cuda.nvtx
from dataclasses import dataclass
from jsonargparse import ArgumentParser

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

    return parser.parse_args()

args = get_args()
output_prefix = "megatron"

BENCH_TIME = 1200
WARM_UP = 50

@dataclass
class DistInfo:
    cp_size: int
    tp_size: int
    tp_comm_ranks: list
    cp_comm_ranks: list
    tp_comm_group: torch.distributed.ProcessGroup
    cp_comm_group: torch.distributed.ProcessGroup

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
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float32, device=times.device)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()

def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"序列长度必须至少为 {num_answers + 1}")

    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    # 前extra个部分多加1
    lengths = [base + (1 if i < extra else 0) for i in range(num_answers + 1)]

    return lengths

def do_bench_te_dpa_cp(q_local, k_local, v_local, o_grad_local, cu_seqlens, max_seqlen, is_causal, dist_info, 
    warmup=WARM_UP, rep=BENCH_TIME, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
    cp_comm_type="a2a", qkv_format="bshd"
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    rank = dist.get_rank()

    if qkv_format == "bshd":
        attn_mask_type = "causal" if is_causal else "no_mask"
    else:
        attn_mask_type = "padding_causal" if is_causal else "padding"

    print(f"wsm debug: {q_local.shape=}, {k_local.shape=}, {v_local.shape=}, {o_grad_local.shape=}")
    print(f"wsm debug {cu_seqlens=}")
    print(f"wsm debug {max_seqlen=}")

    core_attn = DotProductAttention(
      num_attention_heads=q_local.shape[2] if qkv_format == "bshd" else q_local.shape[1],
      kv_channels=k_local.shape[-1],
      attention_dropout=0.0,
      # attn_mask_type="causal" if is_causal else "no_mask",
      attn_mask_type=attn_mask_type,
      sequence_parallel=False, # FIXME(umiswing): what happen if we enable sp?
      tp_size=dist_info.tp_size,
      get_rng_state_tracker=None,
      tp_group=dist_info.tp_comm_group,
      layer_number=1,
      num_gqa_groups=(k_local.shape[2] if qkv_format == "bshd" else k_local.shape[1]) * dist_info.tp_size,
      attention_type="self",
      cp_group=dist_info.cp_comm_group,
      cp_global_ranks=dist_info.cp_comm_ranks,
      cp_stream=torch.cuda.Stream(),
      cp_comm_type=cp_comm_type,
      softmax_scale=None,
      qkv_format="sbhd",
      window_size=None,
      softmax_type="vanilla"
    ).cuda()

    attn_args = {
      "query_layer": q_local,
      "key_layer": k_local,
      "value_layer": v_local,
      "attention_mask": None,
      "attn_mask_type": attn_mask_type,
      "cu_seqlens_q": cu_seqlens,
      "cu_seqlens_kv": cu_seqlens,
      "cu_seqlens_kv_padded": None,
      "qkv_format": qkv_format,
      "cu_seqlens_q_padded": None,
      "max_seqlen_kv": max_seqlen,
      "max_seqlen_q": max_seqlen,
      "window_size": None,
      "checkpoint_core_attention": False,
      "core_attention_bias_type": "no_bias",
      "core_attention_bias": None,
      "alibi_slopes": None,
      "fast_zero_fill": True,
      "inference_params": None,
      "pad_between_seqs": None,
      "fp8_output": False
    }

    out_local = core_attn(**attn_args)
    out_local.backward(o_grad_local)
    torch.cuda.synchronize()
    dist.barrier(dist_info.cp_comm_group)

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = torch.empty([int(cache_size // 4)], dtype=torch.int32)
    else:
        cache = torch.empty([int(cache_size)], dtype=torch.int8)

    # compute number of warmup and repeat
    n_warmup = max(3, warmup)
    n_repeat = max(5, rep)
    print(f"Record info (Megatron). profile: warmup: {WARM_UP}, rep: {BENCH_TIME}. Actual warmup: {n_warmup}, rep: {n_repeat}")
    # Warm-up
    for _ in range(n_warmup):
        out_local = core_attn(**attn_args)
        out_local.backward(o_grad_local, retain_graph=True)
        dist.barrier(dist_info.cp_comm_group)
        torch.cuda.synchronize()
    # Benchmark
    times_fwd = []
    times_bwd = []
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        dist.barrier(dist_info.cp_comm_group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_local = core_attn(**attn_args)
        dist.barrier(dist_info.cp_comm_group)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out_local.backward(o_grad_local, retain_graph=True)
        dist.barrier(dist_info.cp_comm_group)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times_fwd.append(1000 * (t1 - t0))
        times_bwd.append(1000 * (t2 - t1))
        
    # Record clocks
    dist.barrier(dist_info.cp_comm_group)
    torch.cuda.synchronize()
    # print('pt3')
    return sum(times_fwd) / n_repeat, sum(times_bwd) / n_repeat

def test_te_dpa_cp(
    generate_mask_fn,
    B: int = 16,
    S: int = 8192,
    H: int = 16,
    D: int = 64,
    dtype = 'bf16',
    dist_info:  DistInfo = None,
):
    """
    测试上下文并行FlashMask注意力机制的性能基准
    
    该函数用于测试在分布式并行环境中FlashMask注意力机制的前向传播和后向传播性能，
    支持不同类型的注意力掩码生成策略。

    Args:
        generate_mask_fn: 注意力掩码生成函数，用于生成startend_row_indices和因果关系标记
        B: 批次大小，默认16
        S: 序列长度，默认8192
        H: 注意力头数，默认16
        D: 每个注意力头的维度，默认64
        dtype: 数据类型，默认'bf16'

    Returns:
        tuple: 包含前向传播时间和后向传播时间的元组 (fwd_time, bwd_time)，单位为毫秒
    """
    torch.manual_seed(2024)
    batch_size = B
    seqlen_q = S
    seqlen_kv = S
    num_head = H
    num_head_q = 8 * H
    head_size = D
    # total_k = total_q * 2
    local_seqlen_q = seqlen_q // dist_info.cp_size
    local_seqlen_kv = seqlen_kv // dist_info.cp_size

    if H % dist_info.cp_size == 0:
        cp_comm_type="a2a"
        qkv_format = "thd"
    else:
        cp_comm_type="p2p"
        qkv_format = "bshd"

    if qkv_format == "thd":
      # assert False
      q=torch.randn(batch_size * local_seqlen_q, num_head_q, head_size, dtype=torch.bfloat16).cuda()
      k=torch.randn(batch_size * local_seqlen_kv, num_head, head_size, dtype=torch.bfloat16).cuda()
      v=torch.randn(batch_size * local_seqlen_kv, num_head, head_size, dtype=torch.bfloat16).cuda()
      g=torch.randn(batch_size * local_seqlen_q, num_head_q * head_size, dtype=torch.bfloat16).cuda()
    elif qkv_format == "bshd":
      q=torch.randn(batch_size, local_seqlen_q, num_head_q, head_size, dtype=torch.bfloat16).cuda()
      k=torch.randn(batch_size, local_seqlen_kv, num_head, head_size, dtype=torch.bfloat16).cuda()
      v=torch.randn(batch_size, local_seqlen_kv, num_head, head_size, dtype=torch.bfloat16).cuda()
      g=torch.randn(batch_size, local_seqlen_q, num_head_q * head_size, dtype=torch.bfloat16).cuda()
    else:
      assert False, f"{qkv_format=}"

    q.requires_grad=True
    k.requires_grad=True
    v.requires_grad=True

    cu_seqlens, max_seqlen, causal = None, None, True
    if generate_mask_fn is not None:
        print("enter",generate_mask_fn)
        cu_seqlens, max_seqlen, causal = generate_mask_fn(batch_size, local_seqlen_q, num_head, head_size)

    fwd_time, bwd_time = do_bench_te_dpa_cp(q, k, v, g, cu_seqlens, max_seqlen, causal, dist_info, cp_comm_type=cp_comm_type, qkv_format=qkv_format)
    torch.cuda.synchronize()
    total_time = fwd_time + bwd_time
    return fwd_time, bwd_time, total_time

def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_cu_seqlens(B, S, H, D, causal, doc_seq_lens=[2538, 1742, 3213]):
    cu_seqlens = [0]
    doc_seq_lens = doc_seq_lens * B
    for seqlen in doc_seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    max_seqlen = max(doc_seq_lens)
    return torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32), torch.tensor(max_seqlen, device="cuda", dtype=torch.int32), causal

def dist_init(is_long_seqlen: bool = False):
    # Note(wusiming): in megatron, cp x tp = world_size

    if is_long_seqlen:  # CP 16
        cp_size = 16
        tp_size = 1
    else:
        cp_size = 4
        tp_size = 4

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_tp_group = cp_size
    num_cp_group = tp_size

    # set up communication group for TP
    for i in range(num_tp_group):
        tp_start = i * tp_size
        comm_ranks = list(range(tp_start, tp_start + tp_size))
        comm_group = dist.new_group(comm_ranks, backend="nccl")
        print(f"create tp_group for {comm_ranks}")
        if rank in comm_ranks:
            tp_comm_ranks = comm_ranks
            tp_comm_group = comm_group
    
    # set up communication group for CP
    for i in range(num_cp_group):
        comm_ranks = list(range(i, world_size, tp_size))
        comm_group = dist.new_group(comm_ranks, backend="nccl")
        print(f"create cp_group for {comm_ranks}")
        if rank in comm_ranks:
            cp_comm_ranks = comm_ranks
            cp_comm_group = comm_group

    return DistInfo(cp_size=cp_size,
                    tp_size=tp_size,
                    tp_comm_ranks=tp_comm_ranks,
                    cp_comm_ranks=cp_comm_ranks,
                    tp_comm_group=tp_comm_group,
                    cp_comm_group=cp_comm_group)

def main(examples: List[str] = ["all"], dtype='bf16'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """

    os.environ["NVTE_FP8_DPA_BWD"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "1"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    total_length = 0
    doc_seq_lens_list = []
    rank = dist.get_rank()
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
        
        for H in [1, 4, 8]:
            for D in [128]:
                for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
                    B = 1 if S == 131072 else 2
                    dist_info = dist_init(S >= 131072)

                    doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
                    print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")

                    available_examples = {
                        # "Full": lambda: test_te_dpa_cp(generate_mask_fn=partial(generate_none_mask, causal=False), B=B, S=S, H=H, D=D, dtype=dtype),
                        # "Causal": lambda: test_te_dpa_cp(generate_mask_fn=partial(generate_none_mask, causal=True), B=B, S=S, H=H, D=D, dtype=dtype),
                        "Causal Document Mask": lambda: test_te_dpa_cp(generate_mask_fn=partial(generate_cu_seqlens, doc_seq_lens=doc_seq_lens, causal=True), B=B, S=S, H=H, D=D, dtype=dtype, dist_info=dist_info),
                        "Document Mask": lambda: test_te_dpa_cp(generate_mask_fn=partial(generate_cu_seqlens, doc_seq_lens=doc_seq_lens, causal=False), B=B, S=S, H=H, D=D, dtype=dtype, dist_info=dist_info),
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
                    os.makedirs(f"{dtype}_dist_test_te", exist_ok=True)
                    text_file = open(f"{dtype}_dist_test_te/{output_prefix}_{B}_{S}_{H}_{D}_{idx}_{rank}.csv","w")
                    text_file.write(content2)
                    text_file.close()

if __name__ == "__main__":
    main(**vars(args))
