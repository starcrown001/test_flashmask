import os
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from pydantic import TypeAdapter
from tabulate import tabulate

from sparsity_utils import ranges_block_sparsity

ROOT_DIR = Path(__file__).resolve().parents[2]
MAGI_ROOT = ROOT_DIR / "magiattn" / "MagiAttention"

if str(MAGI_ROOT) not in sys.path:
    sys.path.insert(0, str(MAGI_ROOT))

import magi_attention
from exps.dist_attn.benchmark.enums import FlashMaskType
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType


def _load_module(module_name: str, file_path: Path):
    spec = spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stable = _load_module(
    "run_magi_benchmark_mod",
    MAGI_ROOT / "exps" / "dist_attn" / "run_magi_benchmark.py",
)

torch.set_default_device("cuda")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
CP_SIZE = WORLD_SIZE
ITERATION = 40
WARMUP = 5
GQA_FAC = 8

BENCH_MODE: Any = None
ATTN_CONFIG: Any = None
BENCH_CONFIG: Any = None
DATA_CONFIG: Any = None
SAMPLE_CONFIG: Any = None
SEED: Any = None
TOTAL_SEQLENS: List[int] = []


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def copy_mask_for_batchs(q_ranges, k_ranges, attn_mask_type, seqlen_qkv, bs):
    q_ranges_multi = q_ranges.copy()
    k_ranges_multi = k_ranges.copy()
    attn_mask_type_multi = attn_mask_type.copy()
    for _ in range(1, bs):
        q_ranges_i = [[x + seqlen_qkv, y + seqlen_qkv] for [x, y] in q_ranges]
        k_ranges_i = [[x + seqlen_qkv, y + seqlen_qkv] for [x, y] in k_ranges]
        q_ranges_multi.extend(q_ranges_i)
        k_ranges_multi.extend(k_ranges_i)
        attn_mask_type_multi.extend(attn_mask_type)
    return q_ranges_multi, k_ranges_multi, attn_mask_type_multi


def generate_prefix_lm_document_mask(
    doc_seq_lens=[(2538, 1742), (1742, 1742), (3213, 3213)],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
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

    return q_ranges, k_ranges, attn_mask_type


def generate_causal_document_mask(
    doc_seq_lens=[2538, 1742, 3213],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    attn_mask_type = [1] * len(seqlens)
    return ranges, ranges, attn_mask_type


def generate_document_mask(
    doc_seq_lens=[2538, 1742, 3213],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    seqlens = doc_seq_lens
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    ranges = []
    for i in range(len(seqlens)):
        ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
    attn_mask_type = [0] * len(seqlens)
    return ranges, ranges, attn_mask_type


def split_sequence(sequence_length):
    if sequence_length < 3:
        raise ValueError("序列长度必须至少为 3，以保证能够分配给一个 Question 和两个 Answer。")

    num_answers = random.randint(2, 6)
    lengths = [1] * (num_answers + 1)
    remaining_length = sequence_length - sum(lengths)

    for _ in range(remaining_length):
        index = random.randint(0, num_answers)
        lengths[index] += 1

    return lengths


def cal_flops(B, H, Sq, Sk, D, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def cal_tflops(flops, time_ms):
    return flops * (1e3 / time_ms) / 1e12


def _to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_attn_mask_types(attn_mask_type: list[int]) -> list[AttnMaskType]:
    mapping = {
        0: AttnMaskType.FULL,
        1: AttnMaskType.CAUSAL,
        2: AttnMaskType.INVCAUSAL,
        3: AttnMaskType.BICAUSAL,
    }
    return [mapping.get(mask_type, AttnMaskType.FULL) for mask_type in attn_mask_type]


def load_py_as_dict(config_path: str) -> dict[str, Any]:
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


def load_bench_config(config_file: str):
    config_dict = load_py_as_dict(config_file)

    global BENCH_MODE, BENCH_CONFIG, ATTN_CONFIG, DATA_CONFIG, SAMPLE_CONFIG
    global SEED, TOTAL_SEQLENS, CP_SIZE

    BENCH_MODE = config_dict["BENCH_MODE"]
    BENCH_CONFIG = config_dict["BENCH_CONFIG"]
    ATTN_CONFIG = config_dict["ATTN_CONFIG"]
    DATA_CONFIG = config_dict["DATA_CONFIG"]
    SAMPLE_CONFIG = config_dict["SAMPLE_CONFIG"]
    SEED = config_dict["SEED"]

    CP_SIZE = WORLD_SIZE
    TOTAL_SEQLENS = [s * CP_SIZE for s in DATA_CONFIG.seqlens_per_rank]

    _stable.BENCH_MODE = BENCH_MODE
    _stable.BENCH_CONFIG = BENCH_CONFIG
    _stable.ATTN_CONFIG = ATTN_CONFIG
    _stable.DATA_CONFIG = DATA_CONFIG
    _stable.SAMPLE_CONFIG = SAMPLE_CONFIG
    _stable.SEED = SEED
    _stable.TOTAL_SEQLENS = TOTAL_SEQLENS
    _stable.WORLD_SIZE = WORLD_SIZE

    if BENCH_CONFIG.output_path is not None:
        os.makedirs(BENCH_CONFIG.output_path, exist_ok=True)


def _bench_time_ms(fn, grad_to_none=None, warmup=WARMUP, rep=ITERATION, return_mode="mean"):
    perf_dict = _stable.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        grad_to_none=grad_to_none,
        quantiles=None,
        return_mode=return_mode,
        return_flops=True,
        return_mem=False,
        mem_record_mode="peak",
        to_gc_collect=False,
        to_empty_cache=False,
    )
    return perf_dict["flops"]


def test_mask(
    mask_mod: Optional[tuple[list[int], list[int], list[int]]] = None,
    B: int = 1,
    H: int = 1,
    S: int = 8192,
    D: int = 128,
    dtype: str = "bf16",
):
    if mask_mod is None:
        raise ValueError("mask_mod must not be None")

    data_type = _to_torch_dtype(dtype)
    q_ranges, k_ranges, attn_mask_type = mask_mod

    sparsity = ranges_block_sparsity(q_ranges, k_ranges, attn_mask_type, S, S)
    density = 1.0 - sparsity

    q_ranges, k_ranges, attn_mask_type = copy_mask_for_batchs(
        q_ranges, k_ranges, attn_mask_type, S, B
    )

    q_ranges_ = AttnRanges.from_ranges(ranges=q_ranges)
    k_ranges_ = AttnRanges.from_ranges(ranges=k_ranges)
    attn_mask_type_ = _to_attn_mask_types(attn_mask_type)

    cp_group_or_mesh = _stable.init_magi_cp_group(WORLD_SIZE)

    _stable.already_known_oom_before_run = False
    fwd_fn = _stable.run_magi_attn(
        total_seqlen=B * S,
        embed_dim=H * D,
        num_heads_q=H * GQA_FAC,
        num_heads_kv=H,
        head_dim=D,
        dtype=data_type,
        q_ranges=q_ranges_,
        k_ranges=k_ranges_,
        world_size=CP_SIZE,
        chunk_size=ATTN_CONFIG.chunk_size,
        attn_mask_type=attn_mask_type_,
        cp_group_or_mesh=cp_group_or_mesh,
        wd="fwd",
        iteration=0,
    )
    if _stable.already_known_oom_before_run:
        raise RuntimeError("OOM before forward benchmark")

    fwd_time_ms = _bench_time_ms(
        fwd_fn,
        grad_to_none=None,
        warmup=WARMUP,
        rep=ITERATION,
        return_mode=BENCH_CONFIG.bench_mode,
    )

    _stable.already_known_oom_before_run = False
    bwd_fn = _stable.run_magi_attn(
        total_seqlen=B * S,
        embed_dim=H * D,
        num_heads_q=H * GQA_FAC,
        num_heads_kv=H,
        head_dim=D,
        dtype=data_type,
        q_ranges=q_ranges_,
        k_ranges=k_ranges_,
        world_size=CP_SIZE,
        chunk_size=ATTN_CONFIG.chunk_size,
        attn_mask_type=attn_mask_type_,
        cp_group_or_mesh=cp_group_or_mesh,
        wd="bwd",
        iteration=0,
    )
    if _stable.already_known_oom_before_run:
        raise RuntimeError("OOM before backward benchmark")

    bwd_time_ms = _bench_time_ms(
        bwd_fn,
        grad_to_none=None,
        warmup=WARMUP,
        rep=ITERATION,
        return_mode=BENCH_CONFIG.bench_mode,
    )

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode="fwd") * GQA_FAC / CP_SIZE
    bwd_flops = density * cal_flops(B, H, S, S, D, mode="bwd") * GQA_FAC / CP_SIZE
    total_flops = density * cal_flops(B, H, S, S, D, mode="fwd_bwd") * GQA_FAC / CP_SIZE

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return (
        fwd_time_ms,
        bwd_time_ms,
        total_time_ms,
        fwd_flops,
        bwd_flops,
        total_flops,
        fwd_tflops,
        bwd_tflops,
        total_tflops,
        sparsity,
    )


def main(examples: List[str] = ["all"], dtype="bf16", config="none", fast_eval=False):
    del config

    rank = int(os.environ.get("RANK", 0))
    total_length = 0
    doc_seq_lens_list = []

    use_mp = WORLD_SIZE != CP_SIZE
    input_file = "kernel_test_dist_seq_info.txt"
    if use_mp and fast_eval:
        input_file = "kernel_test_dist_seq_info-32k.txt"
    elif fast_eval:
        input_file = "kernel_test_dist_seq_info-128k.txt"

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

    for H in [1]:
        D = 128
        for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
            B = 2 if use_mp else 1

            doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
            offset = 0
            maskout_pair = []
            print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")

            if sum(qksparse_mask) == 0:
                maskout_pair = [(1024, 538), (2358, 1700)]
            else:
                for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                    if is_maskout:
                        maskout_pair.append((offset, doc_seq))
                    offset += doc_seq

            _ = maskout_pair
            _ = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]

            available_examples = {
                "Causal Document Mask": lambda: test_mask(
                    mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens),
                    B=B,
                    S=S,
                    H=H,
                    D=D,
                    dtype=dtype,
                ),
                "Document Mask": lambda: test_mask(
                    mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens),
                    B=B,
                    S=S,
                    H=H,
                    D=D,
                    dtype=dtype,
                ),
                "Prefix LM Document Mask": lambda: test_mask(
                    mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens),
                    B=B,
                    S=S,
                    H=H,
                    D=D,
                    dtype=dtype,
                ),
            }

            if "all" in examples:
                ex_to_run = list(available_examples.keys())
            else:
                ex_to_run = examples

            results = []
            for ex in ex_to_run:
                if ex in available_examples:
                    print(ex, flush=True)
                    (
                        fw_time,
                        bw_time,
                        total_time,
                        fw_flops,
                        bw_flops,
                        total_flops,
                        fw_tflops,
                        bw_tflops,
                        total_tflops,
                        sparsity,
                    ) = available_examples[ex]()
                    results.append(
                        [
                            ex,
                            f"{fw_time:.4f}",
                            f"{bw_time:.4f}",
                            f"{total_time:.4f}",
                            f"{fw_flops:.4f}",
                            f"{bw_flops:.4f}",
                            f"{total_flops:.4f}",
                            f"{fw_tflops:.4f}",
                            f"{bw_tflops:.4f}",
                            f"{total_tflops:.4f}",
                            f"{sparsity:.4f}",
                        ]
                    )
                else:
                    print(f"Warning: Unknown example key '{ex}'. Skipping.")

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

            print(tabulate(results, headers=headers, tablefmt="grid"))

            content2 = tabulate(results, headers=headers, tablefmt="tsv")
            os.makedirs(f"{dtype}_dist_test_cpu", exist_ok=True)
            with open(
                f"{dtype}_dist_test_cpu/magiattention_{rank}_{CP_SIZE}_{B}_{S}_{H}_{D}_{idx}.csv",
                "w",
            ) as text_file:
                text_file.write(content2)


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")

    parser = ArgumentParser(
        description="Run specific examples with stable MagiAttention benchmark core."
    )
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples.",
    )
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--config", type=str, default="magi_benchmark_conf.py")
    parser.add_argument("--fast_eval", type=bool, default=False)

    args = parser.parse_args()
    load_bench_config(args.config)
    main(**vars(args))