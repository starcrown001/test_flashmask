# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MagiAttention-only distributed benchmark configuration.
Used by run_magi_benchmark.py via --config magi_benchmark_conf.py.

Config class names must start with a capital letter to be recognized.
"""

from dataclasses import dataclass

import torch

from magi_attention.common.enum import AttnOverlapMode
from magi_attention.meta.solver.dispatch_solver import MinHeapDispatchAlg

from enum import Enum


class FlashMaskType(Enum):
    FULL = "full"
    CAUSAL = "causal"
    CAUSAL_DOCUMENT = "causal_document"
    FULL_DOCUMENT = "full_document"
    SHARE_QUESTION = "share_question"
    CAUSAL_BLOCKWISE = "causal_blockwise"
    PREFIX_LM_DOCUMENT = "prefix_lm_document"
    PREFIX_LM_CAUSAL = "prefix_lm_causal"
    QK_SPARSE = "qk_sparse"
    HASH_SPARSE = "hash_sparse"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_CAUSAL = "sliding_window_causal"
    GLOBAL_SLIDING_WINDOW = "global_sliding_window"
    BLOCK_CAUSAL_DOCUMENT = "block_causal_document"
SEED = 42


@dataclass
class BENCH_MODE:
    """
    Benchmark runtime mode configuration.
        - enable_profile: whether to enable nsys profiling.
        - profile_only: if True, skip statistic recording.
        - stat_warmup_iters: number of warmup iterations.
        - stat_iters: number of timing iterations.
    """

    enable_profile = False
    profile_only = False
    stat_warmup_iters = 5
    stat_iters = 20
    profile_warmup_iters = 1
    profile_iters = 3


@dataclass
class BENCH_CONFIG:
    """
    Benchmark combination configuration.
        - quantiles: quantile points for latency/throughput summary.
        - bench_flops: whether to benchmark flops (TFLOPs/s).
        - bench_mem: whether to benchmark memory.
        - bench_mode: aggregation mode (mean/median/min/max) when quantiles=None.
        - output_path: output folder for CSV results and plots.
        - mask_pattern: attention mask types to evaluate.
        - workload: pipeline pass modes ("fwd", "bwd", "1f1b").
    """

    quantiles = [0.5, 0.2, 0.8]
    bench_flops = True
    bench_mem = False
    bench_mode = "mean"
    output_path = "./outs_magi"
    mask_pattern = [
        FlashMaskType.FULL,
        FlashMaskType.CAUSAL,
        FlashMaskType.FULL_DOCUMENT,
        FlashMaskType.CAUSAL_DOCUMENT,
    ]
    workload = [
        "fwd",
        "bwd",
        "1f1b",
    ]
    gc_per_iter = False
    empty_cache_per_iter = False


@dataclass
class SAMPLE_CONFIG:
    """
    Varlen mask sampler configuration (only used for DOCUMENT mask types).
        - dataset_path: path to the document length distribution CSV.
        - pack_num: number of random document packs to average over.
        - chunk_ratio: max single-doc length as fraction of pack_len.
        - is_binned: whether the dataset CSV is binned (intervals + counts).
        - to_attn_ranges: convert to AttnRanges objects.
        - drop_thres: drop samples longer than this threshold (-1 = no drop).
    """

    dataset_path = "./benchmark/datasets/default/doc_length_distribution.csv"
    pack_num = 20
    chunk_ratio = 0.25
    is_binned = True
    to_attn_ranges = True
    drop_thres = -1


@dataclass
class DATA_CONFIG:
    """
    Data configuration.
        - seqlens_per_rank: list of per-rank sequence lengths to sweep.
          Total seqlen = seqlen_per_rank * world_size.
          With 8 GPUs: [4K,8K,16K,32K,64K] → [32K,64K,128K,256K,512K] total.
        - embed_dim: embedding / hidden dimension.
        - head_dim: per-head dimension.
        - num_heads_q: number of query heads.
        - num_heads_kv: number of key/value heads (GQA).
        - dtype: tensor dtype.
    """

    seqlens_per_rank = [4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024]
    embed_dim = 1024
    head_dim = 128
    num_heads_q = 64
    num_heads_kv = 8
    dtype = torch.bfloat16


@dataclass
class ATTN_CONFIG:
    """
    MagiAttention configuration.
        - chunk_size: dispatch chunk granularity.
        - dispatch_alg: dispatch load-balancing algorithm.
        - enable_overlap: enable compute-communication overlap.
        - overlap_mode: STATIC or DYNAMIC overlap scheduling.
        - degree / min_chunk_size / max_num_chunks: overlap tuning knobs.
        - num_sms / nvl_* / rdma_*: native GrpColl comm buffer configuration.
    """

    # dispatch & overlap
    chunk_size = 2048
    dispatch_alg = MinHeapDispatchAlg
    enable_overlap = True
    overlap_mode = AttnOverlapMode.STATIC
    degree = 2
    min_chunk_size = 512
    max_num_chunks = 4096

    # native grpcoll buffer sizes (single-node: rdma is unused)
    num_sms = 48
    nvl_chunk_size = 4
    nvl_buffer_size = 144
    rdma_chunk_size = 16
    rdma_buffer_size = 128
    num_nvl_bytes = int(5e9)   # ~5 GB NVLink buffer
    num_rdma_bytes = int(5e9)  # ~5 GB RDMA buffer (ignored for single-node)
