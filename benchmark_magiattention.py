import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random

import torch
import torch.nn.functional as F

from tabulate import tabulate
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.mask import AttnMask
from magi_attention.common.range import AttnRange
from magi_attention.functional import flex_flash_attn_func as ffa_func
from magi_attention.api.functools import infer_attn_mask_from_sliding_window
from magi_attention.meta import make_global_bucket_from_qk_ranges
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges

from triton.testing import do_bench

torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

global total_num 
total_num = 0
global cur_num
cur_num = 0



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

# def calculate_attn_flops(
#     q_ranges: AttnRanges,
#     k_ranges: AttnRanges,
#     attn_mask_type: list[AttnMaskType],
#     total_seqlen_q: int,
#     num_heads_q: int,
#     head_dim: int,
# ) -> dict[str, float]:
#     attn_area = make_global_bucket_from_qk_ranges(
#         q_ranges,
#         k_ranges,
#         attn_mask_type,
#         num_chunks=1,
#         chunk_size=total_seqlen_q,
#     ).area

#     flops_fwd = 4 * attn_area * num_heads_q * head_dim
#     flops_bwd = flops_fwd * 2.5  # 2.0(bwd) + 0.5(recompute)
#     flops_1f1b = flops_fwd + flops_bwd

#     return {
#         "fwd": flops_fwd,
#         "bwd": flops_bwd,
#         "1f1b": flops_1f1b,
#     }

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
    disable_fwd_atomic_reduction: bool = False
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
    # H = 8
    GQA_fac = 1
    q = torch.randn(B * S, H * GQA_fac , D, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(B * S, H , D, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn(B * S, H , D, device=device, dtype=data_type, requires_grad=True)
    # q,k,v = [
    #     torch.randn(B * S, H , D, device=device, dtype=data_type, requires_grad=True)
    #     for _ in range(3)
    # ]
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    gradOut = torch.randn(B * S, H * GQA_fac, D, device=device, dtype=data_type)
    
    q_ranges ,k_ranges, attn_mask_type = mask_mod
    q_ranges_tensor = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges_tensor = torch.tensor(k_ranges, device=device, dtype=torch.int32)
    attn_mask_type_tensor = torch.tensor(attn_mask_type, device=device, dtype=torch.int32)
    # print(q.shape)
    # print(k.shape)
    # print(v.shape)
    
    magi_attention_call = lambda: ffa_func(q, k, v, q_ranges_tensor, k_ranges_tensor, attn_mask_type_tensor, disable_fwd_atomic_reduction = disable_fwd_atomic_reduction)

    results = []
    q_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=q_ranges)
    k_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=k_ranges)
    attn_mask_type_: list[AttnMaskType] = [
        [
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.INVCAUSAL,
            AttnMaskType.BICAUSAL,
        ][mask_idx]
        for mask_idx in attn_mask_type
    ]
    sparsity = calculate_sparsity(q_ranges_, k_ranges_, attn_mask_type_, S, S)
    if mask_mod is not None:
        density = 1.0 - sparsity
    else:
        density = 1.0

    ç = 1.0 - density

    # Forward pass
    fwd_time_ms = do_bench(magi_attention_call)
    torch._functorch.config.donated_buffer=False
    # Backward pass
    magi_out, _ = magi_attention_call()
    bwd_time_ms = do_bench(lambda: magi_out.backward(gradOut, retain_graph=True))
    
    q.grad = None
    k.grad = None
    v.grad = None
    magi_out.backward(gradOut, retain_graph=True)
 
    global cur_num
        
    # np.save(f"tmp_res/q_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", q.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/k_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", k.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/v_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", v.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/gradOut_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", gradOut.view(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/magi_out_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", magi_out.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/q_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", q.grad.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/k_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", k.grad.type(torch.float32).detach().cpu().numpy())
    # np.save(f"tmp_res/v_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy", v.grad.type(torch.float32).detach().cpu().numpy())

    cur_num += 1
    
    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd') * GQA_fac
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd') * GQA_fac
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd') * GQA_fac

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
        attn_type_map.append(1)

        q_ranges.append([0, window_size_single])
        k_ranges.append([window_size_single, total_seqlen])
        attn_type_map.append(1)

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
    """处理单个文档序列块，生成对应的注意力掩码范围
    
    Args:
        seqlens: 当前文档中各段的长度列表
        cu_seqlens: 累计序列长度列表
        cu_seqlens_offset: 当前文档在累计序列中的偏移量
        q_ranges: 查询范围列表（会被修改）
        k_ranges: 键范围列表（会被修改）
        is_causal_mapping: 因果掩码标记列表（会被修改）
    """
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
    """扁平化文档序列长度并计算累计序列长度
    
    Args:
        doc_seq_lens: 文档序列长度列表，每个元素是一个文档中各段的长度列表
        
    Returns:
        tuple[list[int], list[int]]: (扁平化后的序列长度, 累计序列长度)
    """
    seqlens_flatten = [num for sublist in doc_seq_lens for num in sublist]
    cu_seqlens = seqlens2cu_seqlens(seqlens_flatten)
    return seqlens_flatten, cu_seqlens


def generate_share_question_mask(doc_seq_lens=[2538, 1742, 3213]) -> tuple[list[list[int]], list[list[int]], list[bool]]:
    """生成共享问题注意力掩码
    
    Args:
        doc_seq_lens: 文档序列长度列表，每个元素是一个文档中各段的长度列表，默认值为[2538, 1742, 3213]
        
    Returns:
        tuple[list[list[int]], list[list[int]], list[bool]]: 
            (查询范围列表, 键范围列表, 因果掩码标记列表)
    """
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

def main(examples: List[str] = ["all"], dtype='bf16'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    total_length = 0
    doc_seq_lens_list = []
    # rank = int(os.environ.get("RANK", 0))
    # cp_group = init_dist_environment(
    #     world_size=WORLD_SIZE
    # )
    # cp_mesh  = init_hierarchical_mesh(WORLD_SIZE)
    with open('kernel_test_seq_info.txt', 'r') as f:
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
            # H = 64
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
                    "Full": lambda: test_mask(mask_mod=generate_full_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal": lambda: test_mask(mask_mod=generate_causal_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Sliding Window": lambda: test_mask(mask_mod=generate_sliding_window_mask(window_size=int(S*0.0625),total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Causal Document Mask": lambda: test_mask(mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Document Mask": lambda: test_mask(mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Share Question Mask": lambda: test_mask(mask_mod=generate_share_question_mask(doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True),
                    "Global Sliding Window": lambda: test_mask(mask_mod=generate_global_sliding_window_mask(global_token=1024, window_size=1024, total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True),
                    "Causal Blockwise Mask": lambda: test_mask(mask_mod=generate_causal_blockwise_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Prefix LM Document Mask": lambda: test_mask(mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                    "Prefix LM Causal Mask": lambda: test_mask(mask_mod=generate_prefix_lm_causal_mask(seqlen=int(S*0.5),total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    "QK-sparse Mask": lambda: test_mask(mask_mod=generate_qk_sparse_mask(maskout_pair=maskout_pair, total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype),
                    # "Random Eviction Mask": lambda: test_mask(mask_mod=generate_random_eviction_mask(start_row=S//2, total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True),
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
                os.makedirs(f"{dtype}", exist_ok=True)
                text_file = open(f"{dtype}/magiattention_{B}_{S}_{H}_{D}_{idx}.csv","w")
                text_file.write(content2)
                text_file.close()
                # assert False

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

