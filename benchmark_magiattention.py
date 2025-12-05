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

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity
    
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
                [window_size, total_seqlen - window_size],[total_seqlen - window_size,total_seqlen],]
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
                k_ranges.append([cu_seqlens[cu_seqlens_offset], cu_seqlens[cu_seqlens_offset] + total_seqlen])
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
            
            q_ranges.append([offset,offset + mask_offset ])
            k_ranges.append([0, offset])
            is_causal_mapping.append(False)
            
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
        for D in [64, 128]:
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
                    "Full": lambda: test_mask(mask_mod=generate_full_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Causal": lambda: test_mask(mask_mod=generate_causal_mask(total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Sliding Window": lambda: test_mask(mask_mod=generate_sliding_window_mask(window_size=int(S*0.0625),total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Causal Document Mask": lambda: test_mask(mask_mod=generate_causal_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Document Mask": lambda: test_mask(mask_mod=generate_document_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Share Question Mask": lambda: test_mask(mask_mod=generate_share_question_mask(doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Global Sliding Window": lambda: test_mask(mask_mod=generate_global_sliding_window_mask(global_token=16, window_size=int(S*0.0625), total_seqlen = S), B=B, S=S, H=H, D=D, dtype=dtype, disable_fwd_atomic_reduction = True, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Causal Blockwise Mask": lambda: test_mask(mask_mod=generate_causal_blockwise_mask(doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Prefix LM Document Mask": lambda: test_mask(mask_mod=generate_prefix_lm_document_mask(doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "Prefix LM Causal Mask": lambda: test_mask(mask_mod=generate_prefix_lm_causal_mask(seqlen=int(S*0.5),total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
                    "QK-sparse Mask": lambda: test_mask(mask_mod=generate_qk_sparse_mask(maskout_pair=maskout_pair, total_seqlen=S), B=B, S=S, H=H, D=D, dtype=dtype, cp_group = cp_group,cp_mesh = cp_mesh),
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

