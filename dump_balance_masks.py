"""
Dump before/after balance masks for eval_comm_cost.py evaluation.

- before: 
    - Default: global original mask (single file, shared by all ranks)
    - With --before_is_local: per-rank masks from balance_flashmask_input
- after (controlled by --after_method):
    - inter_machine:           balance_flashmask_input_inter_machine (default)
    - inter_machine_locality:  balance_flashmask_input_inter_machine + use_locality_swap
    - inter_machine_intra_balance: inter_machine + use_intra_machine_balance
    - inter_machine_locality_intra_balance: inter_machine + locality_swap + intra_machine_balance
    - comm:                    balance_flashmask_input_comm
    - compute_only:            balance_flashmask_input

Output format: {before|after}_{rank}_mask_{idx}_{mask_name}.npy
Each npy is startend_row_indices [1, 1, S, 2]

Usage:
    # Default: inter_machine
    python dump_balance_masks.py --cp_size 32 --output_dir record_mask_inter_machine

    # inter_machine + locality swap
    python dump_balance_masks.py --cp_size 32 --after_method inter_machine_locality

    # inter_machine + intra-machine balance
    python dump_balance_masks.py --cp_size 32 --after_method inter_machine_intra_balance

    # inter_machine + locality swap + intra-machine balance
    python dump_balance_masks.py --cp_size 32 --after_method inter_machine_locality_intra_balance

    # Use comm-aware balance
    python dump_balance_masks.py --cp_size 32 --after_method comm --comm_penalty 0.2

    # Use compute-only balance
    python dump_balance_masks.py --cp_size 32 --after_method compute_only
"""

import sys
import os
import numpy as np
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xhy-flash-attention', 'flashmask'))

import paddle

from flash_mask.cp_balance import balance_flashmask_input, balance_flashmask_input_inter_machine, balance_flashmask_input_comm


# ============================================================================
#  Mask 生成函数
# ============================================================================

def generate_causal_document_mask(B, S, H, D, doc_seq_lens):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens_padded = list(doc_seq_lens)
    doc_seq_lens_padded[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens_padded)

    lts = np.repeat(seq_cusums, doc_seq_lens_padded)
    lts = paddle.to_tensor(lts, dtype=paddle.int32).reshape((1, 1, S, 1))
    ute = paddle.arange(S, dtype='int32').reshape((1, 1, S, 1))
    startend_row_indices = paddle.concat([lts, ute], axis=-1)
    startend_row_indices = startend_row_indices.repeat_interleave(B, 0)
    return startend_row_indices


def generate_document_mask(B, S, H, D, doc_seq_lens):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)

    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    return startend_row_indices


def generate_prefix_lm_document_mask(B, S, H, D, doc_seq_lens):
    """doc_seq_lens: list of (prefix_length, seq_length)"""
    total_seq_len = sum(sl for _, sl in doc_seq_lens)
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i + 1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend(
            [cur_len_so_far] * prefix_length
            + list(range(cur_len_so_far + prefix_length, cur_len_so_far + seq_length))
        )
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    return startend_row_indices


# ============================================================================
#  主流程
# ============================================================================

def parse_input_file(input_file):
    """解析 benchmark 使用的输入文件。"""
    samples = []
    total_length = 0
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
                sample_name = line.split(",")[0].strip()
                samples.append((total_length, doc_list, qksparse_mask, sample_name))
    return samples


def dump_masks_for_sample(
    startend_row_indices,
    mask_name,
    idx,
    cp_size,
    output_dir,
    before_is_local=False,
    after_method="inter_machine",
    balance_chunk_size=2048,
    q_block_size=128,
    k_block_size=128,
    buckets_per_machine=8,
    epsilon=1,
    max_swap_iterations=100,
    comm_penalty=0.1,
    adaptive_comm=False,
    use_locality_swap=False,
    max_locality_iterations=100,
):
    """
    对一个 mask 配置，dump 所有 rank 的 before/after masks。
    
    before_is_local:
        - True:  before-mask 按 rank 分别保存（每个 rank 的处理结果）
        - False: before-mask 只保存一份全局原始 mask（所有 rank 共享）
    
    after_method:
        - "inter_machine":          balance_flashmask_input_inter_machine
        - "inter_machine_locality": balance_flashmask_input_inter_machine + use_locality_swap
        - "inter_machine_intra_balance": inter_machine + use_intra_machine_balance
        - "inter_machine_locality_intra_balance": inter_machine + locality_swap + intra_machine_balance
        - "comm":                   balance_flashmask_input_comm
        - "compute_only":           balance_flashmask_input (仅计算均衡)
    """
    os.makedirs(output_dir, exist_ok=True)
    B, H, S, _ = startend_row_indices.shape

    print(f"  Dumping {mask_name} (idx={idx}, S={S}, cp_size={cp_size}, "
          f"before_is_local={before_is_local}, after_method={after_method})")

    # -- Before-mask --
    if before_is_local:
        # 为每个 rank 生成并保存 before-mask
        for rank in range(cp_size):
            before_local_mask, _ = balance_flashmask_input(
                startend_row_indices, cp_size, rank,
                balance_chunk_size=balance_chunk_size,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
            )
            before_np = before_local_mask.cpu().numpy()
            before_path = os.path.join(output_dir, f"before_{rank}_mask_{idx}_{mask_name}.npy")
            # np.save(before_path, before_np)
        print(f"    Saved {cp_size} before-mask files (per-rank)")
    else:
        # 只保存一份全局原始 mask
        before_path = os.path.join(output_dir, f"before_0_mask_{idx}_{mask_name}.npy")
        # np.save(before_path, startend_row_indices.cpu().numpy())
        print(f"    Saved 1 before-mask file (global)")

    # -- After-mask (始终按 rank 分别保存) --
    for rank in range(cp_size):
        if after_method in ("inter_machine", "inter_machine_locality",
                            "inter_machine_intra_balance",
                            "inter_machine_locality_intra_balance"):
            after_local_mask, _ = balance_flashmask_input_inter_machine(
                startend_row_indices, cp_size, rank,
                balance_chunk_size=balance_chunk_size,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
                buckets_per_machine=buckets_per_machine,
                epsilon=epsilon,
                max_swap_iterations=max_swap_iterations,
                use_locality_swap=(after_method in ("inter_machine_locality",
                                                     "inter_machine_locality_intra_balance")),
                max_locality_iterations=max_locality_iterations,
                use_intra_machine_balance=(after_method in ("inter_machine_intra_balance",
                                                             "inter_machine_locality_intra_balance")),
            )
        elif after_method == "comm":
            after_local_mask, _ = balance_flashmask_input_comm(
                startend_row_indices, cp_size, rank,
                balance_chunk_size=balance_chunk_size,
                q_block_size=q_block_size,
                comm_penalty=comm_penalty,
                adaptive_comm=adaptive_comm,
            )
        else:  # compute_only
            after_local_mask, _ = balance_flashmask_input(
                startend_row_indices, cp_size, rank,
                balance_chunk_size=balance_chunk_size,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
            )
        after_np = after_local_mask.cpu().numpy()
        after_path = os.path.join(output_dir, f"after_{rank}_mask_{idx}_{mask_name}.npy")
        np.save(after_path, after_np)

    after_count = cp_size
    before_count = cp_size if before_is_local else 1
    print(f"    Saved {before_count + after_count} files (before: {before_count}, after: {after_count})")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Dump before/after balance masks for eval_comm_cost.py evaluation")
    parser.add_argument("--input_file", type=str, default="/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/benchmark_flashmask_cp/kernel_test_dist_seq_info.txt")
    parser.add_argument("--cp_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--buckets_per_machine", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max_swap_iterations", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples (-1 = all)")
    parser.add_argument("--before_is_local", action="store_true",
                        help="If set, dump per-rank before-masks; otherwise dump single global before-mask")
    parser.add_argument("--after_method", type=str, default="inter_machine",
                        choices=["inter_machine", "inter_machine_locality",
                                 "inter_machine_intra_balance",
                                 "inter_machine_locality_intra_balance",
                                 "comm", "compute_only"],
                        help="Balance method for after-mask: "
                             "'inter_machine' = balance_flashmask_input_inter_machine, "
                             "'inter_machine_locality' = inter_machine + locality swap, "
                             "'inter_machine_intra_balance' = inter_machine + intra-machine balance, "
                             "'inter_machine_locality_intra_balance' = inter_machine + locality swap + intra-machine balance, "
                             "'comm' = balance_flashmask_input_comm, "
                             "'compute_only' = balance_flashmask_input (same as before)")
    parser.add_argument("--comm_penalty", type=float, default=0.1,
                        help="comm_penalty for balance_flashmask_input_comm (default: 0.1)")
    parser.add_argument("--adaptive_comm", action="store_true",
                        help="adaptive_comm for balance_flashmask_input_comm")
    parser.add_argument("--max_locality_iterations", type=int, default=100,
                        help="Max iterations for locality swap (default: 100)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save npy files (default: record_mask_{after_method})")
    args = parser.parse_args()

    # 默认 output_dir 以 after_method 为前缀
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), f"record_mask_{args.after_method}"
        )

    # 定位输入文件
    input_file = args.input_file
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)

    LOCAL_CHUNK_SIZE = 8192
    expected_S = LOCAL_CHUNK_SIZE * args.cp_size
    print(f"Input file: {input_file}")
    print(f"cp_size: {args.cp_size}, expected S: {expected_S}")
    print(f"after_method: {args.after_method}")
    print(f"buckets_per_machine: {args.buckets_per_machine}, epsilon: {args.epsilon}")
    print(f"comm_penalty: {args.comm_penalty}, adaptive_comm: {args.adaptive_comm}")
    print(f"max_locality_iterations: {args.max_locality_iterations}")
    print(f"Output dir: {args.output_dir}")
    print()

    samples = parse_input_file(input_file)
    B = 1
    H = args.num_heads
    D = 128

    count = 0
    for idx, (S, prefix_doc_seq_lens, qksparse_mask, sample_name) in enumerate(samples):
        # 仅处理匹配 cp_size 的序列长度
        if S != expected_S:
            continue

        doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
        print(f"[{idx}] {sample_name} (S={S})")

        mask_configs = [
            ("causal_doc", partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens)),
            ("doc", partial(generate_document_mask, doc_seq_lens=doc_seq_lens)),
            ("prefix_lm_doc", partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens)),
        ]

        for mask_name, mask_fn in mask_configs:
            try:
                startend_row_indices = mask_fn(B, S, H, D)
                if H > 1 and startend_row_indices.shape[1] == 1:
                    startend_row_indices = startend_row_indices.repeat_interleave(H, 1)

                dump_masks_for_sample(
                    startend_row_indices,
                    mask_name,
                    idx,
                    args.cp_size,
                    args.output_dir,
                    before_is_local=args.before_is_local,
                    after_method=args.after_method,
                    balance_chunk_size=args.chunk_size,
                    buckets_per_machine=args.buckets_per_machine,
                    epsilon=args.epsilon,
                    max_swap_iterations=args.max_swap_iterations,
                    comm_penalty=args.comm_penalty,
                    adaptive_comm=args.adaptive_comm,
                    max_locality_iterations=args.max_locality_iterations,
                )
            except Exception as e:
                assert False, e

        count += 1
        if args.max_samples > 0 and count >= args.max_samples:
            break

    print(f"\nDone. Total samples processed: {count}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nTo evaluate, run:")
    if args.before_is_local:
        print(f"  python eval_comm_cost.py --mask_dir {args.output_dir} --before_is_local")
    else:
        print(f"  python eval_comm_cost.py --mask_dir {args.output_dir} --plot --plot_computation")


if __name__ == "__main__":
    main()
