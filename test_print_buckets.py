"""
仅打印各阶段的 rank/bucket 分配信息（task idx 和 weight），不保存文件，不遍历 rank。

balance_flashmask_input_inter_machine 内部已包含 _print_bucket_summary，
调用一次（cp_rank=0）即可打印所有 rank 的全局分配信息。

Usage:
    python test_print_buckets.py --cp_size 32
    python test_print_buckets.py --cp_size 32 --after_method inter_machine_locality_intra_balance
"""

import sys
import os
import numpy as np
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xhy-flash-attention', 'flashmask'))

import paddle

from flash_mask.cp_balance import balance_flashmask_input_inter_machine


# ============================================================================
#  Mask 生成函数（与 dump_balance_masks.py 一致）
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


def print_buckets_for_sample(
    startend_row_indices,
    mask_name,
    idx,
    cp_size,
    after_method="inter_machine",
    balance_chunk_size=2048,
    q_block_size=128,
    k_block_size=128,
    buckets_per_machine=8,
    epsilon=0.05,
    max_swap_iterations=100,
    max_locality_iterations=100,
):
    """
    对一个 mask 配置，调用一次 balance_flashmask_input_inter_machine（cp_rank=0），
    打印各阶段所有 rank 的 bucket 信息。不保存文件，不遍历 rank。
    """
    B, H, S, _ = startend_row_indices.shape

    use_locality_swap = after_method in ("inter_machine_locality",
                                          "inter_machine_locality_intra_balance")
    use_intra_machine_balance = after_method in ("inter_machine_intra_balance",
                                                   "inter_machine_locality_intra_balance")

    print(f"\n{'='*70}")
    print(f"  mask={mask_name}, idx={idx}, S={S}, cp_size={cp_size}")
    print(f"  locality_swap={use_locality_swap}, intra_machine_balance={use_intra_machine_balance}")
    print(f"{'='*70}")

    # 只调用一次，内部的 _print_bucket_summary 会打印所有 rank 的全局信息
    _, buckets = balance_flashmask_input_inter_machine(
        startend_row_indices, cp_size, cp_rank=0,
        balance_chunk_size=balance_chunk_size,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        buckets_per_machine=buckets_per_machine,
        epsilon=epsilon,
        max_swap_iterations=max_swap_iterations,
        use_locality_swap=use_locality_swap,
        max_locality_iterations=max_locality_iterations,
        use_intra_machine_balance=use_intra_machine_balance,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Print rank/bucket allocation info for each balance phase (no file I/O)")
    parser.add_argument("--input_file", type=str, default="/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/test_flashmask/benchmark_flashmask_cp/kernel_test_dist_seq_info.txt")
    parser.add_argument("--cp_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--buckets_per_machine", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max_swap_iterations", type=int, default=100)
    parser.add_argument("--max_locality_iterations", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples (-1 = all)")
    parser.add_argument("--after_method", type=str, default="inter_machine",
                        choices=["inter_machine", "inter_machine_locality",
                                 "inter_machine_intra_balance",
                                 "inter_machine_locality_intra_balance"],
                        help="Balance method")
    parser.add_argument("--mask_type", type=str, default=None,
                        choices=["causal_doc", "doc", "prefix_lm_doc"],
                        help="Only run a specific mask type (default: all)")
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)

    LOCAL_CHUNK_SIZE = 8192
    expected_S = LOCAL_CHUNK_SIZE * args.cp_size
    print(f"Input file: {input_file}")
    print(f"cp_size: {args.cp_size}, expected S: {expected_S}")
    print(f"after_method: {args.after_method}")
    print(f"buckets_per_machine: {args.buckets_per_machine}, epsilon: {args.epsilon}")
    print(f"max_locality_iterations: {args.max_locality_iterations}")

    samples = parse_input_file(input_file)
    B = 1
    H = args.num_heads
    D = 128

    count = 0
    for idx, (S, prefix_doc_seq_lens, qksparse_mask, sample_name) in enumerate(samples):
        if S != expected_S:
            continue

        doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
        print(f"\n[{idx}] {sample_name} (S={S})")

        all_mask_configs = [
            ("causal_doc", partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens)),
            ("doc", partial(generate_document_mask, doc_seq_lens=doc_seq_lens)),
            ("prefix_lm_doc", partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens)),
        ]
        mask_configs = [(n, fn) for n, fn in all_mask_configs if args.mask_type is None or n == args.mask_type]

        for mask_name, mask_fn in mask_configs:
            try:
                startend_row_indices = mask_fn(B, S, H, D)
                if H > 1 and startend_row_indices.shape[1] == 1:
                    startend_row_indices = startend_row_indices.repeat_interleave(H, 1)

                print_buckets_for_sample(
                    startend_row_indices,
                    mask_name,
                    idx,
                    args.cp_size,
                    after_method=args.after_method,
                    balance_chunk_size=args.chunk_size,
                    buckets_per_machine=args.buckets_per_machine,
                    epsilon=args.epsilon,
                    max_swap_iterations=args.max_swap_iterations,
                    max_locality_iterations=args.max_locality_iterations,
                )
            except Exception as e:
                print(f"    ERROR: {mask_name}: {e}")

        count += 1
        if args.max_samples > 0 and count >= args.max_samples:
            break

    print(f"\nDone. Total samples processed: {count}")


if __name__ == "__main__":
    main()
