"""Generate mask info files with format:
Sample N, num_docs M: [(prefix_len, block_len), ...]# [block_bool_0, ...]

Constraints:
  - block_len 之和等于 total_seqlen
  - prefix_len < block_len for each pair
  - block_bool list length equals number of pairs
"""

import random
import argparse


def generate_one_sample(total_seqlen, min_docs, max_docs):
    """Generate one sample satisfying all constraints."""
    num_docs = random.randint(min_docs, max_docs)

    # Step 1: randomly split total_seqlen into num_docs block_len values
    # Use Dirichlet-like approach: generate random proportions, then scale
    raw = [random.random() for _ in range(num_docs)]
    raw_sum = sum(raw)
    block_lens = [max(1, int(r / raw_sum * total_seqlen)) for r in raw]

    # Fix rounding: adjust so that sum(block_lens) == total_seqlen
    diff = total_seqlen - sum(block_lens)
    while diff != 0:
        idx = random.randint(0, num_docs - 1)
        if diff > 0:
            block_lens[idx] += 1
            diff -= 1
        elif block_lens[idx] > 1:
            block_lens[idx] -= 1
            diff += 1

    # Step 2: for each block_len, generate prefix_len < block_len
    # prefix_len should be reasonably smaller; cap at block_len - 1
    pairs = []
    for bl in block_lens:
        # prefix_len in range [1, bl-1), use a reasonable upper bound
        max_prefix = bl - 1
        prefix_len = random.randint(1, max(1, max_prefix))
        pairs.append((prefix_len, bl))

    # Step 3: generate block_bool list (random 0/1)
    block_bools = [random.randint(0, 1) for _ in range(num_docs)]

    return num_docs, pairs, block_bools


def format_sample(sample_idx, num_docs, pairs, block_bools):
    """Format one sample line."""
    pairs_str = ", ".join(f"({a}, {b})" for a, b in pairs)
    bools_str = ", ".join(str(b) for b in block_bools)
    return f"Sample {sample_idx}, num_docs {num_docs}: [{pairs_str}]# [{bools_str}]"


def main():
    parser = argparse.ArgumentParser(description="Generate mask seq info files")
    parser.add_argument("--total_seqlen", type=int, default=524288)
    parser.add_argument("--min_docs", type=int, default=14)
    parser.add_argument("--max_docs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path. If not set, prints to stdout.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    lines = []
    lines.append(f"Total length: {args.total_seqlen}, Document count range: ({args.min_docs}, {args.max_docs})")

    for i in range(1, args.num_samples + 1):
        num_docs, pairs, block_bools = generate_one_sample(
            args.total_seqlen, args.min_docs, args.max_docs
        )
        lines.append(format_sample(i, num_docs, pairs, block_bools))

    output = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
