import argparse
import os
import sys
from functools import partial

import paddle
from tabulate import tabulate

REPO_ROOT = "/root/paddlejob/workspace/env_run/xiehaoyang"
PADDLEFLEET_SRC = os.path.join(REPO_ROOT, "PaddleFleet", "src")
TEST_FLASHMASK_DIR = os.path.join(REPO_ROOT, "test_flashmask")

if PADDLEFLEET_SRC not in sys.path:
    sys.path.insert(0, PADDLEFLEET_SRC)
if TEST_FLASHMASK_DIR not in sys.path:
    sys.path.insert(0, TEST_FLASHMASK_DIR)

from generate_startend_row_indices import (  # noqa: E402
    generate_causal_blockwise_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_empty_mask,
    generate_global_sliding_window_mask,
    generate_none_mask,
    generate_prefix_lm_causal_mask,
    generate_prefix_lm_document_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask,
    generate_share_question_mask,
    generate_sliding_window_mask,
)
from paddlefleet.transformer import sink_impl_attnsink_new, sink_impl_new  # noqa: E402


MASK_FNS = {
    "full": partial(generate_none_mask, causal=False),
    "causal": partial(generate_none_mask, causal=True),
    "sliding": generate_sliding_window_mask,
    "causal_doc": generate_causal_document_mask,
    "doc": generate_document_mask,
    "share_question": generate_share_question_mask,
    "global_sliding": generate_global_sliding_window_mask,
    "causal_blockwise": generate_causal_blockwise_mask,
    "prefix_lm_doc": generate_prefix_lm_document_mask,
    "prefix_lm_causal": generate_prefix_lm_causal_mask,
    "qk_sparse": generate_qk_sparse_mask,
    "random_eviction": generate_random_eviction_mask,
    "empty": generate_empty_mask,
}

IMPLS = {
    "sink_impl_new": sink_impl_new.sink_attention_forward,
    "sink_impl_attnsink_new": sink_impl_attnsink_new.sink_attention_forward,
}


def _summarize(times, mode):
    times = paddle.to_tensor(times, dtype="float32")
    if mode == "mean":
        return paddle.mean(times).item()
    if mode == "median":
        return paddle.median(times).item()
    if mode == "min":
        return paddle.min(times).item()
    raise ValueError(f"Unsupported return mode: {mode}")


def do_bench(fn, warmup=10, repeat=50, return_mode="mean"):
    fn()
    paddle.device.synchronize()

    for _ in range(warmup):
        fn()
    paddle.device.synchronize()

    start_events = [paddle.device.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [paddle.device.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    paddle.device.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return _summarize(times, return_mode)


def clear_grads(*xs):
    for x in xs:
        if x is not None:
            x.grad = None


def clone_inputs(q0, k0, v0, sink0):
    q = q0.detach().clone()
    k = k0.detach().clone()
    v = v0.detach().clone()
    sink = sink0.detach().clone()
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    sink.stop_gradient = False
    return q, k, v, sink


def make_inputs(batch, seqlen_q, seqlen_k, heads, kv_heads, head_dim, dtype):
    q = paddle.randn([batch, seqlen_q, heads, head_dim], dtype=dtype)
    k = paddle.randn([batch, seqlen_k, kv_heads, head_dim], dtype=dtype)
    v = paddle.randn([batch, seqlen_k, kv_heads, head_dim], dtype=dtype)
    sink = paddle.randn([heads], dtype=dtype)
    grad = paddle.randn([batch, seqlen_q, heads, head_dim], dtype=dtype)
    return q, k, v, sink, grad


def bench_impl(
    impl_name,
    impl_fn,
    q0,
    k0,
    v0,
    sink0,
    grad,
    startend_row_indices,
    causal,
    warmup,
    repeat,
    return_mode,
):
    q, k, v, sink = clone_inputs(q0, k0, v0, sink0)

    def forward_only():
        impl_fn(
            q,
            k,
            v,
            sink=sink,
            startend_row_indices=startend_row_indices,
            causal=causal,
        )

    def forward_backward():
        clear_grads(q, k, v, sink)
        out = impl_fn(
            q,
            k,
            v,
            sink=sink,
            startend_row_indices=startend_row_indices,
            causal=causal,
        )
        out.backward(grad)

    fwd_ms = do_bench(forward_only, warmup=warmup, repeat=repeat, return_mode=return_mode)
    fwd_bwd_ms = do_bench(forward_backward, warmup=warmup, repeat=repeat, return_mode=return_mode)
    return {
        "impl": impl_name,
        "fwd_ms": fwd_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "bwd_est_ms": fwd_bwd_ms - fwd_ms,
    }


def compare_outputs(q0, k0, v0, sink0, startend_row_indices, causal):
    outs = {}
    for name, fn in IMPLS.items():
        q, k, v, sink = clone_inputs(q0, k0, v0, sink0)
        outs[name] = fn(
            q,
            k,
            v,
            sink=sink,
            startend_row_indices=startend_row_indices,
            causal=causal,
        )
    base = outs["sink_impl_new"]
    other = outs["sink_impl_attnsink_new"]
    return (base - other).abs().max().item(), (base - other).abs().mean().item()


def run_case(args, mask_name, head_dim):
    mask_fn = MASK_FNS[mask_name]
    startend_row_indices, causal = mask_fn(
        args.batch,
        args.seqlen_q,
        args.seqlen_k,
        args.mask_heads,
    )

    dtype = paddle.bfloat16 if args.dtype == "bf16" else paddle.float16
    paddle.seed(args.seed)
    q0, k0, v0, sink0, grad = make_inputs(
        args.batch,
        args.seqlen_q,
        args.seqlen_k,
        args.heads,
        args.kv_heads,
        head_dim,
        dtype,
    )

    max_diff, mean_diff = compare_outputs(
        q0, k0, v0, sink0, startend_row_indices, causal
    )

    rows = []
    results = []
    for impl_name, impl_fn in IMPLS.items():
        result = bench_impl(
            impl_name,
            impl_fn,
            q0,
            k0,
            v0,
            sink0,
            grad,
            startend_row_indices,
            causal,
            args.warmup,
            args.repeat,
            args.return_mode,
        )
        results.append(result)
        rows.append(
            [
                mask_name,
                head_dim,
                impl_name,
                f"{result['fwd_ms']:.4f}",
                f"{result['bwd_est_ms']:.4f}",
                f"{result['fwd_bwd_ms']:.4f}",
                f"{max_diff:.6g}",
                f"{mean_diff:.6g}",
            ]
        )

    if len(results) == 2:
        old = results[0]
        new = results[1]
        speedup = old["fwd_bwd_ms"] / new["fwd_bwd_ms"]
        rows.append(
            [
                mask_name,
                head_dim,
                "attnsink_new speedup",
                "-",
                "-",
                f"{speedup:.4f}x",
                f"{max_diff:.6g}",
                f"{mean_diff:.6g}",
            ]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FA3 attention-sink wrappers: sink_impl_new vs sink_impl_attnsink_new."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen-q", type=int, default=8192)
    parser.add_argument("--seqlen-k", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--mask-heads", type=int, default=1)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--masks",
        nargs="+",
        default=["causal", "sliding", "causal_doc", "causal_blockwise", "prefix_lm_causal"],
        choices=sorted(MASK_FNS.keys()),
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--return-mode", choices=["mean", "median", "min"], default="mean")
    args = parser.parse_args()

    paddle.set_flags({"FLAGS_flash_attn_version": 3})

    rows = []
    for mask_name in args.masks:
        for head_dim in args.dims:
            rows.extend(run_case(args, mask_name, head_dim))

    headers = [
        "mask",
        "D",
        "impl",
        "fwd ms",
        "bwd est ms",
        "fwd+bwd ms",
        "out max diff",
        "out mean diff",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
