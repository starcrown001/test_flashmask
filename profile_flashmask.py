import paddle
import tqdm
import configargparse
from functools import partial
from generate_startend_row_indices import (
    generate_global_sliding_window_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_causal_blockwise_mask,
    generate_share_question_mask,
    generate_random_eviction_mask,
    generate_qk_sparse_mask,
    generate_prefix_lm_document_mask,
    generate_prefix_lm_causal_mask,
    generate_sliding_window_mask
)
from paddle.nn.functional.flash_attention import flashmask_attention

func_map = {
    "global_swin": generate_global_sliding_window_mask,
    "re": generate_random_eviction_mask,
    "qk_sparse": generate_qk_sparse_mask,
    "prefix_lm_doc": generate_prefix_lm_document_mask,
    "prefix_lm_causal": generate_prefix_lm_causal_mask,
    "swin": generate_sliding_window_mask,
    "causal_doc": generate_causal_document_mask,
    "doc": generate_document_mask,
    "causal_blockwise": generate_causal_blockwise_mask,
    "share_question": generate_share_question_mask,
}

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True, help="Config file path")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seqlen_q", type=int, default=32 * 1024, help="Q sequence length")
    parser.add_argument("--seqlen_k", type=int, default=32 * 1024, help="K sequence length")
    parser.add_argument("--nheads", type=int, default=32, help="Number of heads")
    parser.add_argument("--nheads_startend_row_indices", type=int, default=1, help="Start end row indices")
    parser.add_argument("--d", type=int, default=128, help="Latent dim d")
    parser.add_argument("--dv", type=int, default=128, help="Latent dim dv")
    parser.add_argument("--warmup_times", type=int, default=50, help="Number of times for warmup")
    parser.add_argument("--profile_times", type=int, default=4, help="Number of times for actual profiling")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for attention calculation")
    parser.add_argument("-b", "--backward_prof", default=False, action="store_true", help="Whether to profile backward")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Whether to print runtime config")
    parser.add_argument("-g", "--generic_fav3", default=False, action="store_true", help="Whether to profile generic FA v3")
    parser.add_argument("--causal", default=False, action="store_true", help="Whether to use causal mask")
    parser.add_argument(
        "--mask_type",
        type=str,
        choices=list(func_map.keys()),
        default="global_swin",
        help="Available profiling mask types: global sliding window by default"
    )
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_args()
    batch_size = opts.batch_size
    seqlen_q = opts.seqlen_q
    seqlen_k = opts.seqlen_k
    nheads = opts.nheads
    nheads_startend_row_indices = opts.nheads_startend_row_indices
    d = opts.d
    dv = opts.dv
    dtype = opts.dtype
    warmup_times = opts.warmup_times

    print(f"Mask Type used in profiling: {opts.mask_type}")
    gen_startend_row_indices = partial(func_map[opts.mask_type])

    q = paddle.randn(shape=[batch_size, seqlen_q, nheads, d], dtype=dtype)
    k = paddle.randn(shape=[batch_size, seqlen_k, nheads, d], dtype=dtype)
    v = paddle.randn(shape=[batch_size, seqlen_k, nheads, dv], dtype=dtype)

    if opts.verbose:
        print("FlashAttn profiling configuration:")
        print(opts)

    NO_BACKWARD = not opts.backward_prof
    if NO_BACKWARD:
        print("Backward profiling is disabled.")
    else:
        print("Backward profiling is enabled.")

    q.stop_gradient = NO_BACKWARD
    k.stop_gradient = NO_BACKWARD
    v.stop_gradient = NO_BACKWARD

    if opts.generic_fav3:
        startend_row_indices, causal = (None, opts.causal)
    else:
        startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    paddle.set_flags({'FLAGS_cudnn_deterministic': 0})

    print(f"Warming up run for {warmup_times} time(s)...")
    for i in tqdm.tqdm(range(warmup_times)):
        out, lse = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True
        )
        paddle.device.synchronize()
        print(out.shape)
        if not NO_BACKWARD:
            out.backward()
    print("Warming up completed.")

    print(f"Profiling starts for {opts.profile_times} time(s)...")
    for i in tqdm.tqdm(range(opts.profile_times)):
        paddle.base.core.nvprof_nvtx_push("flashmask")
        out, lse = flashmask_attention(
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True
        )
        if not NO_BACKWARD:
            out.backward()
        paddle.base.core.nvprof_nvtx_pop()
    print("Profiling completed.")
