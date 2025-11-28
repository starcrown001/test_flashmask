import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import time
import paddle
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flashmask_attention
from test_util import attention_ref , blockmask_to_densemask, random_blockmask, flashmask_to_densemask


# import flash_attn_interface
import torch

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

def from_paddle(x: paddle.Tensor):
    if x.dtype == paddle.bfloat16 or x.dtype == "bfloat16":
      return torch.from_numpy(x.view("uint16").numpy()).to("cuda").view(torch.bfloat16)
    elif x.dtype == paddle.float32 or x.dtype == "float32":
      return torch.from_numpy(x.numpy()).to("cuda")
    else:
      assert False
      
def from_torch(x: torch.Tensor):
    if x.dtype == torch.bfloat16 or x.dtype == "bfloat16":
      return paddle.to_tensor(x.view(torch.uint16).numpy()).to("gpu").view("bfloat16")
    elif x.dtype == torch.float32 or x.dtype == "float32":
      return paddle.to_tensor(x.numpy()).to("gpu")
    else:
      assert False
      
def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

def do_bench(fn, warmup=1, rep=1, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
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
    paddle.base.core.nvprof_nvtx_push("paddle")

    fn()

    paddle.device.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = paddle.empty([int(cache_size // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(cache_size)], dtype=paddle.int8)

    # Estimate the runtime of the function
    start_event = paddle.device.Event(enable_timing=True)
    end_event = paddle.device.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    paddle.device.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        #cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    paddle.device.synchronize()
    times = paddle.to_tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=paddle.float32)
    paddle.base.core.nvprof_nvtx_pop()
    return _summarize_statistics(times, quantiles, return_mode)
      
def attention_naive_with_mask_varlen(q, k, v, attn_bias,cu_seqlens_k):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    
    qs = []
    ks = []
    vs = []
    attns = []
    
    for i in range(1,len(cu_seqlens_k)):
        qs.append(qt[:,:, cu_seqlens_k[i-1]:cu_seqlens_k[i], :])
        ks.append(kt[:,:, cu_seqlens_k[i-1]:cu_seqlens_k[i], :])
        vs.append(vt[:,:, cu_seqlens_k[i-1]:cu_seqlens_k[i], :])
        attns.append(attn_bias[:,:, cu_seqlens_k[i-1]:cu_seqlens_k[i],cu_seqlens_k[i-1]:cu_seqlens_k[i]])
    
    os = []    
    for i in range(len(qs)):
        scale = 1.0 / np.sqrt(qs[i].shape[-1])
        s = paddle.matmul(qs[i], paddle.transpose(ks[i], [0, 1, 3, 2]))
        s = paddle.scale(s, scale)
        p = F.softmax(s + attns[i].cast(s.dtype))
        o = paddle.matmul(p, vs[i])
        os.append(o)
    # print(f"len_os:{len(os)}")
    o = paddle.concat(os, axis=2)
    return paddle.transpose(o, [0, 2, 1, 3])

def get_dv(q, k, v, attn_bias,o_grad ,lse):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    o_gradt = paddle.transpose(o_grad, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(qt.shape[-1])
    s = paddle.matmul(qt,paddle.transpose(kt, [0,1, 3, 2]))
    s = paddle.scale(s, scale)
    s += attn_bias
    s -= lse.unsqueeze(-1)
    p = F.softmax(s, axis = -1)
    dv = paddle.matmul(p.transpose([0, 1, 3, 2]), o_gradt.cast("float32")) 
    return paddle.transpose(dv, [0, 2, 1, 3])
    


def test_mask(
    generate_mask_fn,
    # cu_seqlens_q: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
    # cu_seqlens_k: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
    cu_seqlens_q: list = [0, 63, 128],
    cu_seqlens_k: list = [0, 63, 128],
    batch_size: int = 1,
    num_head: int = 1,
    head_size: int = 64,
    max_seqlen_q: int = 65,
    max_seqlen_k: int = 65,
    sliding_window_size: int = -1,
):
    # paddle.seed(2024)
    paddle.seed(2025)
    # batch_size = 8
    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]
    query = paddle.ones([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16) 
    key = paddle.ones([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    o_grad = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
    
    
    # query = paddle.ones([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
    # key = paddle.ones([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    # value = paddle.ones([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    # o_grad = paddle.ones([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)

    # ref_query = query.reshape([total_q, num_head, head_size])
    # ref_key = key.reshape([total_k, num_head, head_size])
    # ref_value = value.reshape([total_k, num_head, head_size])

    ref_query = from_paddle(query.reshape([total_q * batch_size, num_head, head_size]))
    ref_key = from_paddle(key.reshape([total_k * batch_size, num_head, head_size]))
    ref_value = from_paddle(value.reshape([total_k* batch_size, num_head, head_size]))
    ref_o_grad = from_paddle(o_grad.reshape([total_q* batch_size, num_head, head_size]))
    # ref_query = from_paddle(query)
    # ref_key = from_paddle(key)
    # ref_value = from_paddle(value)
    # ref_o_grad = from_paddle(o_grad)
    

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False
    
    ref_query.requires_grad = True
    ref_key.requires_grad = True
    ref_value.requires_grad = True

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        startend_row_indices, causal = generate_mask_fn(batch_size, total_q, num_head, head_size)
        # startend_row_indices, causal = generate_mask_fn(total_q)
        
    # empty_tensor = paddle.empty(shape=[0])

    # flashmask_out, flashmask_lse = flashmask_attention(query, key, value, startend_row_indices=startend_row_indices, causal=causal, return_softmax_lse=True)
    # flashmask_out, flashmask_lse = flashmask_attention(query, key, value,startend_row_indices=empty_tensor,causal=causal, return_softmax_lse=True)
    # paddle.device.synchronize()
    # paddle.device.synchronize()
    if(sliding_window_size == -1):
        ref_out, ref_lse = flash_attn_interface.flash_attn_varlen_func(ref_query, ref_key, ref_value, torch.tensor(cu_seqlens_q, device="cuda", dtype=torch.int32), torch.tensor(cu_seqlens_k, device="cuda", dtype=torch.int32), max_seqlen_q, max_seqlen_k, causal=causal)    
    else:
        ref_out, ref_lse = flash_attn_interface.flash_attn_varlen_func(ref_query, ref_key, ref_value, torch.tensor(cu_seqlens_q, device="cuda", dtype=torch.int32), torch.tensor(cu_seqlens_k, device="cuda", dtype=torch.int32), max_seqlen_q, max_seqlen_k, causal=causal,window_size=(0,sliding_window_size))
    ref_out.backward(ref_o_grad)
    
    out = from_torch(ref_out.detach().cpu()).reshape([batch_size, total_q, num_head, head_size])
    lse = from_torch(ref_lse.detach().cpu())
    
    empty_tensor = paddle.zeros(shape=[total_q,num_head],dtype=paddle.int64)
    
    dq1,dk1,dv1 = paddle._C_ops.flashmask_attention_grad(query, key, value, startend_row_indices, out, lse, empty_tensor, o_grad,0.0,causal)
    
    
    paddle.device.synchronize()
    dq, dk, dv = flashmask_attn_bwd(
        query,
        key,
        value,
        out,
        lse,
        startend_row_indices,
        o_grad,
        query.shape[-1] ** (-0.5),
        causal
    )
    
    paddle.device.synchronize()
    # for x,y in [(dk, from_torch(ref_key.grad.detach().cpu())),(dv, from_torch(ref_value.grad.detach().cpu())),(dq, from_torch(ref_query.grad.detach().cpu()))]:
    #     strict_check(x.flatten(), y.flatten())
    
    for x,y in [(dk, dk1),(dv, dv1),(dq, dq1)]:
        strict_check(x.flatten(), y.flatten())
    
def test_mask_with_v1(
    generate_mask_fn,
    # cu_seqlens_q: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
    # cu_seqlens_k: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
    cu_seqlens_q: list = [0, 63, 128],
    cu_seqlens_k: list = [0, 63, 128],
    batch_size: int = 1,
    num_head: int = 1,
    head_size: int = 64,
    max_seqlen_q: int = 65,
    max_seqlen_k: int = 65
):
    # paddle.seed(2024)
    paddle.seed(2024)
    # batch_size = 1
    total_q = cu_seqlens_q[-1] 
    total_k = cu_seqlens_k[-1] 
    query = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16) 
    key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    o_grad = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
    empty_tensor = paddle.empty(shape=[0],dtype="float32")
    print(query.shape)
    
    # tiles = [1, 1, num_head, 1]  # 在倒数第二个维度上重复 num_head 次
    # query = paddle.tile(query, tiles)
    # key = paddle.tile(key, tiles)
    # value = paddle.tile(value, tiles)
    # o_grad = paddle.tile(o_grad, tiles)
    

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        print("enter",generate_mask_fn)
        startend_row_indices, causal = generate_mask_fn(batch_size, total_k, num_head, head_size)
        if(total_q < total_k):
            startend_row_indices = startend_row_indices.clip(max=total_q)
        # startend_row_indices, causal = generate_mask_fn(total_q)
        
    # print(startend_row_indices)
    # paddle.set_printoptions(precision=None, threshold=10000000, edgeitems=None, sci_mode=None, linewidth=None)
    paddle.set_flags({'FLAGS_flash_attn_version': 2})
    paddle.device.synchronize()
    out,lse = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse = True
        )
    out.backward(o_grad)
    paddle.device.synchronize()
    # fa1_bwd = lambda:out.backward(o_grad,retain_graph=True)
    paddle.device.synchronize()
    # print("pass")
    
    query1 = query.detach().clone()
    key1 = key.detach().clone()
    value1 = value.detach().clone()
    # out1 = out.detach().clone()
    # lse1 = lse.detach().clone()
    startend_row_indices1 = startend_row_indices.detach().clone()
    ograd1 = o_grad.detach().clone()
    
    query1.stop_gradient = False
    key1.stop_gradient = False
    value1.stop_gradient = False
    
    paddle.device.synchronize()
    # print("pypt1:")
    # print(startend_row_indices)
    start_time = time.time()
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    paddle.device.synchronize()
    
    # startend_row_indices1 = paddle.load("startend_row_indices.pd")
    

    blockmask = random_blockmask(
        shape=[
            startend_row_indices1.shape[0],
            startend_row_indices1.shape[1],
            startend_row_indices1.shape[2] // 128,
            startend_row_indices1.shape[2] // 128
        ],
        dtype=paddle.int32
    )
    # blockmask = paddle.load("blockmask.pd")
    # blockmask = paddle.tensor(       [[[[1, 0, 0],
    #       [0, 1,0],
    #       [1, 0, 1]]]],dtype=paddle.int32)
    paddle.base.core.nvprof_nvtx_push("paddle")
    print(1)
    (out1,lse1) = flashmask_attention(
            query1,
            key1,
            value1,
            startend_row_indices=startend_row_indices1,
            causal=causal,
            return_softmax_lse = True,
            block_mask=blockmask)
    paddle.device.synchronize()
    print(2)
    out1.backward(ograd1)
    print(3)
    paddle.device.synchronize()
    
    
    # print("pypt2:")
    # print(startend_row_indices)
    paddle.device.synchronize()
    paddle.base.core.nvprof_nvtx_pop()
    # for x,y in [(key1.grad, key.grad),(value1.grad, value.grad),(query1.grad, query.grad)]:
    #     strict_check(x.flatten(), y.flatten())
    # for x,y in [(out1,out),(lse1,lse)]:
    #     strict_check(x.flatten(), y.flatten())
    # print("pass")
    
    q1 = query.detach()
    k1 = key.detach()
    v1 = value.detach()
    q1.stop_gradient = False
    k1.stop_gradient = False
    v1.stop_gradient = False
    
    # print(blockmask)
    print(startend_row_indices1)
    mask_flash = flashmask_to_densemask(startend_row_indices1,total_q, num_head, causal)
    print(mask_flash)
    # print(blockmask)
    mask_block = blockmask_to_densemask(blockmask,q1.shape[1],k1.shape[1],paddle.int32,causal)
    # print(mask_block)
    mask_inf = mask_flash & mask_block
    # print(mask_inf)
    mask = paddle.zeros((batch_size, num_head, total_q, total_k), dtype=paddle.bfloat16)
    mask = paddle.where(mask_inf, paddle.zeros_like(mask), paddle.full_like(mask, float('-inf')))    
    paddle.set_printoptions(precision=None, threshold=10000000, edgeitems=None, sci_mode=None, linewidth=None)
    # print(mask)
    # attn_bias = paddle.load("attn_bias.pd")
    # assert paddle.equal_all(attn_bias.astype(paddle.float32),mask.astype(paddle.float32))
    ref_out1, attn_ref = attention_ref(
        q1,
        k1,
        v1,
        causal=causal,
        attn_bias=mask
    )
    # ref_out1 = attention_naive_with_mask_varlen(q1, k1, v1, mask,[0,q1.shape[1]])
    # ref_out1 = paddle.where(paddle.isnan(ref_out1), paddle.zeros_like(ref_out1), ref_out1)
    paddle.device.synchronize()
    # strict_check(ref_out1.flatten(), out1.flatten())
    ref_out1.backward(o_grad)
    print(q1.grad[0,0,:,0])
    
    # paddle.device.synchronize()
    for x,y in [(key1.grad,k1.grad),(value1.grad,v1.grad),(query1.grad,q1.grad)]:
        strict_check(x.flatten(), y.flatten())
    # print("---------q-----------")
    # print(query[:,:,:,0])
    # print("---------k-----------")
    # print(key[:,:,:,0])
    # print("---------v-----------")
    # print(value[:,:,:,0])
    # print("---------lse_v1-----------")
    # print(lse)
    # print("---------dv_v1-----------")
    # print(value.grad[:,:,:,0])
    # print("---------dv_v2-----------")
    # print(dv[:,:,:,0])
    # print("---------dv_orin-----------")
    # dv_orin = get_dv(query[:,:,:,0:1], key[:,:,:,0:1],value[:,:,:,0:1],flashmask_to_densemask(startend_row_indices,paddle.bfloat16, causal),o_grad[:,:,:,0:1],lse)
    # print(dv_orin[:,:,:,0])

def strict_check(x, y):
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bfloat16 or x.dtype == "float16":
          # x = x.view("float16").numpy()
          x = x.cast("float32").numpy()
        else:
          x = x.numpy()
    else:
      assert False

    # if isinstance(y, torch.Tensor):
    #     if y.dtype == torch.bfloat16 or y.dtype == "bfloat16":
    #       # x = x.view("float16").numpy()
    #       y = y.to(torch.float32).detach().cpu().numpy()
    #     else:
    #       y = y.detach().cpu().numpy()

    if isinstance(y, paddle.Tensor):
        if y.dtype == paddle.bfloat16 or y.dtype == "float16":
          # y = y.view("float16").numpy()
          y = y.cast("float32").numpy()
        else:
          y = y.numpy()

    try:
        print(f"{x=}, {y=}")
        np.testing.assert_allclose(x.flatten(), y.flatten(),rtol=1e-2, atol=1e-1)
    except Exception as e:
        print('---------------')
        idx = np.where(~(x == y))
        print(f"fail idx: {idx=}")
        print(f"shape:'{x.shape}'")
        # print(f"fail idx:'{np.unique(idx[0])}'")
        print(x[idx])
        print(y[idx])
        raise e
    

def ele_check(x, y):
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bfloat16 or x.dtype == "bfloat16":
          # x = x.view("uint16").numpy()
          x = x.cast("float32").numpy()
        else:
          x = x.numpy()
    else:
      assert False

    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16 or y.dtype == "bfloat16":
          # x = x.view("uint16").numpy()
          y = y.to(torch.float32).detach().cpu().numpy()
        else:
          y = y.detach().cpu().numpy()

    # if isinstance(y, paddle.Tensor):
    #     if y.dtype == paddle.bfloat16 or y.dtype == "bfloat16":
    #       # y = y.view("uint16").numpy()
    #       y = y.cast("float32").numpy()
    #     else:
    #       y = y.numpy()

    try:
        print(f"{x=}, {y=}")
        np.testing.assert_allclose(np.sort(x.flatten()), np.sort(y.flatten()),rtol=1e-3, atol=1e-6)
    except Exception as e:
        print('---------------')
        idx = np.where(~(x == y))
        print(f"fail idx: {idx=}")
        print(f"shape:'{x.shape}'")
        # print(f"fail idx:'{np.unique(idx[0])}'")
        print(x[idx])
        print(y[idx])
        raise e

# def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
#     if startend_row_indices is None:
#         return None
#     bz, num_head, seq_len, bound_num = startend_row_indices.shape
#     m = paddle.ones((bz, num_head, seq_len, seq_len), dtype=bool)
#     has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
#     for bi in range(bz):
#         for hi in range(num_head):
#             for j in range(seq_len):
#                 downstart = startend_row_indices[bi, hi, j, 0]
#                 if has_end:
#                     downend = startend_row_indices[bi, hi, j, 1]
#                     m[bi, hi, downstart:downend, j] = False
#                 else:
#                     m[bi, hi, downstart:, j] = False
#                 if causal:
#                     m[bi, hi, :j, j] = False
#                 else:
#                     if has_end:
#                         upstart = startend_row_indices[bi, hi, j, 2]
#                         upend = startend_row_indices[bi, hi, j, 3]
#                         m[bi, hi, upstart:upend, j] = False
#                     else:
#                         upend = startend_row_indices[bi, hi, j, 1]
#                         m[bi, hi, :upend, j] = False
#     return m

# def random_blockmask_with_at_least_one(shape, dtype='int32'):
#     """
#     生成一个随机 0/1 blockmask，每一行（最后一维）至少有一个1。

#     Args:
#         shape: list/tuple，形如 [B, S, Q, K]
#         dtype: 输出类型，通常为 'int32'

#     Returns:
#         mask: paddle.Tensor, 0/1，满足每行至少一个1
#     """
#     # 先随机生成 0/1 mask
#     mask = paddle.randint(0, 2, shape, dtype=dtype)
#     # 找出每行（最后一维）全为0的位置
#     row_sum = paddle.sum(mask, axis=-1)
#     need_fix = (row_sum == 0)  # [B, S, Q]

#     if paddle.any(need_fix):
#         # 找到需要修正的所有行的 index
#         idx = paddle.nonzero(need_fix)
#         for b, s, q in idx.numpy():
#             # 随机选一个列位置，置为1
#             k = paddle.randint(0, shape[-1], [1]).item()
#             mask[b, s, q, k] = 1
#     return mask

# def blockmask_to_densemask(blockmask, q_len, k_len, dtype, causal=True):
#     """
#     Args:
#         blockmask: [b, s, q_blocks, k_blocks]  (0/1 mask, 1表示masked, 0表示可见)
#         q_len: int, query序列长度
#         k_len: int, key序列长度
#         dtype: paddle.float32等
#         causal: bool, 是否加自回归遮挡

#     Returns:
#         densemask: [b, s, q_len, k_len]，可直接用于attention
#     """
#     if blockmask is None:
#         return None
#     bz, num_head, q_blocks, k_blocks = blockmask.shape
#     block_q = (q_len + q_blocks - 1) // q_blocks
#     block_k = (k_len + k_blocks - 1) // k_blocks

#     # 1. 展开到[bs, s, q_len, k_len]
#     densemask = blockmask.astype(dtype).repeat_interleave(block_q, axis=2).repeat_interleave(block_k, axis=3)
#     densemask = densemask[:, :, :q_len, :k_len]

#     # 2. 构造 causal mask（上三角全为1，下三角为0），形状 [q_len, k_len]
#     if causal:
#         causal_mask = paddle.triu(paddle.ones((q_len, k_len), dtype='bool'), 1)
#         # 合并两类mask，1表示被mask
#         densemask = paddle.where(causal_mask, paddle.ones_like(densemask), densemask)

#     # densemask = paddle.where(densemask == 0, paddle.full_like(densemask, float('-inf')), paddle.zeros_like(densemask))

#     return densemask.astype(paddle.bool)

def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_ones_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 2), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = False
    return startend_row_indices, causal

def generate_causal_mask(B,S,H,D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 1), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = True
    return startend_row_indices, causal

def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)

    causal=True
    return startend_row_indices, causal

# def generate_causal_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
def generate_causal_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S, f"{total_seq_len=}, {S=}"
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1))
    startend_row_indices = startend_row_indices.repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_share_question_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - total_seq_len

    startend_row_indices = [S] * doc_seq_lens[0]

    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
        
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_global_sliding_window_mask(B, S, H, D, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start_row_indices[:global_token] = S
    down_left_start_row_indices =  down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_start_row_indices = paddle.full([S], global_token, dtype="int32")
    up_right_start_row_indices[:global_token+right_window_size+1] = 0
    up_right_start_row_indices = up_right_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_end_row_indices = paddle.arange(
        -right_window_size, S - right_window_size, dtype="int32"
    )
    up_right_end_row_indices[:global_token+right_window_size+1] = 0
    up_right_end_row_indices = up_right_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_start_row_indices, down_left_end_row_indices, up_right_start_row_indices, up_right_end_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seq_lens[-1] + [S] * padding
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_prefix_lm_document_mask(B, S, H, D, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    """
    tuple(prefix_length, seq_length)
    """
    assert len(doc_seq_lens) >= 2
    total_seq_len = 0
    for prefix_length, seq_length in doc_seq_lens:
        total_seq_len += seq_length
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_causal_mask(B, S, H, D, prefix_length=1024):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S
    down_left_row_indices = paddle.full([S], S, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, S)), dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
    """
    tuple(offset, maskout_len)
    """
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset > last_offset
        start_row_indices.extend([S]*(offset-last_offset))
        end_row_indices.extend([S]*(offset-last_offset))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend([S]*(S-last_offset))
    end_row_indices.extend([S]*(S-last_offset))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

#def generate_hash_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
#    """
#    tuple(offset, maskout_len)
#    """
#    start_row_indices = []
#    end_row_indices  = []
#    last_offset = 0
#    for offset, maskout_len in maskout_pair:
#        assert offset > last_offset
#        start_row_indices.append([S]*(offset-last_offset))
#        end_row_indices.append([S]*(offset-last_offset))
#
#        start_row_indices.append(list(range(offset, offset+maskout_len)))
#        end_row_indices.append([offset+maskout_len]*(maskout_len))
#
#        last_offset = offset + maskout_len
#
#    last_offset <= S
#    start_row_indices.append([S]*(S-last_offset))
#    end_row_indices.append([S]*(S-last_offset))
#
#    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
#    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
#    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
#
#    causal = False
#    return startend_row_indices, causal


def generate_random_eviction_mask(B, S, H, D, start_row=4096):
    np.random.seed(0)
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            min_index = np.arange(1,S+1)
            start_rows = np.maximum(start_rows, min_index)
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    causal = True
    return startend_row_indices, causal

def main(examples: List[str] = ["all"]):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    head_size = 128
    params_list = [
        # [{
        #     # "Causal Document Mask":{'doc_seq_lens': list([640 for _ in range(64)])}, #sucess
        #     "Document Mask": {'doc_seq_lens': list([640 for _ in range(64)])}, #fail
        #     # "Global Sliding Window": {'global_token': 16, 'window_size': (32, 32)},#fail
        #     # "Sliding Window": {'window_size': 32}#fail
        # },
        # {
        #     'cu_seqlens_q': list([640*i for i in range(65)]),
        #     'cu_seqlens_k': list([640*i for i in range(65)]),
        #     'num_head': int(1),
        #     'head_size': int(128),
        #     'max_seqlen_q': int(640),
        #     'max_seqlen_k': int(640)
        # }],
        # [{
        #     # "Sliding Window": {'window_size': 32},
        #     # "Full": {'causal': False},
        #     # "Ones": {},
        #     # "Causal": {},
        #     # "Causal Document Mask": {'doc_seq_lens': list([127, 233-127,333-233])},
        #     "Document Mask": {'doc_seq_lens': list([64, 128-64])}
        # },
        # {
        #     'cu_seqlens_q': list([0,64,128]),
        #     'cu_seqlens_k': list([0,64,128]),
        #     'num_head': int(1),
        #     'head_size': int(64),
        #     'max_seqlen_q': int(64),
        #     'max_seqlen_k': int(64),
        #     'sliding_window_size': int(-1)
        # }],
        # [{
        #     "Sliding Window": {'window_size': 64},
        #     "Global Sliding Window": {'global_token': 1, 'window_size': (32, 32)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Document Mask": {'doc_seq_lens': list([2, 63, 63])},
        #     "Share Question Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Causal Blockwise Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Prefix LM Document Mask": {'doc_seq_lens': list([(1, 127), (35, 133), (50, 124)])},
        #     "Prefix LM Causal Mask": {'prefix_length': 1},
        #     "QK-sparse Mask": {'maskout_pair': list([(1, 34), (63, 33)])},
        #     "Random Eviction Mask": {"start_row": 32},
        # },
        # {
        #     'cu_seqlens_q': list([0,128]),
        #     'cu_seqlens_k': list([0,128]),
        #     'num_head': int(1),
        #     'head_size': int(128),
        #     'batch_size': int(1),
        #     'max_seqlen_q': int(128),
        #     'max_seqlen_k': int(128),
        # }],
        # [{
        #     "Sliding Window": {'window_size': 64},
        #     "Global Sliding Window": {'global_token': 1, 'window_size': (32, 32)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Document Mask": {'doc_seq_lens': list([2, 133, 124])},
        #     "Share Question Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Causal Blockwise Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Prefix LM Document Mask": {'doc_seq_lens': list([(1, 127), (35, 133), (50, 124)])},
        #     "Prefix LM Causal Mask": {'prefix_length': 1},
        #     "QK-sparse Mask": {'maskout_pair': list([(1, 34), (63, 33)])},
        #     "Random Eviction Mask": {"start_row": 32},
        # },
        # {
        #     'cu_seqlens_q': list([0,128 * 2]),
        #     'cu_seqlens_k': list([0,128 * 2]),
        #     'num_head': int(1),
        #     'head_size': int(128),
        #     'batch_size': int(1),
        #     'max_seqlen_q': int(384),
        #     'max_seqlen_k': int(384), 
        # }],
        # [{
        #     "Sliding Window": {'window_size': 32},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (32, 32)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': list([63, 32,33])},
        #     "Document Mask": {'doc_seq_lens': list([63, 32,33])},
        #     "Share Question Mask": {'doc_seq_lens': list([63, 32,33])},
        #     "Causal Blockwise Mask": {'doc_seq_lens': list([63, 32,33])},
        #     "Prefix LM Document Mask": {'doc_seq_lens': list([(1, 63), (16, 32), (10, 33)])},
        #     "Prefix LM Causal Mask": {'prefix_length': 63},
        #     "QK-sparse Mask": {'maskout_pair': list([(1, 34), (63, 33)])},
        #     "Random Eviction Mask": {"start_row": 32},
        # },
        # {
        #     'cu_seqlens_q': list([0,128]),
        #     'cu_seqlens_k': list([0,128]),
        #     'num_head': int(8),
        #     'head_size': int(head_size),
        #     'batch_size': int(8),
        #     'max_seqlen_q': int(128),
        #     'max_seqlen_k': int(128),
        # }],
        # [{
        #     "Sliding Window": {'window_size': 512},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': list([2538, 1742,3213])},
        #     "Document Mask": {'doc_seq_lens': list([2538, 1742,3213])},
        #     "Share Question Mask": {'doc_seq_lens': list([2538, 1742,3213])},
        #     "Causal Blockwise Mask": {'doc_seq_lens': list([2538, 1742,3213])},
        #     "Prefix LM Document Mask": {'doc_seq_lens': list([(1024, 2538), (1742, 1742), (512, 3213)])},
        #     "Prefix LM Causal Mask": {'prefix_length': 1024},
        #     "QK-sparse Mask": {'maskout_pair': list([(1024, 538), (2333, 1700)])},
        #     "Random Eviction Mask": {"start_row": 4096},
        # },
        # {
        #     'cu_seqlens_q': list([0,2538,4280,7493]),
        #     'cu_seqlens_k': list([0,2538,4280,7493]),
        #     'num_head': int(8),
        #     'head_size': int(head_size),
        #     'batch_size': int(1),
        #     'max_seqlen_q': int(3213),
        #     'max_seqlen_k': int(3213),
        # }],
        [{
            "Sliding Window": {'window_size': 512},
            "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
            "Ones": {},
            "Causal": {},
            "Causal Document Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Document Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Share Question Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Causal Blockwise Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(1024, 2538), (1742, 1742), (1146, 3912)]},
            "Prefix LM Causal Mask": {'prefix_length': 1024},
            "QK-sparse Mask": {'maskout_pair': [(1024, 538), (2333, 1700)]},
            "Random Eviction Mask": {"start_row": 4096},
        },
        {
            'cu_seqlens_q': [0, 2538, 4278, 8192],  # 0, 2538, 2538+1742=4278, 4278+3912=8190
            'cu_seqlens_k': [0, 2538, 4278, 8192],
            'num_head': 1,
            'head_size': head_size,
            'batch_size': 1,
            'max_seqlen_q': 3912,
            'max_seqlen_k': 3912,
        }],
        # [{
        #     "Sliding Window": {'window_size': 512},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': [4096, 6100, 6188]},
        #     "Document Mask": {'doc_seq_lens': [4096, 6100, 6188]},
        #     "Share Question Mask": {'doc_seq_lens': [4096, 6100, 6188]},
        #     "Causal Blockwise Mask": {'doc_seq_lens': [4096, 6100, 6188]},
        #     "Prefix LM Document Mask": {'doc_seq_lens': [(1024, 4096), (2048, 6100), (2048, 6188)]},
        #     "Prefix LM Causal Mask": {'prefix_length': 1025},
        #     "QK-sparse Mask": {'maskout_pair': [(4096, 2048), (8192, 1024)]},
        #     "Random Eviction Mask": {"start_row": 8191},
        # },
        # {
        #     'cu_seqlens_q': [0, 4096, 10240, 16384],  # 0, 4096, 4096+6144=10240, 10240+6144=16384
        #     'cu_seqlens_k': [0, 4096, 10240, 16384],
        #     'num_head': 8,
        #     'head_size': head_size,
        #     'batch_size': 1,
        #     'max_seqlen_q': 6144,
        #     'max_seqlen_k': 6144,
        # }],
        # [{
        #     "Sliding Window": {'window_size': 512},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': [8192, 12293, 12283]},
        #     "Document Mask": {'doc_seq_lens': [1024 for  _ in range(33)]},
        #     "Share Question Mask": {'doc_seq_lens': [8192, 12293, 12283]},
        #     "Causal Blockwise Mask": {'doc_seq_lens': [8192, 12293, 12283]},
        #     "Prefix LM Document Mask": {'doc_seq_lens': [(4096, 8192), (4096, 12293), (4096, 12283)]},
        #     "Prefix LM Causal Mask": {'prefix_length': 4096},
        #     "QK-sparse Mask": {'maskout_pair': [(8192, 4096), (24576, 4096)]},
        #     "Random Eviction Mask": {"start_row": 3333},
        # },
        # {
        #     'cu_seqlens_q': [0, 8192, 20480, 32768+1024],  # 0, 8192, 8192+12288=20480, 20480+12288=32768
        #     'cu_seqlens_k': [0, 8192, 20480, 32768+1024],
        #     'num_head': 4,
        #     'head_size': head_size,
        #     'batch_size': 4,
        #     'max_seqlen_q': 12288,
        #     'max_seqlen_k': 12288,
        # }],
        # [{
        #     "Sliding Window": {'window_size': 512},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': [20480, 16384, 28672]},    # 总和65536
        #     "Document Mask": {'doc_seq_lens': [20480, 16384, 28672]},
        #     "Share Question Mask": {'doc_seq_lens': [20480, 16384, 28672]},
        #     "Causal Blockwise Mask": {'doc_seq_lens': [20480, 16384, 28672]},
        #     "Prefix LM Document Mask": {'doc_seq_lens': [(10240, 20480), (8192, 16384), (14336, 28672)]},
        #     "Prefix LM Causal Mask": {'prefix_length': 10240},
        #     "QK-sparse Mask": {'maskout_pair': [(16384, 8192), (49152, 8192)]},
        #     "Random Eviction Mask": {"start_row": 20480},
        # },
        # {
        #     'cu_seqlens_q': [0, 20480, 36864, 65536],   # [0, 20480, 20480+16384=36864, 36864+28672=65536]
        #     'cu_seqlens_k': [0, 20480, 36864, 65536],
        #     'num_head': 8,
        #     'head_size': head_size,
        #     'batch_size': 1,
        #     'max_seqlen_q': 28672,      # 3个doc最大长度
        #     'max_seqlen_k': 28672,
        # }],
        # [{
        #     "Sliding Window": {'window_size': 512},
        #     "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
        #     "Ones": {},
        #     "Causal": {},
        #     "Causal Document Mask": {'doc_seq_lens': [65213, 33091, 32768]},
        #     "Document Mask": {'doc_seq_lens': [65213, 33091, 32768]},
        #     "Share Question Mask": {'doc_seq_lens': [65213, 33091, 32768]},
        #     "Causal Blockwise Mask": {'doc_seq_lens': [65213, 33091, 32768]},
        #     "Prefix LM Document Mask": {'doc_seq_lens': [(32768, 65213), (16384, 33091), (16384, 32768)]},
        #     "Prefix LM Causal Mask": {'prefix_length': 65213},
        #     "QK-sparse Mask": {'maskout_pair': [(2, 32768), (64433, 32768)]},
        #     "Random Eviction Mask": {"start_row": 65213},
        # },
        # {
        #     'cu_seqlens_q': [0, 65213, 98304, 131072],  # 0, 65536, 65536+32768=98304, 98304+32768=131072
        #     'cu_seqlens_k': [0, 65213, 98304, 131072],
        #     'num_head': 32,
        #     'head_size': head_size,
        #     'batch_size': 1,
        #     'max_seqlen_q': 65213,  # Should be the maximum in doc_seq_lens
        #     'max_seqlen_k': 65213,
        # }]
    ]

    available_examples = {
        # "Ones": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_ones_mask, **params0), **params1),
        # "Causal": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_causal_mask, **params0), **params1),
        # "Sliding Window": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_sliding_window_mask, **params0), **params1),
        # "Causal Document Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_causal_document_mask, **params0), **params1),
        # "Document Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_document_mask, **params0), **params1),
        "Share Question Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_share_question_mask,**params0), **params1),
        # "Global Sliding Window": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_global_sliding_window_mask,**params0), **params1),
        # "Causal Blockwise Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_causal_blockwise_mask,**params0), **params1),
        # "Prefix LM Causal Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_prefix_lm_causal_mask, **params0), **params1),
        # "Prefix LM Document Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_prefix_lm_document_mask, **params0), **params1),
        # "QK-sparse Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_qk_sparse_mask, **params0), **params1),
        # "Random Eviction Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_random_eviction_mask, **params0), **params1),
    }
    

    # available_examples = {
    #     "Ones": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_ones_mask, **params0), **params1),
    #     "Causal": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_causal_mask, **params0), **params1),
    #     "Sliding Window": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_sliding_window_mask, **params0), **params1),
    #     "Causal Document Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_causal_document_mask, **params0), **params1),
    #     "Document Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_document_mask, **params0), **params1),
    #     "Share Question Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_share_question_mask,**params0), **params1),
    #     "Global Sliding Window": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_global_sliding_window_mask,**params0), **params1),
    #     "Causal Blockwise Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_causal_blockwise_mask,**params0), **params1),
    #     "Prefix LM Causal Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, **params0), **params1),
    #     "QK-sparse Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_qk_sparse_mask, **params0), **params1),
    #     "Random Eviction Mask": lambda params0,params1: test_mask(generate_mask_fn=partial(generate_random_eviction_mask, **params0), **params1),
    # }
    
    # test_naive_examples = {
    #     "Sliding Window": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_sliding_window_mask, **params0), **params1),
    #     "Causal Document Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_causal_document_mask, **params0), **params1),
    #     "Document Mask": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_document_mask, **params0), **params1),
    #     "Global Sliding Window": lambda params0,params1: test_mask_with_v1(generate_mask_fn=partial(generate_global_sliding_window_mask,**params0), **params1),
    #  }

    if "all" in examples:
        ex_to_run = list(available_examples.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in available_examples:
            for params in params_list:
                print(f"Running {ex}\n")
                with open("execution_times_fwd_128_1.txt", "a") as log_file:
                    log_file.write(f"{ex}: ")
                available_examples[ex](params[0][ex], params[1])
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")

if __name__ == "__main__":
    paddle.set_flags({'FLAGS_flash_attn_version': 2})
    from jsonargparse import ArgumentParser
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

    args = parser.parse_args()
    main(**vars(args))
