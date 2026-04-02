import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import time
import paddle
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flashmask_attention
from flashmask_cp_exp import flashmask_cp_allgatherkv, flashmask_cp_allgatherkv_balance
from context_parallel_utils import flashmask_attention_cp
import paddle.distributed.fleet as fleet
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
    
class parallel_env:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        ranks = range(paddle.distributed.get_world_size())
        self.group = paddle.distributed.new_group(ranks, backend="nccl")
        self.degree = self.group.world_size
        self.rank = self.group.rank
        mp_degree = self.degree
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "mp_degree": mp_degree,
        }
        fleet.init(is_collective=True, strategy=strategy)
        
        
    def get_parallel_env_info(self):
        return self.group, self.degree, self.rank

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
     
    
def cp_flashmask_balance(q, k, v, startend_row_indices, is_causal,o_grad, parrel_env):
    group, cp_size, rank = parrel_env.get_parallel_env_info()
    q_blocksize = (int)(q.shape[1] // (2 * cp_size))
    k_blocksize = (int)(k.shape[1] // cp_size)
    q_local_1 = q[:, rank*q_blocksize:(rank+1)*q_blocksize, :, :]
    q_local_2 = q[:, (cp_size *2 -rank -1)*q_blocksize:(cp_size *2 -rank)*q_blocksize, :, :]
    q_local = paddle.concat([q_local_1, q_local_2], axis=1).detach()
    # k_local = k[:, rank*k_blocksize:(rank+1)*k_blocksize, :, :].detach().contiguous()
    # v_local = v[:, rank*k_blocksize:(rank+1)*k_blocksize, :, :].detach().contiguous()
    k_local_1 = k[:, rank*q_blocksize:(rank+1)*q_blocksize, :, :]
    k_local_2 = k[:, (cp_size *2 -rank -1)*q_blocksize:(cp_size *2 -rank)*q_blocksize, :, :]
    k_local = paddle.concat([k_local_1, k_local_2], axis=1).detach()
    
    v_local_1 = v[:, rank*q_blocksize:(rank+1)*q_blocksize, :, :]
    v_local_2 = v[:, (cp_size *2 -rank -1)*q_blocksize:(cp_size *2 -rank)*q_blocksize, :, :]
    v_local = paddle.concat([v_local_1, v_local_2], axis=1).detach()
    
    o_grad_local_1 = o_grad[:, rank * q_blocksize : (rank + 1) * q_blocksize, :, :].detach()
    o_grad_local_2 = o_grad[:, (cp_size * 2 - rank - 1) * q_blocksize : (cp_size * 2 - rank) * q_blocksize, :, :].detach()
    o_grad_local = paddle.concat([o_grad_local_1, o_grad_local_2], axis=1).contiguous()

    
    q_local.stop_gradient = False
    k_local.stop_gradient = False
    v_local.stop_gradient = False
    # startend_row_indices.stop_gradient = False
    out_local = flashmask_attention_cp(q_local, k_local, v_local, startend_row_indices)
    # print(out_local)
    out_local.backward(o_grad_local)
    dq_local = q_local.grad
    dk_local = k_local.grad
    dv_local = v_local.grad
    out_global = []
    dq_global = []
    dk_global = []
    dv_global = []
    paddle.distributed.all_gather(out_global, out_local, group=group)
    paddle.distributed.all_gather(dq_global, dq_local, group=group)
    paddle.distributed.all_gather(dk_global, dk_local, group=group)
    paddle.distributed.all_gather(dv_global, dv_local, group=group)
    out_first_halves = []
    out_second_halves = []
    dq_first_halves = []
    dq_second_halves = []
    dk_first_halves = []
    dk_second_halves = []
    dv_first_halves = []
    dv_second_halves = []

    for i in range(cp_size):
        out_first_halves.append(out_global[i][:, :q_blocksize, :, :])
        out_second_halves.append(out_global[i][:, q_blocksize:, :, :])
        
        dq_first_halves.append(dq_global[i][:, :q_blocksize, :, :])
        dq_second_halves.append(dq_global[i][:, q_blocksize:, :, :])
        
        dk_first_halves.append(dk_global[i][:, :q_blocksize, :, :])
        dk_second_halves.append(dk_global[i][:, q_blocksize:, :, :])
        
        dv_first_halves.append(dv_global[i][:, :q_blocksize, :, :])
        dv_second_halves.append(dv_global[i][:, q_blocksize:, :, :])

    out_global_part1 = paddle.concat(out_first_halves, axis=1)
    dq_global_part1 = paddle.concat(dq_first_halves, axis=1)
    dk_global_part1 = paddle.concat(dk_first_halves, axis=1)
    dv_global_part1 = paddle.concat(dv_first_halves, axis=1)

    out_global_part2 = paddle.concat(out_second_halves[::-1], axis=1)
    dq_global_part2 = paddle.concat(dq_second_halves[::-1], axis=1)
    dk_global_part2 = paddle.concat(dk_second_halves[::-1], axis=1)
    dv_global_part2 = paddle.concat(dv_second_halves[::-1], axis=1)

    out_global = paddle.concat([out_global_part1, out_global_part2], axis=1)
    dq_global = paddle.concat([dq_global_part1, dq_global_part2], axis=1)
    dk_global = paddle.concat([dk_global_part1, dk_global_part2], axis=1)
    dv_global = paddle.concat([dv_global_part1, dv_global_part2], axis=1)
    

    # return out_global
    return out_global, dq_global, dk_global, dv_global

def test_cp_famask(
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
    parrel_env: Optional[parallel_env] = None
):
    # paddle.seed(2024)
    paddle.seed(2024)
    # batch_size = 1
    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]
    # total_k = total_q * 2
    query = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16) 
    key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
    o_grad = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
    print(key.shape)

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        print("enter",generate_mask_fn)
        startend_row_indices, causal = generate_mask_fn(batch_size, total_q, num_head, head_size)
        # startend_row_indices, causal = generate_mask_fn(total_q)
        
    # print(startend_row_indices)
    # paddle.set_printoptions(precision=None, threshold=10000000, edgeitems=None, sci_mode=None, linewidth=None)
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    paddle.device.synchronize()
    (out,lse) = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse = True
        )
    paddle.device.synchronize()
    start_time = time.time()
    out.backward(o_grad)
    paddle.device.synchronize()
    flashattnv1_time = time.time() - start_time
    print("pass")
    
    query1 = query.detach().clone()
    key1 = key.detach().clone()
    value1 = value.detach().clone()
    out1 = out.detach().clone()
    startend_row_indices1 = startend_row_indices.detach().clone()
    o_grad1 = o_grad.detach().clone()
    
    query1.stop_gradient = False
    key1.stop_gradient = False
    value1.stop_gradient = False
    
    paddle.device.synchronize()
    # print(startend_row_indices1)
    # (out1,lse1) = flashmask_attention(
    #         query1,
    #         key1,
    #         value1,
    #         startend_row_indices=startend_row_indices1,
    #         causal=causal,
    #         return_softmax_lse = True
    #     )
    out1,dq1,dk1,dv1 = cp_flashmask_balance(query1, key1, value1, startend_row_indices1, causal,o_grad1, parrel_env)
    paddle.device.synchronize()
    # out1.backward(o_grad1)
    # paddle.device.synchronize()
    
    # print("pypt2:")
    # print(startend_row_indices)
    paddle.device.synchronize()
    flashattnv2_time = time.time() - start_time
    # with open("execution_times.txt", "a") as log_file:
    #     log_file.write(f"bsz: {batch_size},num_head_k: {num_head},num_head_q: {num_head * 4},hsz: {head_size},seqlen: {total_q}, flashattnv1: {flashattnv1_time:.6f}s, "
    #                     f"flashattnv2: {flashattnv2_time:.6f}s\n")
    for x,y in [(out1,out),(dq1,query.grad),(dk1,key.grad),(dv1,value.grad)]:
        strict_check(x.flatten(), y.flatten())
    # for x,y in [(out1,out)]:
    #     strict_check(x.flatten(), y.flatten())
    
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
        np.testing.assert_allclose(x.flatten(), y.flatten(),rtol=1e-2, atol=1e-2)
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

def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m

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

def generate_upper_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213],padding_size = 256):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    up_right_row_indices = []

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    down_left_row_indices =  paddle.ones_like(up_right_row_indices) * (S - padding_size)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
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

def generate_prefix_lm_padding_document_mask(B, S, H, D, start_id, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
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

    causal_padding = paddle.arange(
        0, S, dtype="int32"
    ).reshape((1, 1, S, 1))
    
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    startend_row_indices[:,:,start_id:,:] = causal_padding[:,:,start_id:,:]

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
    bias = 0
    padding_ratio = 0.1
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
        #     # "Sliding Window": {'window_size': 64},
        #     # "Global Sliding Window": {'global_token': 1, 'window_size': (32, 32)},
        #     "Ones": {},
        #     # "Causal": {},
        #     # "Causal Document Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     "Document Mask": {'doc_seq_lens': list([2, 63, 63])},
        #     # "Share Question Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     # "Causal Blockwise Mask": {'doc_seq_lens': list([127, 133, 124])},
        #     # "Prefix LM Document Mask": {'doc_seq_lens': list([(1, 127), (35, 133), (50, 124)])},
        #     # "Prefix LM Causal Mask": {'prefix_length': 1},
        #     # "QK-sparse Mask": {'maskout_pair': list([(1, 34), (63, 33)])},
        #     # "Random Eviction Mask": {"start_row": 32},
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
        [{
            "Sliding Window": {'window_size': 13},
            "Global Sliding Window": {'global_token': 1, 'window_size': (32, 32)},
            "Ones": {},
            "Causal": {},
            "Ones Padding Mask": {'start_id': (int)(512 - padding_ratio * 512)},
            "Causal Padding Mask": {'start_id': (int)(512 - padding_ratio * 512)},
            "Causal Document Mask": {'doc_seq_lens': list([127, 256, 129+bias])},
            "Upper Document Mask": {'doc_seq_lens': list([127, 256, 129+bias]), 'padding_size': int(32)},
            "Document Mask": {'doc_seq_lens': list([127, 256, 129+bias])},
            "Share Question Mask": {'doc_seq_lens': list([127, 256, 129+bias])},
            "Causal Blockwise Mask": {'doc_seq_lens': list([127, 256, 129+bias])},
            "Prefix LM Document Mask": {'doc_seq_lens': list([(1, 127), (35, 256), (50, 129+bias)])},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': list([(1, 127), (35, 256), (50, 129+bias)]), 'start_id': (int)(512 - padding_ratio * 512)},
            "Prefix LM Causal Mask": {'prefix_length': 100},
            "QK-sparse Mask": {'maskout_pair': list([(1, 34), (63, 33)])},
            "Random Eviction Mask": {"start_row": 32},
        },
        {
            'cu_seqlens_q': list([0,512+bias]),
            'cu_seqlens_k': list([0,512+bias]),
            'num_head': int(1),
            'head_size': int(128),
            'batch_size': int(1),
            'max_seqlen_q': int(512+bias),
            'max_seqlen_k': int(512+bias),
        }],
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
        #     'head_size': int(128),
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
        #     "Upper Document Mask": {'doc_seq_lens': list([2538, 1742, 3213]), 'padding_size': int(333)},
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
        #     'num_head': int(1),
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
            "Upper Document Mask": {'doc_seq_lens': [2538, 1742, 3912], 'padding_size': int(133)},
            "Document Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Share Question Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Causal Blockwise Mask": {'doc_seq_lens': [2538, 1742, 3912]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(1024, 2538), (1742, 1742), (1146, 3912)]},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': [(1024, 2538), (1742, 1742), (1146, 3912)], 'start_id': (int)(8192 - padding_ratio * 8192)},
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
        [{
            "Sliding Window": {'window_size': 512},
            "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
            "Ones": {},
            "Causal": {},
            "Causal Document Mask": {'doc_seq_lens': [4096, 6100, 6188+bias]},
            "Upper Document Mask": {'doc_seq_lens': [4096, 6100, 6188+bias], 'padding_size': int(311)},
            "Document Mask": {'doc_seq_lens': [4096, 6100, 6188+bias]},
            "Share Question Mask": {'doc_seq_lens': [4096, 6100, 6188+bias]},
            "Causal Blockwise Mask": {'doc_seq_lens': [4096, 6100, 6188+bias]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(1024, 4096), (2048, 6100), (2048, 6188+bias)]},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': [(1024, 4096), (2048, 6100), (2048, 6188+bias)], 'start_id': (int)(16384 - padding_ratio * 16384)}, 
            "Prefix LM Causal Mask": {'prefix_length': 1025},
            "QK-sparse Mask": {'maskout_pair': [(4096, 2048), (8192, 1024)]},
            "Random Eviction Mask": {"start_row": 8191},
        },
        {
            'cu_seqlens_q': [0, 4096, 10240, 16384+bias],  # 0, 4096, 4096+6144=10240, 10240+6144=16384
            'cu_seqlens_k': [0, 4096, 10240, 16384+bias],
            'num_head': 1,
            'head_size': head_size,
            'batch_size': 1,
            'max_seqlen_q': 6144,
            'max_seqlen_k': 6144,
        }],
        [{
            "Sliding Window": {'window_size': 512},
            "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
            "Ones": {},
            "Causal": {},
            "Causal Document Mask": {'doc_seq_lens': [4096, 6100, 6188]},
            "Upper Document Mask": {'doc_seq_lens': [4096, 6100, 6188], 'padding_size': int(311)},
            "Document Mask": {'doc_seq_lens': [4096, 6100, 6188]},
            "Share Question Mask": {'doc_seq_lens': [4096, 6100, 6188]},
            "Causal Blockwise Mask": {'doc_seq_lens': [4096, 6100, 6188]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(1024, 4096), (2048, 6100), (2048, 6188)]},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': [(1024, 4096), (2048, 6100), (2048, 6188)], 'start_id': (int)(16384 - padding_ratio * 16384)}, 
            "Prefix LM Causal Mask": {'prefix_length': 1025},
            "QK-sparse Mask": {'maskout_pair': [(4096, 2048), (8192, 1024)]},
            "Random Eviction Mask": {"start_row": 8191},
        },
        {
            'cu_seqlens_q': [0, 4096, 10240, 16384],  # 0, 4096, 4096+6144=10240, 10240+6144=16384
            'cu_seqlens_k': [0, 4096, 10240, 16384],
            'num_head': 1,
            'head_size': head_size,
            'batch_size': 1,
            'max_seqlen_q': 6144,
            'max_seqlen_k': 6144,
        }],
        [{
            "Sliding Window": {'window_size': 512},
            "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
            "Ones": {},
            "Causal": {},
            "Causal Document Mask": {'doc_seq_lens': [8192, 12293, 12283]},
            "Upper Document Mask": {'doc_seq_lens': [8192, 12293, 12283], 'padding_size': int(313)},
            "Document Mask": {'doc_seq_lens': [1024 for  _ in range(32)]},
            "Share Question Mask": {'doc_seq_lens': [8192, 12293, 12283]},
            "Causal Blockwise Mask": {'doc_seq_lens': [8192, 12293, 12283]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(4096, 8192), (4096, 12293), (4096, 12283)]},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': [(4096, 8192), (4096, 12293), (4096, 12283)], 'start_id': (int)(32768 - padding_ratio * 32768)},
            "Prefix LM Causal Mask": {'prefix_length': 4096},
            "QK-sparse Mask": {'maskout_pair': [(8192, 4096), (24576, 4096)]},
            "Random Eviction Mask": {"start_row": 3333},
        },
        {
            'cu_seqlens_q': [0, 8192, 20480, 32768],  # 0, 8192, 8192+12288=20480, 20480+12288=32768
            'cu_seqlens_k': [0, 8192, 20480, 32768],
            'num_head': 8,
            'head_size': head_size,
            'batch_size': 1,
            'max_seqlen_q': 12288,
            'max_seqlen_k': 12288,
        }],
        [{
            "Sliding Window": {'window_size': 512},
            "Global Sliding Window": {'global_token': 16, 'window_size': (512, 512)},
            "Ones": {},
            "Causal": {},
            "Causal Document Mask": {'doc_seq_lens': [65536, 32768, 32768]},
            "Upper Document Mask": {'doc_seq_lens': [65536, 32768, 32768], 'padding_size': int(323)},
            "Document Mask": {'doc_seq_lens': [65536, 32768, 32768]},
            "Share Question Mask": {'doc_seq_lens': [65536, 32768, 32768]},
            "Causal Blockwise Mask": {'doc_seq_lens': [65536, 32768, 32768]},
            "Prefix LM Document Mask": {'doc_seq_lens': [(32768, 65536), (16384, 32768), (16384, 32768)]},
            "Prefix LM Padding Document Mask": {'doc_seq_lens': [(32768, 65536), (16384, 32768), (16384, 32768)], 'start_id': (int)(131072 - padding_ratio * 131072)},
            "Prefix LM Causal Mask": {'prefix_length': 65536},
            "QK-sparse Mask": {'maskout_pair': [(2, 32768), (64433, 32768)]},
            "Random Eviction Mask": {"start_row": 65536},
        },
        {
            'cu_seqlens_q': [0, 65536, 98304, 131072],  # 0, 65536, 65536+32768=98304, 98304+32768=131072
            'cu_seqlens_k': [0, 65536, 98304, 131072],
            'num_head': 8,
            'head_size': head_size,
            'batch_size': 1,
            'max_seqlen_q': 65536,  # Should be the maximum in doc_seq_lens
            'max_seqlen_k': 65536,
        }]
    ]

    parrel_env = parallel_env()
    print("here")
    available_examples = {
        # "Ones": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_ones_mask, **params0), **params1),
        # "Causal": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_causal_mask, **params0), **params1),
        # "Sliding Window": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_sliding_window_mask, **params0), **params1),
        # "Causal Document Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_causal_document_mask, **params0), **params1),
        # "Upper Document Mask":lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_upper_document_mask, **params0),parrel_env=parrel_env, **params1),
        # "Document Mask": lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_document_mask, **params0),parrel_env=parrel_env, **params1),
        # "Share Question Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_share_question_mask,**params0), **params1),
        # "Global Sliding Window": lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_global_sliding_window_mask,**params0),parrel_env=parrel_env, **params1),
        # "Causal Blockwise Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_causal_blockwise_mask,**params0), **params1),
        # "Prefix LM Document Mask": lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_document_mask, **params0),parrel_env=parrel_env, **params1),
        "Prefix LM Padding Document Mask": lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_padding_document_mask, **params0),parrel_env=parrel_env, **params1),
        # "Prefix LM Causal Mask": lambda params0,params1: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, **params0),parrel_env=parrel_env, **params1),
        # "QK-sparse Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_qk_sparse_mask, **params0), **params1),
        # "Random Eviction Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_random_eviction_mask, **params0), **params1),
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
    #     "Sliding Window": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_sliding_window_mask, **params0), **params1),
    #     "Causal Document Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_causal_document_mask, **params0), **params1),
    #     "Document Mask": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_document_mask, **params0), **params1),
    #     "Global Sliding Window": lambda params0,params1: test_chunk_famask(generate_mask_fn=partial(generate_global_sliding_window_mask,**params0), **params1),
    #  }

    if "all" in examples:
        ex_to_run = list(available_examples.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in available_examples:
            for params in params_list:
                print(f"Running {ex}\n")
                # with open("execution_times.txt", "a") as log_file:
                #     log_file.write(f"{ex}: ")
                available_examples[ex](params[0][ex], params[1])
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")

if __name__ == "__main__":
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
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
