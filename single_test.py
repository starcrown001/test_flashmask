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
# cu_seqlens_q: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
# cu_seqlens_k: list = [0, 63, 128, 256, 1280, 1297, 1397, 1408],
cu_seqlens_q = [0, 128]
cu_seqlens_k = [0, 128]
batch_size = 1
num_head_q = 6
num_head_k = 6
head_size = 128
# paddle.seed(2024)
paddle.seed(2024)
# batch_size = 1
causal = True

total_q = cu_seqlens_q[-1] 
total_k = cu_seqlens_k[-1] 
# print([batch_size, total_q, num_head, head_size])
query = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16) 
key = paddle.randn([batch_size, total_k, num_head_k, head_size], dtype=paddle.bfloat16)
value = paddle.randn([batch_size, total_k, num_head_k, head_size], dtype=paddle.bfloat16)
o_grad = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
empty_tensor = paddle.empty(shape=[0],dtype="float32")
# print(query.shape)

query = paddle.load("query.pd")
key = paddle.load("key.pd")
value = paddle.load("value.pd")
o_grad = paddle.load("g.pd")
o_grad = paddle.randn(shape=query.shape, dtype=paddle.bfloat16)

print(query.shape)
print(key.shape)
print(paddle.__git_commit__)
# tiles = [1, 1, num_head, 1]  # 在倒数第二个维度上重复 num_head 次
# query = paddle.tile(query, tiles)
# key = paddle.tile(key, tiles)
# value = paddle.tile(value, tiles)
# o_grad = paddle.tile(o_grad, tiles)


query.stop_gradient = False
key.stop_gradient = False
value.stop_gradient = False

paddle.set_flags({'FLAGS_flash_attn_version': 3})

startend_row_indices1 = paddle.load("startend_row_indices.pd")


# blockmask = random_blockmask_with_at_least_one(
#     shape=[
#         startend_row_indices1.shape[0],
#         startend_row_indices1.shape[1],
#         (total_q + 127)// 128,
#         (total_k + 127)// 128
#     ],
#     dtype=paddle.int32,
#     ref_q = query
# )
# print(query)
# blockmask = paddle.load("blockmask.pd")
# print(blockmask)
# blockmask = paddle.tensor(       [[[[1, 0, 0],
#       [0, 1,0],
#       [1, 0, 1]]]],dtype=paddle.int32)
print(startend_row_indices1)
paddle.base.core.nvprof_nvtx_push("paddle")
(out1,lse1) = flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=startend_row_indices1,
        causal=causal,
        return_softmax_lse = True)
        # block_mask_indices=blockmask)
out1.backward(o_grad)
paddle.device.synchronize()
paddle.base.core.nvprof_nvtx_pop()

# print("pypt2:")
# print(startend_row_indices)
paddle.device.synchronize()
# for x,y in [(key1.grad, key.grad),(value1.grad, value.grad),(query1.grad, query.grad)]:
#     strict_check(x.flatten(), y.flatten())
# for x,y in [(out1,out),(lse1,lse)]:
#     strict_check(x.flatten(), y.flatten())
# print("pass")

# q1 = query.detach()
# k1 = key.detach()
# v1 = value.detach()
# q1.stop_gradient = False
# k1.stop_gradient = False
# v1.stop_gradient = False

# batch_size = startend_row_indices1.shape[0]
# num_head_mask = startend_row_indices1.shape[1]
# total_q = query.shape[1]
# total_k = key.shape[1]
# # print(blockmask)
# print(startend_row_indices1)
# mask_flash = flashmask_to_densemask(startend_row_indices1,total_q, num_head_mask, causal)
# # print(mask_flash)
# # print(blockmask)
# mask_block = blockmask_to_densemask(blockmask,q1.shape[1],k1.shape[1],paddle.int32,causal)
# # print(mask_block)
# mask_inf = mask_flash & mask_block
# print(mask_inf)
# mask = paddle.zeros((batch_size, num_head_mask, total_q, total_k), dtype=paddle.bfloat16)
# mask = paddle.where(mask_inf, paddle.zeros_like(mask), paddle.full_like(mask, float('-inf')))  
# mask1 = mask_inf.sum(axis = -1)  
# paddle.set_printoptions(precision=None, threshold=10000000, edgeitems=None, sci_mode=None, linewidth=None)
# print(mask1)
# # print(mask)
# # attn_bias = paddle.load("attn_bias.pd")
# # assert paddle.equal_all(attn_bias.astype(paddle.float32),mask.astype(paddle.float32))
# ref_out1, attn_ref = attention_ref(
#     q1,
#     k1,
#     v1,
#     causal=causal,
#     attn_bias=mask
# )
# # ref_out1 = attention_naive_with_mask_varlen(q1, k1, v1, mask,[0,q1.shape[1]])
# # ref_out1 = paddle.where(paddle.isnan(ref_out1), paddle.zeros_like(ref_out1), ref_out1)
# paddle.device.synchronize()
# print(ref_out1.shape,out1.shape)
# strict_check(ref_out1.flatten(), out1.flatten())
# ref_out1.backward(o_grad)

# paddle.device.synchronize()
# # print(o_grad[0,0,:,0])
# print(q1.grad[0,:,0,0])
# print(query.grad[0,:,0,0])
# for x,y in [(key.grad,k1.grad),(value.grad,v1.grad),(query.grad,q1.grad)]:
#     strict_check(x.flatten(), y.flatten())
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

