#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextParallel API 单测
"""
import os
import unittest
import random
import re
import contextlib
from copy import deepcopy
from collections import defaultdict
from paddle.nn.functional.flash_attention import flashmask_attention
from functools import partial

import numpy as np
import paddle
from paddle.nn import functional as F
import paddle.distributed.fleet as fleet
import paddle.distributed as dist

from src.ernie_core.models.context_parallel_utils import flashmask_attention_cp


class TestFlashMaskContextParallelAllgatherKV(unittest.TestCase):
    def setUp(self):
        # 环境变量设置，保证可重复性
        os.environ["FLAGS_cudnn_deterministic"] = "True"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        self.world_size = paddle.distributed.get_world_size()
        self.DTYPE = "bfloat16"
        self.rtol = 1.0e-1
        self.use_flash_attn = True

        assert self.world_size == 8
        self.pp_degree, self.tp_degree, self.dp_degree, self.cp_degree, self.sharding_degree = (1, 1, 1, 4, 1)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 8,
            "sep_degree": 1,
            "ep_degree": 2,
            "moe_sharding_degree": 4,
            "cp_degree": 4,
            "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"],
        }
        self.ACC = 2
        strategy.pipeline_configs = {
            "accumulate_steps": self.ACC,
            "micro_batch_size": 1,
            "enable_partial_send_recv": False,
            "p2p_cache_shape": True,
        }
        strategy.tensor_parallel_configs = {"tensor_init_seed": 42}
        fleet.init(is_collective=True, strategy=strategy)

        self.scaler = paddle.amp.GradScaler(init_loss_scaling=2**12) if self.DTYPE == "float16" else None

        if self.DTYPE != "float32":
            print(f"Running Unittest in {self.DTYPE}")
            self.autocast = lambda: paddle.amp.auto_cast(
                True,
                level="O2",
                dtype=self.DTYPE,
                custom_black_list=[
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                    "elementwise_div",
                    "sin",
                    "cos",
                ],
                custom_white_list=[
                    "lookup_table",
                    "lookup_table_v2",
                    "flash_attn",
                    "flash_attn_v1",
                    "matmul",
                    "matmul_v2",
                    "fused_gemm_epilogue",
                ],
            )
        else:
            self.autocast = lambda: contextlib.nullcontext()

        self.hcg = fleet.get_hybrid_communicate_group()
        self.dp_rank = self.hcg.get_data_parallel_rank()
        self.mp_rank = self.hcg.get_model_parallel_rank()
        self.pp_rank = self.hcg.get_stage_id()
        self.group = self.hcg.get_context_parallel_group()
        self.cp_degree = self.hcg.get_context_parallel_world_size()
        self.cp_rank = self.hcg.get_context_parallel_rank()

        print(
            f"init fleet: hybrid_config: {strategy.hybrid_configs}, "
            f"pp_config: {strategy.pipeline_configs} "
            f"pp={self.pp_rank} mp={self.mp_rank} dp={self.dp_rank}"
        )
        print("init done")
        self.strategy = strategy

        self.test_config = [
            [
                {
                    "Global Sliding Window": {"global_token": 16, "window_size": (512, 512)},
                    "Document Mask": {"doc_seq_lens": [2538, 1742, 3912]},
                    "Prefix LM Causal Mask": {"prefix_length": 1024},
                    "Prefix LM Padding Document Mask": {
                        "doc_seq_lens": [(1024, 2538), (1742, 1742), (1146, 3912)],
                        "start_id": int(8192 - 0.25 * 8192),
                    },
                },
                {
                    "cu_seqlens_q": [0, 2538, 4278, 8192],
                    "cu_seqlens_k": [0, 2538, 4278, 8192],
                    "num_head": 4,
                    "head_size": 8,
                    "batch_size": 1,
                },
            ]
        ]

    def generate_document_mask(self, batch_size, seq_len, num_head, head_size, doc_seq_lens=[2538, 1742, 3213]):
        total_seq_len = np.sum(doc_seq_lens)
        assert total_seq_len <= seq_len
        padding = seq_len - total_seq_len

        down_left_indices, up_right_indices = [], []
        cur_len = doc_seq_lens[0]
        for i, doc_len in enumerate(doc_seq_lens):
            down_left_indices.extend([cur_len] * doc_len)
            if i < len(doc_seq_lens) - 1:
                cur_len += doc_seq_lens[i + 1]
        if padding > 0:
            down_left_indices.extend([cur_len] * padding)

        cur_len = 0
        for i, doc_len in enumerate(doc_seq_lens):
            up_right_indices.extend([cur_len] * doc_len)
            if i < len(doc_seq_lens) - 1:
                cur_len += doc_seq_lens[i]
        if padding > 0:
            up_right_indices.extend([cur_len] * padding)

        down_left_indices = (
            paddle.to_tensor(down_left_indices, dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )
        up_right_indices = (
            paddle.to_tensor(up_right_indices, dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )
        startend_indices = paddle.concat([down_left_indices, up_right_indices], axis=-1)
        causal = False
        return startend_indices, causal

    def generate_global_sliding_window_mask(
        self, batch_size, seq_len, num_head, head_size, global_token=16, window_size=(512, 512)
    ):
        assert len(window_size) == 2
        left_size, right_size = window_size

        down_left_start = paddle.arange(left_size + 1, seq_len + left_size + 1, dtype="int32").clip(max=seq_len)
        down_left_start[:global_token] = seq_len
        down_left_start = down_left_start.reshape((1, 1, seq_len, 1)).repeat_interleave(batch_size, 0)
        down_left_end = (
            paddle.full([seq_len], seq_len, dtype="int32").reshape((1, 1, seq_len, 1)).repeat_interleave(batch_size, 0)
        )

        up_right_start = paddle.full([seq_len], global_token, dtype="int32")
        up_right_start[: global_token + right_size + 1] = 0
        up_right_start = up_right_start.reshape((1, 1, seq_len, 1)).repeat_interleave(batch_size, 0)

        up_right_end = paddle.arange(-right_size, seq_len - right_size, dtype="int32")
        up_right_end[: global_token + right_size + 1] = 0
        up_right_end = up_right_end.reshape((1, 1, seq_len, 1)).repeat_interleave(batch_size, 0)

        startend_indices = paddle.concat([down_left_start, down_left_end, up_right_start, up_right_end], axis=-1)
        causal = False
        return startend_indices, causal

    def generate_prefix_lm_padding_document_mask(
        self, batch_size, seq_len, num_head, head_size, start_id, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]
    ):
        assert len(doc_seq_lens) >= 2
        total_seq_len = sum(seq_length for _, seq_length in doc_seq_lens)
        assert total_seq_len <= seq_len
        padding = seq_len - total_seq_len

        down_left_indices = []
        cur_len = doc_seq_lens[0][1]
        for i, (_, seq_length) in enumerate(doc_seq_lens):
            down_left_indices.extend([cur_len] * seq_length)
            if i < len(doc_seq_lens) - 1:
                cur_len += doc_seq_lens[i + 1][1]
        if padding > 0:
            down_left_indices.extend([cur_len] * padding)
        down_left_indices = (
            paddle.to_tensor(down_left_indices, dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )

        up_right_indices = []
        cur_len = 0
        for prefix_length, seq_length in doc_seq_lens:
            up_right_indices.extend(
                [cur_len] * prefix_length + list(range(cur_len + prefix_length, cur_len + seq_length))
            )
            cur_len += seq_length
        if padding > 0:
            up_right_indices.extend([total_seq_len] * padding)
        up_right_indices = (
            paddle.to_tensor(up_right_indices, dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )

        causal_padding = paddle.arange(0, seq_len, dtype="int32").reshape((1, 1, seq_len, 1))
        startend_indices = paddle.concat([down_left_indices, up_right_indices], axis=-1)
        startend_indices[:, :, start_id:, :] = causal_padding[:, :, start_id:, :]

        causal = False
        return startend_indices, causal

    def generate_prefix_lm_causal_mask(self, batch_size, seq_len, num_head, head_size, prefix_length=1024):
        assert prefix_length <= seq_len
        down_left_indices = (
            paddle.full([seq_len], seq_len, dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )
        up_right_indices = (
            paddle.to_tensor([0] * prefix_length + list(range(prefix_length, seq_len)), dtype=paddle.int32)
            .reshape((1, 1, seq_len, 1))
            .repeat_interleave(batch_size, 0)
        )
        startend_indices = paddle.concat([down_left_indices, up_right_indices], axis=-1)
        causal = False
        return startend_indices, causal

    def cp_flashmask_balance(self, q, k, v, startend_indices, is_causal, o_grad):
        group, cp_size, rank = self.group, self.cp_degree, self.cp_rank
        q_blocksize = int(q.shape[1] // (2 * cp_size))
        k_blocksize = int(k.shape[1] // cp_size)

        def local_slice(x, name):
            first = x[:, rank * q_blocksize : (rank + 1) * q_blocksize, :, :]
            second = x[:, (cp_size * 2 - rank - 1) * q_blocksize : (cp_size * 2 - rank) * q_blocksize, :, :]
            return paddle.concat([first, second], axis=1).detach()

        q_local = local_slice(q, "q")
        k_local = local_slice(k, "k")
        v_local = local_slice(v, "v")
        o_grad_local = local_slice(o_grad, "o_grad").contiguous()

        q_local.stop_gradient = False
        k_local.stop_gradient = False
        v_local.stop_gradient = False

        out_local = flashmask_attention_cp(q_local, k_local, v_local, startend_indices)
        out_local.backward(o_grad_local)
        dq_local = q_local.grad
        dk_local = k_local.grad
        dv_local = v_local.grad

        # all_gather across context parallel group
        out_global, dq_global, dk_global, dv_global = [], [], [], []
        paddle.distributed.all_gather(out_global, out_local, group=group)
        paddle.distributed.all_gather(dq_global, dq_local, group=group)
        paddle.distributed.all_gather(dk_global, dk_local, group=group)
        paddle.distributed.all_gather(dv_global, dv_local, group=group)

        def split_halves(tensors):
            first_halves = [t[:, :q_blocksize, :, :] for t in tensors]
            second_halves = [t[:, q_blocksize:, :, :] for t in tensors]
            return first_halves, second_halves

        out_first, out_second = split_halves(out_global)
        dq_first, dq_second = split_halves(dq_global)
        dk_first, dk_second = split_halves(dk_global)
        dv_first, dv_second = split_halves(dv_global)

        out_global = paddle.concat([paddle.concat(out_first, axis=1), paddle.concat(out_second[::-1], axis=1)], axis=1)
        dq_global = paddle.concat([paddle.concat(dq_first, axis=1), paddle.concat(dq_second[::-1], axis=1)], axis=1)
        dk_global = paddle.concat([paddle.concat(dk_first, axis=1), paddle.concat(dk_second[::-1], axis=1)], axis=1)
        dv_global = paddle.concat([paddle.concat(dv_first, axis=1), paddle.concat(dv_second[::-1], axis=1)], axis=1)

        return out_global, dq_global, dk_global, dv_global

    def strict_check(self, x, y):
        def to_numpy(t):
            if isinstance(t, paddle.Tensor):
                if t.dtype in [paddle.bfloat16, paddle.float16]:
                    return t.cast("float32").numpy()
                return t.numpy()
            raise TypeError("Input must be a paddle.Tensor.")

        x_np = to_numpy(x)
        y_np = to_numpy(y)
        try:
            print(f"{x_np=}, {y_np=}")
            np.testing.assert_allclose(x_np.flatten(), y_np.flatten(), rtol=1e-2, atol=1e-2)
        except Exception as e:
            print("---------------")
            idx = np.where(~(x_np == y_np))
            print(f"fail idx: {idx=}")
            print(f"shape: {x_np.shape}")
            print(x_np[idx])
            print(y_np[idx])
            raise e

    def cp_famask_test(
        self,
        generate_mask_fn,
        cu_seqlens_q=[0, 63, 128],
        cu_seqlens_k=[0, 63, 128],
        batch_size=1,
        num_head=1,
        head_size=64,
    ):
        paddle.seed(2024)
        total_q = cu_seqlens_q[-1]
        total_k = cu_seqlens_k[-1]
        query = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
        key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        o_grad = paddle.randn([batch_size, total_q, num_head, head_size], dtype=paddle.bfloat16)
        print(key.shape)

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        startend_indices, causal = None, True
        if generate_mask_fn is not None:
            print("enter", generate_mask_fn)
            startend_indices, causal = generate_mask_fn(batch_size, total_q, num_head, head_size)

        paddle.device.synchronize()
        out, lse = flashmask_attention(
            query, key, value, startend_row_indices=startend_indices, causal=causal, return_softmax_lse=True
        )
        paddle.device.synchronize()
        out.backward(o_grad)
        paddle.device.synchronize()

        query1 = query.detach().clone()
        key1 = key.detach().clone()
        value1 = value.detach().clone()
        out1 = out.detach().clone()
        startend_indices1 = startend_indices.detach().clone() if startend_indices is not None else None
        o_grad1 = o_grad.detach().clone()

        query1.stop_gradient = False
        key1.stop_gradient = False
        value1.stop_gradient = False

        paddle.device.synchronize()
        out1, dq1, dk1, dv1 = self.cp_flashmask_balance(query1, key1, value1, startend_indices1, causal, o_grad1)
        paddle.device.synchronize()

        for x, y in [(out1, out), (dq1, query.grad), (dk1, key.grad), (dv1, value.grad)]:
            self.strict_check(x.flatten(), y.flatten())

    def test_all(self):
        available_examples = {
            "Document Mask": lambda params0, params1: self.cp_famask_test(
                generate_mask_fn=partial(self.generate_document_mask, **params0), **params1
            ),
            "Global Sliding Window": lambda params0, params1: self.cp_famask_test(
                generate_mask_fn=partial(self.generate_global_sliding_window_mask, **params0), **params1
            ),
            "Prefix LM Padding Document Mask": lambda params0, params1: self.cp_famask_test(
                generate_mask_fn=partial(self.generate_prefix_lm_padding_document_mask, **params0), **params1
            ),
            "Prefix LM Causal Mask": lambda params0, params1: self.cp_famask_test(
                generate_mask_fn=partial(self.generate_prefix_lm_causal_mask, **params0), **params1
            ),
        }

        paddle.set_flags({"FLAGS_flash_attn_version": 2})
        for ex_name, ex_func in available_examples.items():
            for params in self.test_config:
                print(f"Running {ex_name}\n")
                ex_func(params[0][ex_name], params[1])

        if hasattr(paddle.base.libpaddle.pir.ops, "flashmask_attention_v2_grad"):
            paddle.set_flags({"FLAGS_flash_attn_version": 3})
            for ex_name, ex_func in available_examples.items():
                for params in self.test_config:
                    print(f"Running {ex_name}\n")
                    ex_func(params[0][ex_name], params[1])


if __name__ == "__main__":
    unittest.main()