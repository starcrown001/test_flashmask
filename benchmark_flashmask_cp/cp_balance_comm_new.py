# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
通信感知的负载均衡模块。

在原有 cp_balance.py 仅考虑计算均衡的基础上，引入通信惩罚因子，
在任务分配时同时优化计算均衡和跨 rank 通信量。

核心思路：
1. 通过 naive Python 实现计算每个 q_chunk 的 kblock activation map，
   即每个 q_chunk 实际需要 attend 到哪些 kblock。
2. 在贪心分配时，综合考虑桶的当前计算负载和新增通信代价（新增 kblock 数量），
   选择综合得分最低的桶进行分配。
"""

import heapq
import paddle
import numpy as np
from collections import defaultdict
from flash_mask.cp_balance.cp_balance_cuda_kernels import (
    scanMaxMinChunkedKernel,
    reduce_workload,
    indices_to_chunks_cuda,
    indices_rerank_cuda,
    cp_balance_ipo_solve,
)
from flash_mask.cp_balance.cp_balance import get_send_dict, get_recv_dict, balance_alltoall, assign_tasks_heap
import paddle.distributed as dist
from typing import List, Tuple, Dict, Optional


def _compute_activation_map_naive(
    LTStartMax, LTStartMin, LTEndMax, LTEndMin,
    UTStartMax, UTStartMin, UTEndMax, UTEndMin,
    B: int, H: int, Tr: int, Tc: int, m_block_size: int, S: int
) -> np.ndarray:
    """
    Naive Python 实现：计算每个 (q_block, k_block) 对是否为活跃块。

    复现 reduce_workload CUDA kernel 中对 fully_masked 的判断逻辑，
    但保留完整的 (Tr, Tc) 二维信息而非归约为标量。

    FLASHMASK 的 column-wise mask 表示：
      对第 j 列 key token, [LTS_j, LTE_j) ∪ [UTS_j, UTE_j) 为被 mask 的 query 行。
      scanMaxMin 将每个 k_block 内的列索引预聚合为 max/min 值。
      对 q_block [m_block_s, m_block_e)，若整个 q_block 都被 mask 则 fully_masked。

    Args:
        LTStartMax..UTEndMin: scanMaxMinChunkedKernel 的输出，
            形状为 [B*H, padded_Tc] 的 GPU Tensor 或 None。
        B, H, Tr, Tc, m_block_size, S: 维度参数。

    Returns:
        np.ndarray: 形状为 [B*H, Tr, Tc] 的 bool 数组，
                    True 表示该 (q_block, k_block) 对是活跃的（需要计算）。
    """
    BH = B * H

    def to_np(t):
        """将 GPU Tensor 转为 CPU numpy，并裁掉 scanMaxMin 的 padding。"""
        if t is not None:
            return t.cpu().numpy().reshape(BH, -1)[:, :Tc]
        return None

    lt_start_max = to_np(LTStartMax)
    lt_start_min = to_np(LTStartMin)
    lt_end_max = to_np(LTEndMax)
    lt_end_min = to_np(LTEndMin)
    ut_start_max = to_np(UTStartMax)
    ut_start_min = to_np(UTStartMin)
    ut_end_max = to_np(UTEndMax)
    ut_end_min = to_np(UTEndMin)

    # q_block 边界: [Tr]
    m_block_s = np.arange(Tr, dtype=np.int64) * m_block_size
    m_block_e = np.minimum(m_block_s + m_block_size, S)

    # 广播形状: m_s/m_e -> [1, Tr, 1], indices -> [BH, 1, Tc]
    m_s = m_block_s[np.newaxis, :, np.newaxis]  # [1, Tr, 1]
    m_e = m_block_e[np.newaxis, :, np.newaxis]  # [1, Tr, 1]

    lt_s_max = lt_start_max[:, np.newaxis, :]   # [BH, 1, Tc]

    # 根据可用指针类型分派（与 CUDA kernel 的 PtrDispatch 对齐）
    if lt_end_max is not None and ut_start_max is not None:
        # FULL_PTR: 四个向量全部可用 (LTS, LTE, UTS, UTE)
        lt_e_min = lt_end_min[:, np.newaxis, :]
        ut_s_max = ut_start_max[:, np.newaxis, :]
        ut_e_min = ut_end_min[:, np.newaxis, :]
        fully_masked = (
            ((m_s >= lt_s_max) & (m_e <= lt_e_min)) |
            ((m_s >= ut_s_max) & (m_e <= ut_e_min))
        )
    elif lt_end_max is not None:
        # DUAL_PTR causal: LTS + LTE
        lt_e_min = lt_end_min[:, np.newaxis, :]
        fully_masked = (m_s >= lt_s_max) & (m_e <= lt_e_min)
    elif ut_end_max is not None:
        # DUAL_PTR non-causal: LTS + UTE (shape[-1]==2 的情况)
        ut_e_min = ut_end_min[:, np.newaxis, :]
        fully_masked = (m_s >= lt_s_max) | (m_e <= ut_e_min)
    else:
        # SINGLE_PTR: 仅 LTS
        fully_masked = m_s >= lt_s_max

    activation_map = ~fully_masked  # [BH, Tr, Tc]
    return activation_map


def get_q_workload_with_activation_map(
    start_row_indices: paddle.Tensor,
    q_chunk_size: int,
    m_block_size: int,
    n_block_size: int
) -> Tuple[paddle.Tensor, np.ndarray]:
    """
    在估算每个 q_chunk 计算工作负载的同时，计算其 kblock activation map。

    工作负载部分复用 reduce_workload CUDA kernel（与 get_q_workload 完全一致）。
    Activation map 部分使用 naive Python 实现，在 CPU 上通过向量化 numpy 运算
    复现 CUDA kernel 的 fully_masked 判断逻辑，保留完整的 (Tr, Tc) 信息后
    聚合到 chunk 粒度。

    Args:
        start_row_indices (paddle.Tensor): 形状为 [B, H, S, 2] 或 [B, H, S, 4]。
        q_chunk_size (int): 负载均衡分析的 chunk 大小。
        m_block_size (int): FlashAttention 的 query 块大小 (Br)。
        n_block_size (int): FlashAttention 的 key 块大小 (Bc)。

    Returns:
        Tuple:
            - workload (paddle.Tensor): 形状为 [1, 1, Tchunks, 2]，
              对所有 head 的 workload 求和，每行 [workload_sum, chunk_idx]。
            - chunk_kblock_mask (np.ndarray): 形状为 [Tchunks, Tc] 的 bool 数组，
              对所有 head 的 activation map 求并集，每行表示对应 chunk
              在任意 head 中需要访问的 kblock 集合。
    """
    assert start_row_indices is not None, "start_row_indices cannot be None"
    assert q_chunk_size % m_block_size == 0, "q_chunk_size must be divisible by m_block_size"

    # 1. 解析输入的起止索引
    LTS, LTE, UTS, UTE = None, None, None, None
    if start_row_indices.shape[-1] == 4:
        LTS, LTE, UTS, UTE = paddle.split(start_row_indices, 4, axis=-1)
        LTS, LTE, UTS, UTE = [t.squeeze(-1) for t in (LTS, LTE, UTS, UTE)]
    elif start_row_indices.shape[-1] == 2:
        LTS, UTE = paddle.split(start_row_indices, 2, axis=-1)
        LTS, UTE = LTS.squeeze(-1), UTE.squeeze(-1)

    # 2. 获取维度信息
    valid_tensor = next(t for t in [LTS, LTE, UTS, UTE] if t is not None)
    B, H, S = valid_tensor.shape

    Tr = S // m_block_size
    Tc = S // n_block_size
    Tchunks = S // q_chunk_size
    assert Tr % Tchunks == 0, "Total row blocks must be divisible by total chunks"
    blocks_per_chunk = Tr // Tchunks

    # 3. scanMaxMin 预处理
    def scan_max_min(tensor):
        if tensor is not None:
            return scanMaxMinChunkedKernel(tensor, n_block_size, B, H, S)
        return None, None

    LTStartMax_gpu, LTStartMin_gpu = scan_max_min(LTS)
    LTEndMax_gpu, LTEndMin_gpu = scan_max_min(LTE)
    UTStartMax_gpu, UTStartMin_gpu = scan_max_min(UTS)
    UTEndMax_gpu, UTEndMin_gpu = scan_max_min(UTE)

    # 4. 使用 CUDA kernel 计算每个 q_block 的工作负载（与原版一致）
    all_indices_max_min = [
        LTStartMax_gpu, LTStartMin_gpu, LTEndMax_gpu, LTEndMin_gpu,
        UTStartMax_gpu, UTStartMin_gpu, UTEndMax_gpu, UTEndMin_gpu
    ]
    workload_per_block = reduce_workload(all_indices_max_min, B, H, Tr, Tc, m_block_size, S)
    # print(workload_per_block)

    # 5. 聚合到 chunk 级别的工作负载，并对 H 维度求和
    #    与 get_q_workload 不同，这里将所有 head 的 workload 累加，
    #    因为后续 assign_tasks_heap 将所有 head*chunk 展平为一个 task 列表
    workload_grouped = workload_per_block.reshape([B, H, Tchunks, blocks_per_chunk, 1])
    workload_per_chunk = paddle.sum(workload_grouped, axis=3).sum(axis=0).sum(axis=0).reshape([1, 1, Tchunks])

    final_res = paddle.zeros([1, 1, Tchunks, 2], dtype='int32')
    final_res[:, :, :, 0] = workload_per_chunk
    final_res[:, :, :, 1] = paddle.arange(0, Tchunks, dtype="int32")

    # 6. Naive 计算 activation map: [B*H, Tr, Tc]
    activation_map = _compute_activation_map_naive(
        LTStartMax_gpu, LTStartMin_gpu, LTEndMax_gpu, LTEndMin_gpu,
        UTStartMax_gpu, UTStartMin_gpu, UTEndMax_gpu, UTEndMin_gpu,
        B, H, Tr, Tc, m_block_size, S
    )

    # 7. 聚合到 chunk 粒度: [B*H, Tchunks, blocks_per_chunk, Tc] -> any -> [B*H, Tchunks, Tc]
    activation_map_chunked = activation_map.reshape(B * H, Tchunks, blocks_per_chunk, Tc).any(axis=2)

    # 8. 对 batch 和 H 维度求并集: [B, H, Tchunks, Tc] -> [Tchunks, Tc]
    activation_map_chunked = activation_map_chunked.reshape(B, H, Tchunks, Tc).any(axis=0).any(axis=0)

    # 9. 展平为 [Tchunks, Tc]，与 workload 的 H 聚合后一致
    chunk_kblock_mask = activation_map_chunked.reshape(Tchunks, Tc)

    return final_res, chunk_kblock_mask


def assign_tasks_ipo(
    tasks: np.ndarray,
    num_buckets: int
) -> Tuple[List[List[Tuple[int, int]]], List[int], int]:
    """
    使用 IPO 最优求解器执行纯计算负载均衡。

    接口与 assign_tasks_heap 一致。当前 IPO kernel 约束为 N <= 512 且
    N 能被 num_buckets 整除；不满足时自动回退到 assign_tasks_heap。
    """
    if not isinstance(tasks, np.ndarray):
        tasks = tasks.cpu().numpy() if hasattr(tasks, 'cpu') else np.asarray(tasks)

    n = len(tasks)
    if n == 0 or n > 512 or n % num_buckets != 0:
        return assign_tasks_heap(tasks, num_buckets)

    tasks = np.asarray(tasks)
    bucket_size = n // num_buckets
    weights = tasks[:, 0].astype(np.int32)

    assign_matrix, _ = cp_balance_ipo_solve(weights, num_buckets)

    buckets = []
    bucket_weights = []
    for bucket_idx in range(num_buckets):
        bucket = []
        bucket_weight = 0
        for task_pos in range(bucket_size):
            row_idx = int(assign_matrix[bucket_idx, task_pos])
            weight = int(tasks[row_idx][0])
            chunk_idx = int(tasks[row_idx][1])
            bucket.append((weight, chunk_idx))
            bucket_weight += weight
        bucket.sort(key=lambda x: x[1])
        buckets.append(bucket)
        bucket_weights.append(bucket_weight)

    all_assigned_indices = sorted([idx for bucket in buckets for _, idx in bucket])
    cuts = sum(
        1 for i in range(1, len(all_assigned_indices))
        if all_assigned_indices[i] != all_assigned_indices[i - 1] + 1
    )

    return buckets, bucket_weights, cuts


def _assign_compute_balance(
    tasks: np.ndarray,
    num_buckets: int,
    use_ipo: bool = False
) -> Tuple[List[List[Tuple[int, int]]], List[int], int]:
    """根据开关选择纯计算负载均衡算法。"""
    if use_ipo:
        return assign_tasks_ipo(tasks, num_buckets)
    return assign_tasks_heap(tasks, num_buckets)


def assign_tasks_heap_with_comm(
    tasks: np.ndarray,
    num_buckets: int,
    kblock_masks: np.ndarray = None,
    comm_penalty: float = 0.0,
    kblocks_per_chunk: int = 0,
    adaptive_comm: bool = False
) -> Tuple[List[List[Tuple[int, int]]], List[int], int]:
    """
    通信感知的贪心任务分配算法。

    在原有小顶堆贪心（仅优化计算均衡）的基础上，引入通信惩罚因子：
    - 对每个待分配的 task，评估所有候选桶的综合得分
    - 综合得分 = 桶当前权重 + λ_b * comm_after(i, b)
    - comm_after = 分配后桶需要但不在本地的 kblock 数

    每个 task (idx) 被分配到桶时，自带其对应位置的 KV 数据，覆盖 kblock 范围
    [idx * kblocks_per_chunk, (idx+1) * kblocks_per_chunk)，这部分不产生通信。

    当 adaptive_comm = True 时，使用自适应通信惩罚（变体 B-思路三）：
        λ_b = comm_penalty * (1 + comm_current(b) / max(comm_avg, 1))
    通信量已经较高的桶获得更大的惩罚因子，从而倾向于将后续 task
    分配到通信量较低的桶，实现通信量的均衡化。

    当 comm_penalty = 0 时，退化为与 assign_tasks_heap 完全相同的行为。

    Args:
        tasks (np.ndarray): 形状为 (N, 2) 的任务数组，每行是 [weight, index]。
        num_buckets (int): 桶的数量（通常等于 cp_size）。
        kblock_masks (np.ndarray): 形状为 (N, Tc) 的 bool 数组，
            kblock_masks[i] 表示第 i 个 task 需要访问的 kblock 集合。
            为 None 时退化为纯计算均衡。
        comm_penalty (float): 通信惩罚基础因子 λ_base。
        kblocks_per_chunk (int): 每个 chunk 包含的 kblock 数 (= chunk_size / n_block_size)。
            用于计算 task 自带的本地 KV kblock 范围。
        adaptive_comm (bool): 是否启用自适应通信惩罚。
            False: 所有桶使用固定 λ = comm_penalty（方案 A）。
            True: λ_b = comm_penalty * (1 + comm(b) / max(comm_avg, 1))（变体 B-思路三）。

    Returns:
        Tuple:
            - buckets: 分配结果。
            - bucket_weights: 每个桶的总计算权重。
            - cuts: 数据切分次数。
    """
    n = len(tasks)
    if n == 0:
        return [[] for _ in range(num_buckets)], [0] * num_buckets, 0

    batch_size = n // num_buckets

    # 按权重降序排序，优先分配最重的任务
    tasks_sorted = sorted(tasks, key=lambda x: -x[0])

    buckets = [[] for _ in range(num_buckets)]
    bucket_weights = [0] * num_buckets
    bucket_counts = [0] * num_buckets

    use_comm = kblock_masks is not None and comm_penalty > 0
    Tc = kblock_masks.shape[1] if use_comm else 0

    # 每个桶维护两个集合：
    #   bucket_needed[b]: 桶内所有 task 需要的 kblock 并集（来自 activation map）
    #   bucket_local[b]:  桶内所有 task 自带的 kblock 并集（来自 task 的 chunk 位置）
    bucket_needed = [np.zeros(Tc, dtype=bool) for _ in range(num_buckets)] if use_comm else None
    bucket_local = [np.zeros(Tc, dtype=bool) for _ in range(num_buckets)] if use_comm else None
    # 缓存每个桶的当前通信量，避免重复计算
    bucket_comm = [0] * num_buckets if use_comm else None

    def _make_self_kblock_mask(idx):
        """构造 task idx 自带的 KV kblock 掩码。"""
        mask = np.zeros(Tc, dtype=bool)
        kb_start = idx * kblocks_per_chunk
        kb_end = min((idx + 1) * kblocks_per_chunk, Tc)
        if kb_start < Tc:
            mask[kb_start:kb_end] = True
        return mask

    # 保留原始行号用于索引 kblock_masks
    tasks_with_orig_idx = sorted(enumerate(tasks), key=lambda x: -x[1][0])

    for orig_row_idx, (weight, idx) in tasks_with_orig_idx:
        weight = int(weight)
        idx = int(idx)

        if not use_comm:
            # 无通信惩罚：直接找最轻的未满桶（与原版一致）
            best_bi = -1
            best_weight = float('inf')
            for bi in range(num_buckets):
                if bucket_counts[bi] < batch_size and bucket_weights[bi] < best_weight:
                    best_weight = bucket_weights[bi]
                    best_bi = bi
            if best_bi == -1:
                best_bi = min(range(num_buckets), key=lambda bi: bucket_weights[bi])
        else:
            # 通信感知分配：综合评估计算负载 + 通信代价
            task_need_mask = kblock_masks[orig_row_idx]
            task_self_mask = _make_self_kblock_mask(idx)

            # 计算自适应 λ 所需的各桶通信量均值
            if adaptive_comm:
                comm_avg = max(sum(bucket_comm) / num_buckets, 1.0)

            best_bi = -1
            best_score = float('inf')

            candidates = range(num_buckets)
            # 先尝试未满的桶，若全满则尝试所有桶
            for attempt in range(2):
                for bi in candidates:
                    if attempt == 0 and bucket_counts[bi] >= batch_size:
                        continue
                    # 分配后的通信量 = 需要的 kblock 中不在本地的数量
                    needed_after = bucket_needed[bi] | task_need_mask
                    local_after = bucket_local[bi] | task_self_mask
                    comm_after = int(np.count_nonzero(needed_after & ~local_after))

                    # 计算该桶的惩罚因子
                    if adaptive_comm:
                        # 变体 B-思路三：自适应 λ
                        # 通信量高于平均的桶获得更大惩罚，推动通信均衡
                        lambda_b = comm_penalty * (1.0 + bucket_comm[bi] / comm_avg)
                    else:
                        # 方案 A：固定 λ
                        lambda_b = comm_penalty

                    score = bucket_weights[bi] + lambda_b * comm_after
                    if score < best_score:
                        best_score = score
                        best_bi = bi

                if best_bi != -1:
                    break

        # 执行分配
        buckets[best_bi].append((weight, idx))
        bucket_weights[best_bi] += weight
        bucket_counts[best_bi] += 1
        if use_comm:
            bucket_needed[best_bi] |= kblock_masks[orig_row_idx]
            bucket_local[best_bi] |= _make_self_kblock_mask(idx)
            # 更新缓存的通信量
            bucket_comm[best_bi] = int(np.count_nonzero(
                bucket_needed[best_bi] & ~bucket_local[best_bi]
            ))

    # 桶内按原始索引排序
    for i in range(num_buckets):
        buckets[i] = sorted(buckets[i], key=lambda x: x[1])

    # 统计切分次数
    all_assigned_indices = sorted([idx for bucket in buckets for _, idx in bucket])
    cuts = sum(
        1 for i in range(1, len(all_assigned_indices))
        if all_assigned_indices[i] != all_assigned_indices[i - 1] + 1
    )

    return buckets, bucket_weights, cuts


def balance_flashmask_input_comm(
    startend_row_indices: paddle.Tensor,
    cp_size: int,
    cp_rank: int,
    balance_chunk_size: int = 2048,
    q_block_size: int = 128,
    k_block_size: int = 128,
    comm_penalty: float = 0.1,
    adaptive_comm: bool = False,
    use_ipo: bool = False
) -> Tuple[paddle.Tensor, List[List[Tuple[int, int]]]]:
    """
    通信感知的 FlashMask 输入数据负载均衡主流程。

    与原 balance_flashmask_input 相比，唯一的改变是：
    1. 额外计算 kblock activation map
    2. 使用 assign_tasks_heap_with_comm 进行联合优化分配
    其余数据重排、通信流程完全不变。

    Args:
        startend_row_indices (paddle.Tensor): 稀疏 attention 的原始起止索引。
        cp_size (int): 通信组大小。
        cp_rank (int): 当前进程的 rank。
        balance_chunk_size (int): 负载均衡分析和数据移动的块大小。
        q_block_size (int): FlashAttention kernel 的 query 块大小。
        k_block_size (int): FlashAttention kernel 的 key 块大小。
        comm_penalty (float): 通信惩罚基础因子 λ_base。
            0.0 = 纯计算均衡（与原版行为一致）。
        adaptive_comm (bool): 是否启用自适应通信惩罚。
            False: 固定 λ，最小化总通信量（方案 A）。
            True: 自适应 λ_b，推动通信量均衡化（变体 B-思路三）。
        use_ipo (bool): comm_penalty <= 0 时，是否使用 IPO 执行纯计算负载均衡。

    Returns:
        Tuple:
            - local_startend_row_indices (paddle.Tensor): 经过负载均衡重排后的局部索引。
            - buckets (List): 全局任务分配方案。
    """
    # 步骤 1: 估算工作负载 + 计算 activation map
    paddle.base.core.nvprof_nvtx_push("get_q_workload_with_activation_map")
    workload, chunk_kblock_mask = get_q_workload_with_activation_map(
        startend_row_indices, balance_chunk_size, q_block_size, k_block_size
    )
    paddle.base.core.nvprof_nvtx_pop()

    # 步骤 2: 通信感知的任务分配；无通信惩罚时可切换到 IPO 纯计算均衡
    assign_nvtx_name = "assign_tasks_ipo" if use_ipo and comm_penalty <= 0 else "assign_tasks_heap_with_comm"
    paddle.base.core.nvprof_nvtx_push(assign_nvtx_name)
    tasks_np = workload.reshape([-1, 2]).cpu().numpy()
    kblocks_per_chunk = balance_chunk_size // k_block_size
    if use_ipo and comm_penalty <= 0:
        buckets, _, _ = _assign_compute_balance(tasks_np, cp_size, use_ipo=True)
    else:
        buckets, _, _ = assign_tasks_heap_with_comm(
            tasks_np, cp_size,
            kblock_masks=chunk_kblock_mask,
            comm_penalty=comm_penalty,
            kblocks_per_chunk=kblocks_per_chunk,
            adaptive_comm=adaptive_comm
        )
    paddle.base.core.nvprof_nvtx_pop()

    # 步骤 3: 根据分配方案对索引进行重排（与原版完全一致）
    paddle.base.core.nvprof_nvtx_push("startend_row_indices_rerank")
    rerank_indices = np.array([idx for bucket in buckets for _, idx in bucket], dtype=np.int32)
    indices_tensor = paddle.to_tensor(rerank_indices, dtype='int32', place=startend_row_indices.place)
    startend_row_indices_rerank = indices_rerank_cuda(startend_row_indices, indices_tensor)
    paddle.base.core.nvprof_nvtx_pop()

    # 步骤 4: 计算当前 rank 的局部索引（与原版完全一致）
    paddle.base.core.nvprof_nvtx_push("indices_to_chunks")
    local_bucket_indices = [x[1] for x in buckets[cp_rank]]
    local_indices_tensor = paddle.to_tensor(
        local_bucket_indices, dtype='int32', place=startend_row_indices.place
    )
    local_startend_row_indices = indices_to_chunks_cuda(
        startend_row_indices_rerank, local_indices_tensor, balance_chunk_size
    )
    paddle.base.core.nvprof_nvtx_pop()

    return local_startend_row_indices, buckets


# ---------------------------------------------------------------------------
# 机间通信感知的负载均衡（两阶段法）
#
# 阶段一：使用 assign_tasks_heap（纯计算均衡）得到基础 Buckets 分配。
# 阶段二：以机器为单位（每 buckets_per_machine 个桶为一组），通过 task 交换
#          降低机间通信瓶颈。对应 comm_aware_balance.md 变体 B-思路一。
# ---------------------------------------------------------------------------


def _build_bucket_orig_row_map(
    buckets: List[List[Tuple[int, int]]],
    tasks_np: np.ndarray
) -> List[List[int]]:
    """
    为 phase 1 分配结果建立每个 bucket entry 到 kblock_masks 原始行号的映射。

    assign_tasks_heap 输出的 bucket 中存储 (weight, chunk_idx)，在多 head 场景下
    同一个 chunk_idx 可能对应多个原始行号（不同 head）。此函数通过 (weight, idx)
    作为 key 进行匹配，并按消耗顺序去重。

    Args:
        buckets: phase 1 的分配结果。
        tasks_np: 原始任务数组 [N, 2]，每行 [weight, idx]。

    Returns:
        List[List[int]]: 与 buckets 同构的嵌套列表，存储原始行号。
    """
    key_to_orig_rows = defaultdict(list)
    for orig_row in range(len(tasks_np)):
        w = int(tasks_np[orig_row][0])
        idx = int(tasks_np[orig_row][1])
        key_to_orig_rows[(w, idx)].append(orig_row)

    bucket_orig_rows = []
    for bucket in buckets:
        orig_rows = []
        for w, idx in bucket:
            key = (int(w), int(idx))
            orig_rows.append(key_to_orig_rows[key].pop(0))
        bucket_orig_rows.append(orig_rows)

    return bucket_orig_rows


def _compute_single_machine_comm(
    buckets: List[List[Tuple[int, int]]],
    bucket_orig_rows: List[List[int]],
    kblock_masks: np.ndarray,
    kblocks_per_chunk: int,
    Tc: int,
    machine_idx: int,
    buckets_per_machine: int,
    num_buckets: int
) -> int:
    """
    计算单台机器的机间通信量。

    机间通信量 = 机器内所有 task 需要的 kblock 中，不在机器内任何 task
    自带 KV 数据中的部分（即需要从其他机器获取的 kblock 数量）。

    Args:
        buckets: 所有桶的分配结果。
        bucket_orig_rows: 各桶内 task 到 kblock_masks 原始行号的映射。
        kblock_masks: [N, Tc] 的 bool 数组。
        kblocks_per_chunk: 每个 chunk 包含的 kblock 数。
        Tc: 总 kblock 数。
        machine_idx: 机器索引。
        buckets_per_machine: 每台机器的桶数。
        num_buckets: 桶总数。

    Returns:
        int: 该机器的机间通信量（需要从其他机器获取的 kblock 数）。
    """
    bi_start = machine_idx * buckets_per_machine
    bi_end = min(bi_start + buckets_per_machine, num_buckets)

    # 收集机器内所有 task 的原始行号和 chunk idx
    all_orig_rows = []
    all_chunk_indices = []
    for bi in range(bi_start, bi_end):
        for task_pos in range(len(buckets[bi])):
            all_orig_rows.append(bucket_orig_rows[bi][task_pos])
            all_chunk_indices.append(int(buckets[bi][task_pos][1]))

    if not all_orig_rows:
        return 0

    # needed: 所有 task 需要的 kblock 并集（利用 numpy fancy indexing 一次完成）
    needed = kblock_masks[all_orig_rows].any(axis=0)

    # local: 所有 task 自带的 kblock 并集
    local_kblocks = np.zeros(Tc, dtype=bool)
    for idx in all_chunk_indices:
        kb_start = idx * kblocks_per_chunk
        kb_end = min(kb_start + kblocks_per_chunk, Tc)
        if kb_start < Tc:
            local_kblocks[kb_start:kb_end] = True

    return int(np.count_nonzero(needed & ~local_kblocks))


def _compute_machine_idx_dispersion(
    buckets: List[List[Tuple[int, int]]],
    machine_idx: int,
    buckets_per_machine: int,
    num_buckets: int
) -> int:
    """
    计算单台机器内 task chunk_idx 的离散度（dispersion）。

    离散度定义为: (max_idx - min_idx + 1) - count
    即 idx 覆盖范围内的 "空洞" 数量。值越小表示连续性越好，0 为完全连续。

    如果一台机器只有 0 或 1 个 task，离散度为 0。

    Args:
        buckets: 所有桶的分配结果。
        machine_idx: 机器索引。
        buckets_per_machine: 每台机器的桶数。
        num_buckets: 桶总数。

    Returns:
        int: 该机器的 idx 离散度。
    """
    bi_start = machine_idx * buckets_per_machine
    bi_end = min(bi_start + buckets_per_machine, num_buckets)

    indices = []
    for bi in range(bi_start, bi_end):
        for _, idx in buckets[bi]:
            indices.append(int(idx))

    if len(indices) <= 1:
        return 0

    return (max(indices) - min(indices) + 1) - len(indices)


def _locality_swap_phase(
    buckets: List[List[Tuple[int, int]]],
    bucket_weights: List[int],
    bucket_orig_rows: List[List[int]],
    buckets_per_machine: int,
    num_buckets: int,
    W_avg: float,
    epsilon: float = 0.05,
    max_iterations: int = 100
) -> None:
    """
    局部性交换阶段：以 chunk_idx 连续性为优化目标的机间 task 交换。

    在保持负载均衡约束的前提下，通过机间交换 task 来最小化各机器内
    task chunk_idx 的离散度（即最大化 idx 连续性）。

    算法流程:
      repeat until converged:
        1. 计算所有机器的 idx 离散度
        2. 找到离散度最大的机器 mi_worst
        3. 枚举 mi_worst 与每台其他机器的交换对
        4. 若交换后:
             - 两个涉及桶的权重偏离 W_avg 不超过 ε（计算仍均衡）
             - 全局离散度总和严格下降
           则选择全局离散度降幅最大的交换
        5. 执行最优交换，更新状态

    Args:
        buckets: 分配结果（会被原地修改）。
        bucket_weights: 每个桶的总权重（会被原地修改）。
        bucket_orig_rows: 各桶内 task 到原始行号的映射（会被原地修改）。
        buckets_per_machine: 每台机器的桶数。
        num_buckets: 桶总数。
        W_avg: 桶平均权重。
        epsilon: 计算均衡容忍度。
        max_iterations: 最大交换迭代次数。
    """
    num_machines = (num_buckets + buckets_per_machine - 1) // buckets_per_machine
    if num_machines <= 1:
        return

    for iteration in range(max_iterations):
        # 计算每台机器的 idx 离散度
        dispersions = [
            _compute_machine_idx_dispersion(
                buckets, mi, buckets_per_machine, num_buckets
            )
            for mi in range(num_machines)
        ]

        total_dispersion = sum(dispersions)
        if total_dispersion == 0:
            # 所有机器的 idx 已完全连续
            break

        # 找到离散度最大的机器
        mi_worst = int(np.argmax(dispersions))
        if dispersions[mi_worst] == 0:
            break

        bi_worst_start = mi_worst * buckets_per_machine
        bi_worst_end = min(bi_worst_start + buckets_per_machine, num_buckets)

        best_swap = None
        best_dispersion_reduction = 0

        # 枚举 mi_worst 与每台其他机器的交换对
        for mi_other in range(num_machines):
            if mi_other == mi_worst:
                continue

            bi_other_start = mi_other * buckets_per_machine
            bi_other_end = min(bi_other_start + buckets_per_machine, num_buckets)

            for bi_a in range(bi_worst_start, bi_worst_end):
                for ti_a in range(len(buckets[bi_a])):
                    w_a, idx_a = buckets[bi_a][ti_a]
                    orig_a = bucket_orig_rows[bi_a][ti_a]

                    for bi_b in range(bi_other_start, bi_other_end):
                        for ti_b in range(len(buckets[bi_b])):
                            w_b, idx_b = buckets[bi_b][ti_b]
                            orig_b = bucket_orig_rows[bi_b][ti_b]

                            # 检查计算均衡约束
                            new_w_a = bucket_weights[bi_a] - int(w_a) + int(w_b)
                            new_w_b = bucket_weights[bi_b] - int(w_b) + int(w_a)
                            if abs(new_w_a - W_avg) > epsilon * W_avg:
                                continue
                            if abs(new_w_b - W_avg) > epsilon * W_avg:
                                continue

                            # 模拟交换
                            buckets[bi_a][ti_a] = (w_b, idx_b)
                            buckets[bi_b][ti_b] = (w_a, idx_a)

                            # 重算受影响的两台机器的离散度
                            new_disp_worst = _compute_machine_idx_dispersion(
                                buckets, mi_worst, buckets_per_machine, num_buckets
                            )
                            new_disp_other = _compute_machine_idx_dispersion(
                                buckets, mi_other, buckets_per_machine, num_buckets
                            )

                            # 撤销交换
                            buckets[bi_a][ti_a] = (w_a, idx_a)
                            buckets[bi_b][ti_b] = (w_b, idx_b)

                            # 计算离散度降幅
                            old_sum = dispersions[mi_worst] + dispersions[mi_other]
                            new_sum = new_disp_worst + new_disp_other
                            reduction = old_sum - new_sum

                            if reduction > best_dispersion_reduction:
                                best_dispersion_reduction = reduction
                                best_swap = (bi_a, ti_a, bi_b, ti_b)

        if best_swap is None or best_dispersion_reduction <= 0:
            break

        # 执行最优交换
        bi_a, ti_a, bi_b, ti_b = best_swap
        w_a, idx_a = buckets[bi_a][ti_a]
        w_b, idx_b = buckets[bi_b][ti_b]
        orig_a = bucket_orig_rows[bi_a][ti_a]
        orig_b = bucket_orig_rows[bi_b][ti_b]

        buckets[bi_a][ti_a] = (w_b, idx_b)
        buckets[bi_b][ti_b] = (w_a, idx_a)
        bucket_orig_rows[bi_a][ti_a] = orig_b
        bucket_orig_rows[bi_b][ti_b] = orig_a
        bucket_weights[bi_a] = bucket_weights[bi_a] - int(w_a) + int(w_b)
        bucket_weights[bi_b] = bucket_weights[bi_b] - int(w_b) + int(w_a)


def _intra_machine_balance_phase(
    buckets: List[List[Tuple[int, int]]],
    bucket_weights: List[int],
    bucket_orig_rows: List[List[int]],
    buckets_per_machine: int,
    num_buckets: int,
    use_ipo: bool = False,
) -> None:
    """
    机内负载均衡阶段：在每台机器内部，使用纯计算量指标重新分配 task，
    使同一机器内各桶的计算负载尽可能均衡。

    该阶段在 _locality_swap_phase 之后执行，不改变机器间的 task 归属，
    仅在每台机器内部重新分配 task 到各桶中。

    算法流程:
      对每台机器:
        1. 收集该机器内所有桶的全部 task
        2. 按 use_ipo 开关选择 IPO 或 assign_tasks_heap 重新分配到机器内各桶
        3. 原地更新 buckets, bucket_weights, bucket_orig_rows

    Args:
        buckets: 分配结果（会被原地修改）。
        bucket_weights: 每个桶的总权重（会被原地修改）。
        bucket_orig_rows: 各桶内 task 到原始行号的映射（会被原地修改）。
        buckets_per_machine: 每台机器的桶数。
        num_buckets: 桶总数。
        use_ipo: 是否使用 IPO 进行机内纯计算负载均衡。
    """
    num_machines = (num_buckets + buckets_per_machine - 1) // buckets_per_machine
    if num_machines <= 0:
        return

    for mi in range(num_machines):
        bi_start = mi * buckets_per_machine
        bi_end = min(bi_start + buckets_per_machine, num_buckets)
        local_num_buckets = bi_end - bi_start

        if local_num_buckets <= 1:
            continue

        # 1. 收集该机器内所有 task 及其 orig_row 映射
        all_tasks = []       # [(weight, idx), ...]
        all_orig_rows = []   # [orig_row, ...]
        for bi in range(bi_start, bi_end):
            for ti in range(len(buckets[bi])):
                all_tasks.append(buckets[bi][ti])
                all_orig_rows.append(bucket_orig_rows[bi][ti])

        if not all_tasks:
            continue

        # 2. 构造 tasks_np 并按开关选择纯计算负载均衡算法
        tasks_np = np.array(all_tasks, dtype=np.int64)
        new_buckets, new_bucket_weights, _ = _assign_compute_balance(
            tasks_np, local_num_buckets, use_ipo=use_ipo
        )

        # 3. 建立 (weight, idx) -> orig_row 的映射，用于恢复 orig_row 信息
        #    同一个 (weight, idx) 可能出现多次（多 head），按收集顺序消耗
        key_to_orig_rows = defaultdict(list)
        for task, orig_row in zip(all_tasks, all_orig_rows):
            key_to_orig_rows[(int(task[0]), int(task[1]))].append(orig_row)

        # 4. 原地更新 buckets, bucket_weights, bucket_orig_rows
        for local_bi in range(local_num_buckets):
            global_bi = bi_start + local_bi
            new_bucket_tasks = new_buckets[local_bi]
            new_orig_rows = []
            for w, idx in new_bucket_tasks:
                key = (int(w), int(idx))
                new_orig_rows.append(key_to_orig_rows[key].pop(0))

            buckets[global_bi] = new_bucket_tasks
            bucket_weights[global_bi] = new_bucket_weights[local_bi]
            bucket_orig_rows[global_bi] = new_orig_rows


def _print_bucket_summary(phase_name: str, buckets: List[List[Tuple[int, int]]], bucket_weights: List[int]) -> None:
    """
    打印每个阶段后各 rank（桶）的 task idx 列表和计算量。

    Args:
        phase_name: 阶段名称。
        buckets: 当前的分配结果。
        bucket_weights: 每个桶的总权重。
    """
    print(f"\n  === {phase_name} ===")
    for bi in range(len(buckets)):
        task_details = [(int(idx), int(w)) for w, idx in buckets[bi]]
        idx_list = [d[0] for d in task_details]
        w_list = [d[1] for d in task_details]
        print(f"    rank {bi}: indices={idx_list}, weights={w_list}, total={bucket_weights[bi]}")
    weights = bucket_weights
    if weights:
        w_max = max(weights)
        w_min = min(weights)
        w_avg = sum(weights) / len(weights)
        imbalance = (w_max - w_min) / w_avg * 100 if w_avg > 0 else 0
        print(f"    [summary] max={w_max}, min={w_min}, avg={w_avg:.1f}, imbalance={imbalance:.2f}%")


def _inter_machine_two_phase_swap(
    buckets: List[List[Tuple[int, int]]],
    bucket_weights: List[int],
    bucket_orig_rows: List[List[int]],
    kblock_masks: np.ndarray,
    kblocks_per_chunk: int,
    Tc: int,
    buckets_per_machine: int,
    epsilon: float = 0.05,
    max_iterations: int = 100,
    use_locality_swap: bool = False,
    max_locality_iterations: int = 100,
    use_intra_machine_balance: bool = False,
    use_ipo: bool = False
) -> Tuple[List[List[Tuple[int, int]]], List[int]]:
    """
    机间交换优化阶段，包含三个可选交换策略：

    策略 1（始终启用）—— 通信均衡交换：
      以机器为单位进行 task 交换，降低机间通信瓶颈。
      算法流程（对应 comm_aware_balance.md 变体 B-思路一）：
        repeat until converged:
          1. 计算每台机器的机间通信量
          2. 找到通信量最大的机器 mi_max 和最小的机器 mi_min
          3. 枚举 (task_a ∈ mi_max, task_b ∈ mi_min) 的交换对
          4. 若交换满足计算均衡约束且降低通信瓶颈，选择最优交换
          5. 执行交换

    策略 2（use_locality_swap=True 时启用）—— idx 连续性交换：
      在策略 1 之后执行，以 chunk_idx 连续性为优化目标，通过机间交换
      最大化每台机器内 task 的 idx 连续性。
      核心观察: mask 具有空间局部性，chunk_idx 相近的 task 需要的 kblock
      范围也相近。同一机器内 task idx 越连续，自带 KV 数据覆盖率越高，
      机间通信越少。
      算法流程:
        repeat until converged:
          1. 计算每台机器的 idx 离散度 (max_idx - min_idx + 1 - count)
          2. 找到离散度最大的机器 mi_worst
          3. 枚举 mi_worst 与每台其他机器的交换对
          4. 若交换满足计算均衡约束且降低全局离散度总和，选择最优交换
          5. 执行交换

    策略 3（use_intra_machine_balance=True 时启用）—— 机内负载均衡：
      在策略 2 之后执行，以每台机器为单位，使用纯计算量指标的小顶堆贪心
      算法（assign_tasks_heap）重新分配机器内各桶的 task，使机内各桶的
      计算负载尽可能均衡。不改变机器间的 task 归属。

    保证：
    - 桶容量不变（一换一，不改变每个桶的 task 数量）
    - 每个策略的优化指标严格递减（否则终止）

    Args:
        buckets: phase 1 的分配结果。
        bucket_weights: 每个桶的总权重。
        bucket_orig_rows: 各桶内 task 到 kblock_masks 行号的映射。
        kblock_masks: [N, Tc] bool 数组。
        kblocks_per_chunk: 每个 chunk 的 kblock 数。
        Tc: 总 kblock 数。
        buckets_per_machine: 每台机器的桶数。
        epsilon: 计算均衡容忍度（如 0.05 表示允许 5% 偏差）。
        max_iterations: 通信均衡交换的最大迭代次数。
        use_locality_swap: 是否在通信交换后追加 idx 连续性交换。
        max_locality_iterations: idx 连续性交换的最大迭代次数。
        use_intra_machine_balance: 是否在 locality swap 之后追加机内负载均衡。
        use_ipo: 是否在纯计算负载均衡阶段使用 IPO。

    Returns:
        Tuple: (优化后的 buckets, 优化后的 bucket_weights)
    """
    num_buckets = len(buckets)
    num_machines = (num_buckets + buckets_per_machine - 1) // buckets_per_machine

    if num_machines <= 1:
        return buckets, bucket_weights

    total_weight = sum(bucket_weights)
    W_avg = total_weight / num_buckets

    # ---- 策略 1: 通信均衡交换 ----
    for iteration in range(max_iterations):
        # 计算所有机器的机间通信量
        machine_comm = [
            _compute_single_machine_comm(
                buckets, bucket_orig_rows, kblock_masks,
                kblocks_per_chunk, Tc, mi, buckets_per_machine, num_buckets
            )
            for mi in range(num_machines)
        ]

        mi_max = int(np.argmax(machine_comm))
        mi_min = int(np.argmin(machine_comm))

        # 若最大和最小相同或通信量相等，已收敛
        if mi_max == mi_min or machine_comm[mi_max] <= machine_comm[mi_min]:
            break

        current_bottleneck = machine_comm[mi_max]

        # 在 mi_max 和 mi_min 的所有桶间枚举交换对
        best_swap = None
        best_new_bottleneck = current_bottleneck

        bi_max_start = mi_max * buckets_per_machine
        bi_max_end = min(bi_max_start + buckets_per_machine, num_buckets)
        bi_min_start = mi_min * buckets_per_machine
        bi_min_end = min(bi_min_start + buckets_per_machine, num_buckets)

        for bi_a in range(bi_max_start, bi_max_end):
            for ti_a in range(len(buckets[bi_a])):
                w_a, idx_a = buckets[bi_a][ti_a]
                orig_a = bucket_orig_rows[bi_a][ti_a]

                for bi_b in range(bi_min_start, bi_min_end):
                    for ti_b in range(len(buckets[bi_b])):
                        w_b, idx_b = buckets[bi_b][ti_b]
                        orig_b = bucket_orig_rows[bi_b][ti_b]

                        # 检查计算均衡约束
                        new_w_a = bucket_weights[bi_a] - int(w_a) + int(w_b)
                        new_w_b = bucket_weights[bi_b] - int(w_b) + int(w_a)
                        if abs(new_w_a - W_avg) > epsilon * W_avg:
                            continue
                        if abs(new_w_b - W_avg) > epsilon * W_avg:
                            continue

                        # 模拟交换：临时修改数据结构
                        buckets[bi_a][ti_a] = (w_b, idx_b)
                        buckets[bi_b][ti_b] = (w_a, idx_a)
                        bucket_orig_rows[bi_a][ti_a] = orig_b
                        bucket_orig_rows[bi_b][ti_b] = orig_a

                        # 仅重算受影响的两台机器的通信量
                        new_comm_max = _compute_single_machine_comm(
                            buckets, bucket_orig_rows, kblock_masks,
                            kblocks_per_chunk, Tc, mi_max,
                            buckets_per_machine, num_buckets
                        )
                        new_comm_min = _compute_single_machine_comm(
                            buckets, bucket_orig_rows, kblock_masks,
                            kblocks_per_chunk, Tc, mi_min,
                            buckets_per_machine, num_buckets
                        )
                        new_bottleneck = max(new_comm_max, new_comm_min)

                        # 撤销交换
                        buckets[bi_a][ti_a] = (w_a, idx_a)
                        buckets[bi_b][ti_b] = (w_b, idx_b)
                        bucket_orig_rows[bi_a][ti_a] = orig_a
                        bucket_orig_rows[bi_b][ti_b] = orig_b

                        if new_bottleneck < best_new_bottleneck:
                            best_new_bottleneck = new_bottleneck
                            best_swap = (bi_a, ti_a, bi_b, ti_b)

        if best_swap is None:
            break

        # 执行最优交换
        bi_a, ti_a, bi_b, ti_b = best_swap
        w_a, idx_a = buckets[bi_a][ti_a]
        w_b, idx_b = buckets[bi_b][ti_b]
        orig_a = bucket_orig_rows[bi_a][ti_a]
        orig_b = bucket_orig_rows[bi_b][ti_b]

        buckets[bi_a][ti_a] = (w_b, idx_b)
        buckets[bi_b][ti_b] = (w_a, idx_a)
        bucket_orig_rows[bi_a][ti_a] = orig_b
        bucket_orig_rows[bi_b][ti_b] = orig_a
        bucket_weights[bi_a] = bucket_weights[bi_a] - int(w_a) + int(w_b)
        bucket_weights[bi_b] = bucket_weights[bi_b] - int(w_b) + int(w_a)

    # _print_bucket_summary("After comm-aware swap (策略1)", buckets, bucket_weights)

    # ---- 策略 2: idx 连续性交换（可选） ----
    if use_locality_swap:
        _locality_swap_phase(
            buckets, bucket_weights, bucket_orig_rows,
            buckets_per_machine, num_buckets, W_avg,
            epsilon, max_locality_iterations
        )
        # _print_bucket_summary("After locality swap (策略2)", buckets, bucket_weights)

    # ---- 策略 3: 机内负载均衡（可选） ----
    if use_intra_machine_balance:
        _intra_machine_balance_phase(
            buckets, bucket_weights, bucket_orig_rows,
            buckets_per_machine, num_buckets, use_ipo=use_ipo
        )
        # _print_bucket_summary("After intra-machine balance (策略3)", buckets, bucket_weights)

    # 桶内按 chunk_idx 排序（与原有实现保持一致）
    for i in range(num_buckets):
        combined = list(zip(buckets[i], bucket_orig_rows[i]))
        combined.sort(key=lambda x: x[0][1])
        buckets[i] = [x[0] for x in combined]
        bucket_orig_rows[i] = [x[1] for x in combined]

    return buckets, bucket_weights


def balance_flashmask_input_inter_machine(
    startend_row_indices: paddle.Tensor,
    cp_size: int,
    cp_rank: int,
    balance_chunk_size: int = 2048,
    q_block_size: int = 128,
    k_block_size: int = 128,
    buckets_per_machine: int = 8,
    epsilon: float = 0.05,
    max_swap_iterations: int = 100,
    use_locality_swap: bool = False,
    max_locality_iterations: int = 100,
    use_intra_machine_balance: bool = False,
    use_ipo: bool = False
) -> Tuple[paddle.Tensor, List[List[Tuple[int, int]]]]:
    """
    机间通信感知的 FlashMask 负载均衡主流程。

    阶段一：按 use_ipo 开关选择 IPO 或纯计算均衡贪心得到基础分配。
    阶段二：以机器为单位（每 buckets_per_machine 个桶为一组），通过 task 交换
            降低机间通信瓶颈（comm_aware_balance.md 变体 B-思路一）。
    阶段三（可选）：以 idx 连续性为优化目标的机间 task 交换。
            利用 mask 空间局部性，在保持计算均衡的前提下最大化每台机器内
            task chunk_idx 的连续性，进一步降低机间通信量。
    阶段四（可选）：机内负载均衡。在每台机器内部，按 use_ipo 开关选择
            IPO 或纯计算均衡贪心重新分配机器内各桶的 task，使机内各桶的
            计算负载尽可能均衡。不改变机器间的 task 归属。

    Args:
        startend_row_indices (paddle.Tensor): 稀疏 attention 的原始起止索引。
        cp_size (int): 通信组大小（= 总桶数）。
        cp_rank (int): 当前进程的 rank。
        balance_chunk_size (int): 负载均衡的 chunk 大小。
        q_block_size (int): FlashAttention query 块大小。
        k_block_size (int): FlashAttention key 块大小。
        buckets_per_machine (int): 每台机器包含的桶数（机间调度粒度）。
            例如 8 表示 buckets [0,8) 为机器 0，[8,16) 为机器 1，依此类推。
        epsilon (float): 计算均衡容忍度。交换后每个桶的权重偏离平均值不超过此比例。
        max_swap_iterations (int): 通信均衡交换的最大迭代次数。
        use_locality_swap (bool): 是否在通信交换后追加 idx 连续性交换。
            False（默认）: 仅执行通信均衡交换。
            True: 在通信交换后，额外执行 idx 连续性交换，通过机间交换使每台
            机器内 task 的 chunk_idx 尽可能连续，从而利用 mask 空间局部性
            进一步降低机间通信。
        max_locality_iterations (int): idx 连续性交换的最大迭代次数。
        use_intra_machine_balance (bool): 是否在 locality swap 之后追加机内负载均衡。
            False（默认）: 不执行。
            True: 在每台机器内部，按 use_ipo 开关选择 IPO 或 assign_tasks_heap
            重新分配 task 到机器内各桶，使机内计算负载尽可能均衡。
        use_ipo (bool): 是否在纯计算负载均衡阶段使用 IPO。

    Returns:
        Tuple:
            - local_startend_row_indices (paddle.Tensor): 经过负载均衡重排后的局部索引。
            - buckets (List): 全局任务分配方案。
    """
    # 阶段 0: 计算工作负载 + activation map
    paddle.base.core.nvprof_nvtx_push("get_q_workload_with_activation_map")
    workload, chunk_kblock_mask = get_q_workload_with_activation_map(
        startend_row_indices, balance_chunk_size, q_block_size, k_block_size
    )
    paddle.base.core.nvprof_nvtx_pop()

    tasks_np = workload.reshape([-1, 2]).cpu().numpy()
    kblocks_per_chunk = balance_chunk_size // k_block_size
    Tc = chunk_kblock_mask.shape[1]

    # 阶段 1: 纯计算负载均衡分配
    phase1_nvtx_name = "assign_tasks_ipo_phase1" if use_ipo else "assign_tasks_heap_phase1"
    paddle.base.core.nvprof_nvtx_push(phase1_nvtx_name)
    buckets, bucket_weights, _ = _assign_compute_balance(tasks_np, cp_size, use_ipo=use_ipo)
    paddle.base.core.nvprof_nvtx_pop()

    # _print_bucket_summary("After compute balance (阶段1: 纯计算均衡)", buckets, bucket_weights)

    # 阶段 2: 机间通信均衡交换
    paddle.base.core.nvprof_nvtx_push("inter_machine_swap_phase2")
    bucket_orig_rows = _build_bucket_orig_row_map(buckets, tasks_np)
    buckets, bucket_weights = _inter_machine_two_phase_swap(
        buckets, bucket_weights, bucket_orig_rows,
        chunk_kblock_mask, kblocks_per_chunk, Tc,
        buckets_per_machine, epsilon, max_swap_iterations,
        use_locality_swap=use_locality_swap,
        max_locality_iterations=max_locality_iterations,
        use_intra_machine_balance=use_intra_machine_balance,
        use_ipo=use_ipo
    )
    paddle.base.core.nvprof_nvtx_pop()

    # 阶段 3: 数据重排（与原版完全一致）
    paddle.base.core.nvprof_nvtx_push("startend_row_indices_rerank")
    rerank_indices = np.array(
        [idx for bucket in buckets for _, idx in bucket], dtype=np.int32
    )
    indices_tensor = paddle.to_tensor(
        rerank_indices, dtype='int32', place=startend_row_indices.place
    )
    if(cp_rank == 0):
        print("before rerank",startend_row_indices)
    startend_row_indices_rerank = indices_rerank_cuda(
        startend_row_indices, indices_tensor
    )
    if(cp_rank == 0):
        print("after rerank",startend_row_indices_rerank)
    paddle.base.core.nvprof_nvtx_pop()

    # 阶段 4: 计算当前 rank 的局部索引（与原版完全一致）
    paddle.base.core.nvprof_nvtx_push("indices_to_chunks")
    local_bucket_indices = [x[1] for x in buckets[cp_rank]]
    local_indices_tensor = paddle.to_tensor(
        local_bucket_indices, dtype='int32', place=startend_row_indices.place
    )
    local_startend_row_indices = indices_to_chunks_cuda(
        startend_row_indices_rerank, local_indices_tensor, balance_chunk_size
    )
    if(cp_rank == 0):
        print("after indices_to_chunks", local_startend_row_indices)
    print("max,min,cp_rank",local_startend_row_indices.max(),local_startend_row_indices.min(),cp_rank)
    paddle.base.core.nvprof_nvtx_pop()

    return local_startend_row_indices, buckets
