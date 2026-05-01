"""
Unit tests for CUDA custom ops defined in cp_balance_utils.cu.

Covered operators:
  - scan_max_min
  - reduce_workload
  - indices_to_chunks
  - indices_rerank

Usage:
  python test_cp_balance_utils_ops.py
"""

import os
import sys
import unittest

import numpy as np
import paddle


# Keep compatibility with both old/new workspace layouts.
CUR_DIR = os.path.dirname(__file__)
CANDIDATE_FLASHMASK_PATHS = [
    os.path.join(CUR_DIR, "..", "xhy-flash-attention", "flashmask"),
    os.path.join(CUR_DIR, "..", "..", "..", "flashmask", "xhy-flash-attention", "flashmask"),
]
for p in CANDIDATE_FLASHMASK_PATHS:
    abs_p = os.path.abspath(p)
    if os.path.isdir(abs_p) and abs_p not in sys.path:
        sys.path.insert(0, abs_p)

from flash_mask.cp_balance import flashmask_cpbalance_cudaops as cp_balance_ops


SINGLE_PTR = 1
DUAL_PTR = 2
FULL_PTR = 4


def _to_gpu_int32(arr: np.ndarray):
    return paddle.to_tensor(arr, dtype="int32", place=paddle.CUDAPlace(0))


def _pad_to_multiple_of_4(x: int) -> int:
    return ((x + 3) // 4) * 4


def _scan_max_min_ref(input_np: np.ndarray, block_n: int):
    total_batch = int(np.prod(input_np.shape[:-1]))
    seqlen = input_np.shape[-1]
    flat = input_np.reshape(total_batch, seqlen)

    nblocks = (seqlen + block_n - 1) // block_n
    nblocks_padded = _pad_to_multiple_of_4(nblocks)

    max_out = np.zeros((total_batch, nblocks_padded), dtype=np.int32)
    min_out = np.zeros((total_batch, nblocks_padded), dtype=np.int32)

    for b in range(total_batch):
        for blk in range(nblocks):
            s = blk * block_n
            e = min((blk + 1) * block_n, seqlen)
            chunk = flat[b, s:e]
            max_out[b, blk] = int(np.max(chunk))
            min_out[b, blk] = int(np.min(chunk))

    return max_out, min_out


def _infer_block_n_ref(head_size_rounded, is_causal, has_softcap, is_local, seqlen_q, seqlen_k,
                       has_lt_end=False, has_ut_start=False):
    if head_size_rounded <= 64:
        # In this custom-op path, is_flashmask is always true in C++.
        if not is_causal:
            return 96
        return 128
    if head_size_rounded <= 128:
        if is_causal or is_local or has_softcap:
            return 128
        if seqlen_q >= 1024 or seqlen_k >= 1024:
            return 128
        return 64
    if head_size_rounded <= 256:
        if has_lt_end and has_ut_start:
            return 32
        return 64
    raise RuntimeError("head_size_rounded not supported")


def _reduce_workload_ref(
    lt_start_max, lt_end_min, ut_start_max, ut_end_min,
    B, H, Tr, Tc, S, kBlockM, is_causal, ptr_tag
):
    bh_total = B * H
    workload = np.zeros((bh_total, Tr), dtype=np.int32)

    for bh in range(bh_total):
        for tr in range(Tr):
            m_s = tr * kBlockM
            m_e = min(m_s + kBlockM, S)
            cnt = 0
            for tc in range(Tc):
                lsm = int(lt_start_max[bh, tc])
                fully_masked = True

                if ptr_tag == FULL_PTR:
                    lem = int(lt_end_min[bh, tc])
                    usm = int(ut_start_max[bh, tc])
                    uem = int(ut_end_min[bh, tc])
                    fully_masked = (m_s >= lsm and m_e <= lem) or (m_s >= usm and m_e <= uem)
                elif ptr_tag == DUAL_PTR:
                    if is_causal:
                        lem = int(lt_end_min[bh, tc])
                        fully_masked = m_s >= lsm and m_e <= lem
                    else:
                        uem = int(ut_end_min[bh, tc])
                        fully_masked = (m_s >= lsm) or (m_e <= uem)
                elif ptr_tag == SINGLE_PTR:
                    fully_masked = m_s >= lsm

                if not fully_masked:
                    cnt += 1
            workload[bh, tr] = cnt

    return workload.reshape(B, H, Tr, 1)


def _indices_to_chunks_ref(row_indices: np.ndarray, bucket_indices: np.ndarray, chunk_size: int):
    out = np.zeros_like(row_indices, dtype=np.int32)
    for i, row_val in enumerate(row_indices):
        max_chunk_index = 0
        for bucket, bucket_idx in enumerate(bucket_indices):
            chunk_start = int(bucket_idx) * chunk_size
            local_index = int(row_val) - chunk_start
            local_index = max(local_index, 0)
            local_index = min(local_index, chunk_size)
            if local_index > 0:
                local_index += bucket * chunk_size
            if bucket == 0 or local_index > max_chunk_index:
                max_chunk_index = local_index
        out[i] = max_chunk_index
    return out


def _indices_rerank_ref(input_row_indices: np.ndarray, chunk_indices: np.ndarray, chunk_size: int):
    bsz, nheads, seqlen, feat = input_row_indices.shape
    num_chunks = len(chunk_indices)
    out_seqlen = num_chunks * chunk_size
    out = np.empty((bsz, nheads, out_seqlen, feat), dtype=np.int32)

    for b in range(bsz):
        for h in range(nheads):
            for s_out in range(out_seqlen):
                chunk_id = s_out // chunk_size
                chunk_offset = s_out % chunk_size
                src_s = int(chunk_indices[chunk_id]) * chunk_size + chunk_offset
                out[b, h, s_out, :] = input_row_indices[b, h, src_s, :]
    return out


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "CUDA is required for cp_balance CUDA ops tests")
class TestCPBalanceUtilsOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        required_ops = ["scan_max_min", "reduce_workload", "indices_to_chunks", "indices_rerank"]
        missing = [name for name in required_ops if not hasattr(cp_balance_ops, name)]
        if missing:
            raise unittest.SkipTest(f"Missing custom ops: {missing}")
        paddle.device.set_device("gpu")

    def test_scan_max_min_explicit_blocksize_matches_ref(self):
        rng = np.random.default_rng(123)
        x = rng.integers(0, 2048, size=(2, 3, 130), dtype=np.int32)
        x_t = _to_gpu_int32(x)

        max_t, min_t = cp_balance_ops.scan_max_min(
            x_t, 128, 130, 130, 64, False, 0.0, 0, 0
        )

        ref_max, ref_min = _scan_max_min_ref(x, block_n=64)
        np.testing.assert_array_equal(max_t.numpy(), ref_max)
        np.testing.assert_array_equal(min_t.numpy(), ref_min)

    def test_scan_max_min_auto_blocksize_heuristic_shape(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 1000, size=(1, 1, 1024), dtype=np.int32)
        x_t = _to_gpu_int32(x)

        max_t, min_t = cp_balance_ops.scan_max_min(
            x_t, 128, 1024, 1024, -1, False, 0.0, -1, -1
        )

        inferred = _infer_block_n_ref(
            head_size_rounded=128,
            is_causal=False,
            has_softcap=False,
            is_local=False,
            seqlen_q=1024,
            seqlen_k=1024,
        )
        expect_cols = _pad_to_multiple_of_4((1024 + inferred - 1) // inferred)

        self.assertEqual(list(max_t.shape), [1, expect_cols])
        self.assertEqual(list(min_t.shape), [1, expect_cols])

    def test_reduce_workload_single_dual_full_ptr(self):
        rng = np.random.default_rng(42)
        B, H, S, kBlockM = 1, 2, 256, 64
        Tr = (S + kBlockM - 1) // kBlockM
        Tc = _pad_to_multiple_of_4((S + kBlockM - 1) // kBlockM)
        BH = B * H

        lt_start_max = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        lt_start_min = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        lt_end_max = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        lt_end_min = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        ut_start_max = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        ut_start_min = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        ut_end_max = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
        ut_end_min = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)

        # SINGLE_PTR
        out_single = cp_balance_ops.reduce_workload(
            _to_gpu_int32(lt_start_max), _to_gpu_int32(lt_start_min),
            None, None, None, None, None, None,
            B, H, Tr, Tc, S, kBlockM, False, kBlockM
        ).numpy()
        ref_single = _reduce_workload_ref(
            lt_start_max, None, None, None,
            B, H, Tr, Tc, S, kBlockM, False, SINGLE_PTR
        )
        np.testing.assert_array_equal(out_single, ref_single)

        # DUAL_PTR (causal)
        out_dual = cp_balance_ops.reduce_workload(
            _to_gpu_int32(lt_start_max), _to_gpu_int32(lt_start_min),
            _to_gpu_int32(lt_end_max), _to_gpu_int32(lt_end_min),
            None, None,
            _to_gpu_int32(ut_end_max), _to_gpu_int32(ut_end_min),
            B, H, Tr, Tc, S, kBlockM, True, kBlockM
        ).numpy()
        ref_dual = _reduce_workload_ref(
            lt_start_max, lt_end_min, None, ut_end_min,
            B, H, Tr, Tc, S, kBlockM, True, DUAL_PTR
        )
        np.testing.assert_array_equal(out_dual, ref_dual)

        # FULL_PTR
        out_full = cp_balance_ops.reduce_workload(
            _to_gpu_int32(lt_start_max), _to_gpu_int32(lt_start_min),
            _to_gpu_int32(lt_end_max), _to_gpu_int32(lt_end_min),
            _to_gpu_int32(ut_start_max), _to_gpu_int32(ut_start_min),
            _to_gpu_int32(ut_end_max), _to_gpu_int32(ut_end_min),
            B, H, Tr, Tc, S, kBlockM, False, kBlockM
        ).numpy()
        ref_full = _reduce_workload_ref(
            lt_start_max, lt_end_min, ut_start_max, ut_end_min,
            B, H, Tr, Tc, S, kBlockM, False, FULL_PTR
        )
        np.testing.assert_array_equal(out_full, ref_full)

    def test_reduce_workload_m_block_size_switches(self):
        rng = np.random.default_rng(9)
        B, H, S = 1, 1, 384

        for kBlockM in (64, 96, 128):
            Tr = (S + kBlockM - 1) // kBlockM
            Tc = _pad_to_multiple_of_4((S + kBlockM - 1) // kBlockM)
            BH = B * H

            lt_start_max = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)
            lt_start_min = rng.integers(0, S + 1, size=(BH, Tc), dtype=np.int32)

            out = cp_balance_ops.reduce_workload(
                _to_gpu_int32(lt_start_max), _to_gpu_int32(lt_start_min),
                None, None, None, None, None, None,
                B, H, Tr, Tc, S, kBlockM, False, kBlockM
            ).numpy()
            ref = _reduce_workload_ref(
                lt_start_max, None, None, None,
                B, H, Tr, Tc, S, kBlockM, False, SINGLE_PTR
            )
            np.testing.assert_array_equal(out, ref)

    def test_indices_to_chunks_matches_ref(self):
        row_indices = np.array([0, 63, 64, 65, 127, 128, 192, 255], dtype=np.int32)
        chunk_bucket_indices = np.array([3, 1, 2], dtype=np.int32)
        chunk_size = 64

        out = cp_balance_ops.indices_to_chunks(
            _to_gpu_int32(row_indices),
            _to_gpu_int32(chunk_bucket_indices),
            chunk_size,
        ).numpy()
        ref = _indices_to_chunks_ref(row_indices, chunk_bucket_indices, chunk_size)

        np.testing.assert_array_equal(out, ref)

    def test_indices_rerank_matches_ref(self):
        rng = np.random.default_rng(1234)
        B, H, S, D = 2, 2, 16, 2
        chunk_size = 4
        chunk_indices = np.array([2, 0, 3, 1], dtype=np.int32)
        num_chunks = len(chunk_indices)

        x = rng.integers(0, 1000, size=(B, H, S, D), dtype=np.int32)

        out = cp_balance_ops.indices_rerank(
            _to_gpu_int32(x),
            _to_gpu_int32(chunk_indices),
            B,
            H,
            S,
            D,
            num_chunks,
            chunk_size,
        ).numpy()
        ref = _indices_rerank_ref(x, chunk_indices, chunk_size)

        np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    unittest.main()
