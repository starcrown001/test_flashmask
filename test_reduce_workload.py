"""
Unit test for reduce_workload CUDA kernel accuracy.

Covers:
  - SINGLE_PTR / DUAL_PTR / FULL_PTR dispatch modes
  - Small seqlen (Tc <= 1024, within single-pass)
  - Large seqlen > 128k (Tc > 1024, exercises stride-loop fix)
  - Multi-batch / multi-head

Usage:
    python test_reduce_workload.py
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xhy-flash-attention', 'flashmask'))

import paddle
from flash_mask.cp_balance import flashmask_cpbalance_cudaops as cp_balance_ops


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

SINGLE_PTR = 1
DUAL_PTR = 2
FULL_PTR = 4


def reduce_workload_ref(
    lt_start_max, lt_start_min,
    lt_end_max, lt_end_min,
    ut_start_max, ut_start_min,
    ut_end_max, ut_end_min,
    B, H, Tr, Tc, S, kBlockM, is_causal, ptr_tag
):
    """Pure-Python reference for reduce_workload_kernel.

    Args:
        lt_start_max: np.ndarray [BH, Tc] int32
        lt_start_min: np.ndarray [BH, Tc] int32
        lt_end_max:   np.ndarray [BH, Tc] int32 or None
        lt_end_min:   np.ndarray [BH, Tc] int32 or None
        ut_start_max: np.ndarray [BH, Tc] int32 or None
        ut_start_min: np.ndarray [BH, Tc] int32 or None
        ut_end_max:   np.ndarray [BH, Tc] int32 or None
        ut_end_min:   np.ndarray [BH, Tc] int32 or None
    Returns:
        workload: np.ndarray [B, H, Tr, 1] int32
    """
    BH = B * H
    workload = np.zeros((BH, Tr), dtype=np.int32)

    for bh in range(BH):
        for tr in range(Tr):
            m_s = tr * kBlockM
            m_e = min(m_s + kBlockM, S)
            cnt = 0
            for tc in range(Tc):
                lsm = int(lt_start_max[bh, tc])
                lsn = int(lt_start_min[bh, tc])
                fully_masked = True
                partially_masked = False

                if ptr_tag == FULL_PTR:
                    lex = int(lt_end_max[bh, tc])
                    lem = int(lt_end_min[bh, tc])
                    usm = int(ut_start_max[bh, tc])
                    usn = int(ut_start_min[bh, tc])
                    uex = int(ut_end_max[bh, tc])
                    uem = int(ut_end_min[bh, tc])
                    fully_masked = (m_s >= lsm and m_e <= lem) or \
                                   (m_s >= usm and m_e <= uem)
                    partially_masked = (m_s < lex and m_e > lsn) or \
                                       (m_s < uex and m_e > usn)
                elif ptr_tag == DUAL_PTR:
                    if is_causal:
                        lex = int(lt_end_max[bh, tc])
                        lem = int(lt_end_min[bh, tc])
                        fully_masked = m_s >= lsm and m_e <= lem
                        partially_masked = m_s < lex and m_e > lsn
                    else:
                        uex = int(ut_end_max[bh, tc])
                        uem = int(ut_end_min[bh, tc])
                        fully_masked = (m_s >= lsm) or (m_e <= uem)
                        partially_masked = (m_e > lsn) or (m_s < uex)
                elif ptr_tag == SINGLE_PTR:
                    fully_masked = m_s >= lsm
                    partially_masked = m_e > lsn

                cnt += 0 if fully_masked else (2 if partially_masked else 1)
            workload[bh, tr] = cnt

    return workload.reshape(B, H, Tr, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_tc(Tc):
    """scanMaxMin pads nblock_seqlen to multiple of 4."""
    return ((Tc + 3) // 4) * 4


def _rand_bounds(BH, Tc, S, rng):
    """Generate random monotonic-ish boundary arrays in [0, S]."""
    arr = rng.integers(0, S + 1, size=(BH, Tc)).astype(np.int32)
    return arr


def _call_reduce_workload(lt_start_max_np, lt_start_min_np,
                          lt_end_max_np, lt_end_min_np,
                          ut_start_max_np, ut_start_min_np,
                          ut_end_max_np, ut_end_min_np,
                          B, H, Tr, Tc_padded, S, Br, is_causal, m_block_size):
    """Wrap the CUDA op call, handling optional tensors."""
    def _to_gpu(arr):
        if arr is None:
            return None
        return paddle.to_tensor(arr, dtype='int32', place=paddle.CUDAPlace(0))

    lt_start_max_t = _to_gpu(lt_start_max_np)
    lt_start_min_t = _to_gpu(lt_start_min_np)
    lt_end_max_t = _to_gpu(lt_end_max_np)
    lt_end_min_t = _to_gpu(lt_end_min_np)
    ut_start_max_t = _to_gpu(ut_start_max_np)
    ut_start_min_t = _to_gpu(ut_start_min_np)
    ut_end_max_t = _to_gpu(ut_end_max_np)
    ut_end_min_t = _to_gpu(ut_end_min_np)

    workload = cp_balance_ops.reduce_workload(
        lt_start_max_t, lt_start_min_t,
        lt_end_max_t, lt_end_min_t,
        ut_start_max_t, ut_start_min_t,
        ut_end_max_t, ut_end_min_t,
        B, H, Tr, Tc_padded, S, Br, is_causal, m_block_size
    )
    return workload.numpy()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestReduceWorkload(unittest.TestCase):

    def _run_case(self, B, H, S, kBlockM, ptr_tag, is_causal, seed=42):
        rng = np.random.default_rng(seed)
        Tr = (S + kBlockM - 1) // kBlockM
        Tc_raw = (S + kBlockM - 1) // kBlockM
        Tc = _pad_tc(Tc_raw)
        BH = B * H
        Br = kBlockM  # row_block_size passed to kernel

        # Always need LTStartMax/Min
        lt_start_max = _rand_bounds(BH, Tc, S, rng)
        lt_start_min = _rand_bounds(BH, Tc, S, rng)

        lt_end_max_np = lt_end_min_np = None
        ut_start_max_np = ut_start_min_np = None
        ut_end_max_np = ut_end_min_np = None

        if ptr_tag >= DUAL_PTR:
            lt_end_max_np = _rand_bounds(BH, Tc, S, rng)
            lt_end_min_np = _rand_bounds(BH, Tc, S, rng)
            ut_end_max_np = _rand_bounds(BH, Tc, S, rng)
            ut_end_min_np = _rand_bounds(BH, Tc, S, rng)
        if ptr_tag == FULL_PTR:
            ut_start_max_np = _rand_bounds(BH, Tc, S, rng)
            ut_start_min_np = _rand_bounds(BH, Tc, S, rng)

        # CUDA result
        cuda_out = _call_reduce_workload(
            lt_start_max, lt_start_min,
            lt_end_max_np, lt_end_min_np,
            ut_start_max_np, ut_start_min_np,
            ut_end_max_np, ut_end_min_np,
            B, H, Tr, Tc, S, Br, is_causal, kBlockM
        )

        # Reference result
        ref_out = reduce_workload_ref(
            lt_start_max, lt_start_min,
            lt_end_max_np, lt_end_min_np,
            ut_start_max_np, ut_start_min_np,
            ut_end_max_np, ut_end_min_np,
            B, H, Tr, Tc, S, kBlockM, is_causal, ptr_tag
        )

        np.testing.assert_array_equal(
            cuda_out, ref_out,
            err_msg=f"Mismatch: B={B}, H={H}, S={S}, kBlockM={kBlockM}, "
                    f"ptr_tag={ptr_tag}, is_causal={is_causal}, "
                    f"Tr={Tr}, Tc={Tc}"
        )

    # ---- SINGLE_PTR ----

    def test_single_ptr_small(self):
        """SINGLE_PTR, S=8192 (Tc=64, well within 1024)."""
        self._run_case(B=1, H=1, S=8192, kBlockM=128, ptr_tag=SINGLE_PTR, is_causal=False)

    def test_single_ptr_large(self):
        """SINGLE_PTR, S=131200 > 128k (Tc=1025, exercises stride loop)."""
        self._run_case(B=1, H=1, S=131200, kBlockM=128, ptr_tag=SINGLE_PTR, is_causal=False)

    def test_single_ptr_256k(self):
        """SINGLE_PTR, S=262144=256k (Tc=2048)."""
        self._run_case(B=1, H=1, S=262144, kBlockM=128, ptr_tag=SINGLE_PTR, is_causal=False)

    # ---- DUAL_PTR (causal) ----

    def test_dual_ptr_causal_small(self):
        """DUAL_PTR causal, S=16384."""
        self._run_case(B=1, H=1, S=16384, kBlockM=128, ptr_tag=DUAL_PTR, is_causal=True)

    def test_dual_ptr_causal_large(self):
        """DUAL_PTR causal, S=196608=192k (Tc=1536)."""
        self._run_case(B=1, H=1, S=196608, kBlockM=128, ptr_tag=DUAL_PTR, is_causal=True)

    # ---- DUAL_PTR (non-causal) ----

    def test_dual_ptr_noncausal_small(self):
        """DUAL_PTR non-causal, S=32768."""
        self._run_case(B=1, H=1, S=32768, kBlockM=128, ptr_tag=DUAL_PTR, is_causal=False)

    def test_dual_ptr_noncausal_large(self):
        """DUAL_PTR non-causal, S=163840=160k (Tc=1280)."""
        self._run_case(B=1, H=1, S=163840, kBlockM=128, ptr_tag=DUAL_PTR, is_causal=False)

    # ---- FULL_PTR ----

    def test_full_ptr_small(self):
        """FULL_PTR, S=8192."""
        self._run_case(B=1, H=2, S=8192, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    def test_full_ptr_large(self):
        """FULL_PTR, S=131200 > 128k (Tc=1025)."""
        self._run_case(B=1, H=2, S=131200, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    def test_full_ptr_256k(self):
        """FULL_PTR, S=262144=256k (Tc=2048), multi-batch multi-head."""
        self._run_case(B=2, H=2, S=262144, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    # ---- Multi-batch / multi-head ----

    def test_multi_batch_head_small(self):
        """FULL_PTR, B=2, H=4, S=16384."""
        self._run_case(B=2, H=4, S=16384, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    def test_multi_batch_head_large(self):
        """FULL_PTR, B=2, H=2, S=196608=192k."""
        self._run_case(B=2, H=2, S=196608, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    # ---- Edge: Tc exactly 1024 (boundary) ----

    def test_boundary_tc_1024(self):
        """S=131072=128k exactly, Tc=1024 (boundary, no stride needed)."""
        self._run_case(B=1, H=1, S=131072, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    def test_boundary_tc_1025(self):
        """S=131200, Tc_raw=1025 -> Tc_padded=1028 (just over boundary)."""
        self._run_case(B=1, H=1, S=131200, kBlockM=128, ptr_tag=FULL_PTR, is_causal=False)

    # ---- Different kBlockM ----

    def test_kblockm_64_large(self):
        """kBlockM=64, S=131072 (Tc=2048)."""
        self._run_case(B=1, H=1, S=131072, kBlockM=64, ptr_tag=SINGLE_PTR, is_causal=False)

    def test_kblockm_96_large(self):
        """kBlockM=96, S=131136 > 128k (Tc=1366 padded to 1368)."""
        # S must be divisible by 96 for clean blocks (not required, but cleaner)
        self._run_case(B=1, H=1, S=131136, kBlockM=96, ptr_tag=DUAL_PTR, is_causal=True)


if __name__ == '__main__':
    unittest.main()
