import numpy as np


def flashmask_block_sparsity(
    causal,
    flashmask,
    B=None,
    H=None,
    S=None,
    Q_BLOCK_SIZE=128,
    KV_BLOCK_SIZE=128,
):
    """
    Compute block-level sparsity from flashmask representation.

    A block (Q_BLOCK_SIZE x KV_BLOCK_SIZE) is counted as sparse only when
    the *entire* block is masked out.  This matches the actual kernel
    behaviour where partially-masked blocks are still computed.

    Args:
        causal: Whether the base attention pattern is causal.
        flashmask: Tensor of shape (B, H, S, bounds) or None.
            Accepts paddle Tensor, torch Tensor, or numpy array.
            If None, sparsity is computed from the base causal/full
            pattern alone (requires B, H, S).
        B: Batch size (only required when flashmask is None).
        H: Number of heads (only required when flashmask is None).
        S: Sequence length (only required when flashmask is None).
        Q_BLOCK_SIZE: Row (query) block size used by the kernel.
        KV_BLOCK_SIZE: Column (key) block size used by the kernel.

    Returns:
        float: Sparsity ratio in [0, 1].
    """
    if flashmask is None and not causal:
        return 0.0
    elif flashmask is None and causal:
        Br = Q_BLOCK_SIZE
        Bc = KV_BLOCK_SIZE
        Tr = S // Br
        Tc = S // Bc
        total_size = B * H * S * S
        num_sparse_blocks = Tr * (Tc - 1) // 2 * B * H
        return float((num_sparse_blocks * Bc * Br) / total_size)

    # Convert to numpy (works with paddle / torch / numpy inputs)
    if hasattr(flashmask, 'cpu'):
        fm = flashmask.cpu().detach().numpy()
    elif hasattr(flashmask, 'numpy'):
        fm = flashmask.numpy()
    else:
        fm = np.asarray(flashmask)

    # Parse LTS / LTE / UTS / UTE from the last axis
    LTS = LTE = UTS = UTE = None
    bounds = fm.shape[-1]
    if bounds == 4:
        LTS, LTE, UTS, UTE = fm[..., 0], fm[..., 1], fm[..., 2], fm[..., 3]
    elif bounds == 2 and causal:
        LTS, LTE = fm[..., 0], fm[..., 1]
    elif bounds == 2 and not causal:
        LTS, UTE = fm[..., 0], fm[..., 1]
    else:
        LTS = fm[..., 0]

    # Infer B, H, S from the first available component
    for arr in (LTS, LTE, UTS, UTE):
        if arr is not None:
            B, H, S = arr.shape
            break

    Br = Q_BLOCK_SIZE
    Bc = KV_BLOCK_SIZE
    Tr = S // Br
    Tc = S // Bc

    # Fill defaults for missing components
    if LTS is None:
        LTS = np.full((B, H, S), S, dtype=np.int32)
    if LTE is None:
        LTE = np.full((B, H, S), S, dtype=np.int32)
    if UTS is None:
        UTS = np.full((B, H, S), 0, dtype=np.int32)
    if UTE is None:
        UTE = np.tile(np.arange(S, dtype=np.int32).reshape(1, 1, S), (B, H, 1))

    # Per-block min / max over the Bc key-positions inside each KV block
    # Shape after reshape+reduce: (B, H, Tc)
    LTStartMax = LTS.reshape(B, H, Tc, Bc).max(axis=-1)
    LTStartMin = LTS.reshape(B, H, Tc, Bc).min(axis=-1)
    LTEndMax   = LTE.reshape(B, H, Tc, Bc).max(axis=-1)
    LTEndMin   = LTE.reshape(B, H, Tc, Bc).min(axis=-1)
    UTStartMax = UTS.reshape(B, H, Tc, Bc).max(axis=-1)
    UTStartMin = UTS.reshape(B, H, Tc, Bc).min(axis=-1)
    UTEndMax   = UTE.reshape(B, H, Tc, Bc).max(axis=-1)
    UTEndMin   = UTE.reshape(B, H, Tc, Bc).min(axis=-1)

    num_dense_blocks = 0
    for bsz in range(B):
        for head in range(H):
            for i in range(Tr):
                for j in range(Tc):
                    if causal and j > i:
                        continue
                    # Fully inside lower-triangle mask -> sparse
                    if (i * Br >= LTStartMax[bsz, head, j]
                            and (i + 1) * Br <= LTEndMin[bsz, head, j]):
                        continue
                    # Fully inside upper-triangle mask -> sparse
                    if (i * Br >= UTStartMax[bsz, head, j]
                            and (i + 1) * Br <= UTEndMin[bsz, head, j]):
                        continue
                    # Everything else needs compute
                    num_dense_blocks += 1

    num_sparse_blocks = B * H * Tc * Tr - num_dense_blocks
    total_size = B * H * S * S
    return float((num_sparse_blocks * Bc * Br) / total_size)


def ranges_block_sparsity(
    q_ranges,
    k_ranges,
    attn_mask_type,
    seq_len_q,
    seq_len_k,
    Q_BLOCK_SIZE=128,
    KV_BLOCK_SIZE=128,
):
    """
    Compute block-level sparsity from ranges-based attention mask
    (e.g. MagiAttention q_ranges / k_ranges / attn_mask_type format).

    Each triple (q_ranges[i], k_ranges[i], attn_mask_type[i]) defines a
    rectangular attention region.  Mask types:
        0 (or False) = FULL,
        1 (or True)  = CAUSAL,
        2            = INVCAUSAL,
        3            = BICAUSAL.

    A block is counted as *dense* (needs compute) if **any** range-triple
    activates at least one token pair inside that block.

    Args:
        q_ranges:       List of [start, end] query ranges.
        k_ranges:       List of [start, end] key ranges.
        attn_mask_type: List of mask-type indicators (int or bool).
        seq_len_q:      Total query sequence length.
        seq_len_k:      Total key sequence length.
        Q_BLOCK_SIZE:   Row block size used by the kernel.
        KV_BLOCK_SIZE:  Column block size used by the kernel.

    Returns:
        float: Sparsity ratio in [0, 1].
    """
    Br = Q_BLOCK_SIZE
    Bc = KV_BLOCK_SIZE
    Tr = seq_len_q // Br
    Tc = seq_len_k // Bc

    # Block boundary arrays
    q_starts = np.arange(Tr) * Br
    q_ends   = q_starts + Br
    k_starts = np.arange(Tc) * Bc
    k_ends   = k_starts + Bc

    active = np.zeros((Tr, Tc), dtype=bool)

    for q_r, k_r, mt in zip(q_ranges, k_ranges, attn_mask_type):
        mt = int(mt)  # handles bool -> int
        q_s, q_e = q_r[0], q_r[1]
        k_s, k_e = k_r[0], k_r[1]

        # Which blocks overlap with this range pair?
        q_mask = (q_ends > q_s) & (q_starts < q_e)   # shape (Tr,)
        k_mask = (k_ends > k_s) & (k_starts < k_e)   # shape (Tc,)
        overlap = q_mask[:, None] & k_mask[None, :]   # shape (Tr, Tc)

        # Local coordinates within the range pair:
        #   local_row = q_pos - q_s,  local_col = k_pos - k_s
        # Per-block local coordinate extremes (clipped to range):
        min_lr = np.maximum(q_starts, q_s) - q_s          # shape (Tr,)
        max_lr = np.minimum(q_ends, q_e) - 1 - q_s        # shape (Tr,)
        min_lc = np.maximum(k_starts, k_s) - k_s          # shape (Tc,)
        max_lc = np.minimum(k_ends, k_e) - 1 - k_s        # shape (Tc,)

        if mt == 0:  # FULL
            active |= overlap
        elif mt == 1:  # CAUSAL: tril, bottom-right aligned
            # Condition: local_row + (seqlen_k - seqlen_q) >= local_col
            # Block active if max_lr + delta >= min_lc
            seqlen_q_r = q_e - q_s
            seqlen_k_r = k_e - k_s
            delta = seqlen_k_r - seqlen_q_r
            active |= overlap & (max_lr[:, None] + delta >= min_lc[None, :])
        elif mt == 2:  # INVCAUSAL: triu, top-left aligned
            # Condition: local_row <= local_col
            # Block active if min_lr <= max_lc
            active |= overlap & (min_lr[:, None] <= max_lc[None, :])
        elif mt == 3:  # BICAUSAL: intersection of CAUSAL and INVCAUSAL
            # Both conditions must hold for at least one position in the block:
            #   INVCAUSAL: min_lr <= max_lc
            #   CAUSAL:    max_lr + delta >= min_lc
            seqlen_q_r = q_e - q_s
            seqlen_k_r = k_e - k_s
            delta = seqlen_k_r - seqlen_q_r
            active |= (overlap
                        & (min_lr[:, None] <= max_lc[None, :])
                        & (max_lr[:, None] + delta >= min_lc[None, :]))

    num_dense  = int(np.sum(active))
    total_blocks = Tr * Tc
    num_sparse = total_blocks - num_dense
    return float((num_sparse * Br * Bc) / (seq_len_q * seq_len_k))
