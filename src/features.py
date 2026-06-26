from __future__ import annotations

import numpy as np


def combine_pair(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Symmetric pair feature: sorted-concat + bitwise XOR."""
    if fp_a.shape != fp_b.shape:
        raise ValueError("fingerprints must have same length")
    lo, hi = (fp_a, fp_b) if _bit_key(fp_a) <= _bit_key(fp_b) else (fp_b, fp_a)
    xor = np.bitwise_xor(fp_a.astype(np.uint8), fp_b.astype(np.uint8))
    return np.concatenate([lo, hi, xor]).astype(np.float32)


def _bit_key(fp: np.ndarray) -> int:
    return int("".join(map(str, fp.tolist()[:32])), 2)
