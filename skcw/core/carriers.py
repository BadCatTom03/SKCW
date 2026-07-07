"""Carrier construction & modulation for Semantic-Keyed Content Watermarking (v0.3 SS5).

Key derivation:  K_c = SHA256(master_seed || "class" || c)[:8]
Carrier:         per (c, i, t) seed -> PRNG Gaussian pattern [C,H,W]
                 -> DCT band mask (P0: all-pass) -> IDCT -> L2 normalize
Modulation:      Delta_B[:, t] = sum_i (2*payload[i]-1) * P(i, t)
                 B_wm = B + alpha * Delta_B
Demodulation:    R[i] = sum_t < z'[:,t] - B[:,t], P(i,t) >   ; bit_hat = R[i] > 0

All in numpy float32; the runner converts to torch.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np


def derive_class_key(master_seed: str, class_id: int) -> bytes:
    h = hashlib.sha256(f"{master_seed}|class|{class_id}".encode("utf-8")).digest()
    return h[:8]


def _pattern_seed(class_key: bytes, bit_index: int, t: int) -> int:
    h = hashlib.sha256(class_key + f"|bit|{bit_index}|t|{t}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _dct_band_filter(pattern: np.ndarray, band: Optional[Tuple[float, float]]) -> np.ndarray:
    """Apply a radial DCT band-pass mask on the spatial dims of [C,H,W].

    band=(lo, hi) as fractions of the max radius; None or (0,1) = all-pass (P0).
    """
    if band is None or (band[0] <= 0.0 and band[1] >= 1.0):
        return pattern
    from scipy.fft import dctn, idctn  # lazy import; P0 default never hits this

    coeff = dctn(pattern, axes=(-2, -1), norm="ortho")
    h, w = pattern.shape[-2:]
    fy = np.arange(h)[:, None] / max(h - 1, 1)
    fx = np.arange(w)[None, :] / max(w - 1, 1)
    radius = np.sqrt(fy ** 2 + fx ** 2) / np.sqrt(2.0)
    mask = ((radius >= band[0]) & (radius < band[1])).astype(pattern.dtype)
    return idctn(coeff * mask, axes=(-2, -1), norm="ortho").astype(pattern.dtype)


def make_carrier(class_key: bytes, bit_index: int, t: int,
                 shape_cHW: Tuple[int, int, int],
                 band: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """One L2-normalized carrier pattern [C,H,W] for (class, bit, frame)."""
    rng = np.random.default_rng(_pattern_seed(class_key, bit_index, t))
    p = rng.standard_normal(size=shape_cHW).astype(np.float32)
    p = _dct_band_filter(p, band)
    norm = float(np.linalg.norm(p))
    return p / max(norm, 1e-12)


def make_delta(class_key: bytes, payload_bits: List[int],
               shape_cTHW: Tuple[int, int, int, int],
               band: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Full modulation field Delta_B [C,T,H,W] (unit-carrier sum, no alpha)."""
    c, t_len, h, w = shape_cTHW
    delta = np.zeros((c, t_len, h, w), dtype=np.float32)
    for t in range(t_len):
        for i, bit in enumerate(payload_bits):
            sgn = 2.0 * float(bit) - 1.0
            delta[:, t] += sgn * make_carrier(class_key, i, t, (c, h, w), band)
    return delta


def demodulate(residual_cTHW: np.ndarray, class_key: bytes, n_bits: int,
               band: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Correlate residual (z' - B) against carriers. Returns R [n_bits] float."""
    c, t_len, h, w = residual_cTHW.shape
    r = np.zeros(n_bits, dtype=np.float64)
    for t in range(t_len):
        frame = residual_cTHW[:, t].astype(np.float32)
        for i in range(n_bits):
            p = make_carrier(class_key, i, t, (c, h, w), band)
            r[i] += float(np.vdot(frame, p))
    return r
