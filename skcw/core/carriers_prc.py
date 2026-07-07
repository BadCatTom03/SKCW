"""Efficient PRC-chip carrier field for n=512 chips (training-scale).

carriers.make_delta (48-bit, dense per-bit per-frame Gaussians) costs
O(n_bits * T * dim) -- fine for diag, impractical for n=512 online injection.

Here: keyed permutation partitions the FULL latent [C,T,H,W] (dim ~ 500k) into
n disjoint blocks (~dim/n positions each).  Chip i sign-modulates a keyed
Gaussian pattern restricted to block i, L2-normalized per block.

  delta  = sum_i s_i * G_i            (computed as one vectorized pass, O(dim))
  demod  : R_i = <residual, G_i>      (one vectorized pass, O(dim))

Orthogonality between chips is EXACT (disjoint supports).
Temporal note: blocks span all frames; frame-drop robustness is delegated to
P2 temporal alignment (plan v2 Sec.7.1), not to per-frame carrier structure.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def _rng(key: bytes, label: str) -> np.random.Generator:
    h = hashlib.sha256(key + b"|cprc|" + label.encode()).digest()
    return np.random.Generator(np.random.PCG64(list(h[:16])))


class PRCCarrier:
    """Per-(class_key, shape, n) carrier bank. Build once, reuse."""

    def __init__(self, class_key: bytes, shape_cTHW: Tuple[int, int, int, int],
                 n_chips: int = 512):
        self.shape = tuple(shape_cTHW)
        self.n = n_chips
        dim = int(np.prod(self.shape))
        # keyed permutation -> block assignment per latent position
        perm = _rng(class_key, f"perm|{self.shape}").permutation(dim)
        block = np.empty(dim, dtype=np.int32)
        block[perm] = (np.arange(dim) % n_chips).astype(np.int32)
        self.block = block
        # keyed Gaussian field, L2-normalized per block
        g = _rng(class_key, f"field|{self.shape}").standard_normal(dim).astype(np.float32)
        norms = np.sqrt(np.bincount(block, weights=g.astype(np.float64) ** 2,
                                    minlength=n_chips)).astype(np.float32)
        g /= np.maximum(norms[block], 1e-12)
        self.g = g  # flat [dim], unit L2 norm within each block

    def make_delta(self, codeword: np.ndarray) -> np.ndarray:
        """codeword {0,1}^n -> delta [C,T,H,W], per-block unit-norm signed field."""
        s = (np.asarray(codeword, dtype=np.float32) * 2.0 - 1.0)
        return (self.g * s[self.block]).reshape(self.shape)

    def demodulate(self, residual: np.ndarray) -> np.ndarray:
        """residual [C,T,H,W] -> soft chips R[n] (sign: chip=1 -> positive)."""
        r = residual.reshape(-1).astype(np.float64) * self.g
        return np.bincount(self.block, weights=r, minlength=self.n)

    def demodulate_torch(self, residual, block_t, g_t):
        """Differentiable demod for training. residual [B,C,T,H,W] torch."""
        import torch
        b = residual.shape[0]
        r = residual.reshape(b, -1) * g_t.unsqueeze(0)             # [B,dim]
        out = torch.zeros(b, self.n, device=residual.device, dtype=residual.dtype)
        return out.scatter_add_(1, block_t.unsqueeze(0).expand(b, -1), r)

    def torch_buffers(self, device):
        import torch
        return (torch.from_numpy(self.block.astype(np.int64)).to(device),
                torch.from_numpy(self.g).to(device))


if __name__ == "__main__":  # self-test
    from skcw.core.prc_code import prc_encode, prc_soft_decode, msg_bit_acc, DEFAULT_N
    key = b"selftest-key-000"
    shape = (16, 5, 60, 104)
    car = PRCCarrier(key, shape, DEFAULT_N)
    msg = 0xDEADBEEF
    cw = prc_encode(msg, key, DEFAULT_N)
    delta = car.make_delta(cw)
    print(f"delta std={delta.std():.5f} norm={np.linalg.norm(delta):.3f} "
          f"(expect sqrt(n)={np.sqrt(DEFAULT_N):.1f})")
    # clean channel
    soft = car.demodulate(0.3 * 0.4 * delta)  # g_hat=0.3, alpha=0.4
    dec = prc_soft_decode(soft, key, DEFAULT_N)
    print(f"clean: msg_ok={dec['message'] == msg} score={dec['score']:.2f}")
    # noisy channel: content interference ~ N(0, 0.45) latent
    rng = np.random.default_rng(1)
    noisy = 0.3 * 0.4 * delta + rng.normal(0, 0.10, shape).astype(np.float32)
    dec2 = prc_soft_decode(car.demodulate(noisy), key, DEFAULT_N)
    print(f"noise0.10: msg_ok={dec2['message'] == msg} "
          f"bitacc={msg_bit_acc(dec2, msg):.3f} score={dec2['score']:.2f}")
    # null
    dec0 = prc_soft_decode(car.demodulate(
        rng.normal(0, 0.45, shape).astype(np.float32)), key, DEFAULT_N)
    print(f"null: score={dec0['score']:.2f} (expect ~0.8)")
