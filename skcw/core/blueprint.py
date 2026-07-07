"""Blueprint construction for Semantic-Keyed Content Watermarking (v0.3 SS7).

All blueprints live in the NORMALIZED Wan VAE latent space
(per-channel (z - mean) / std, real-video std ~= 0.45).

P0 levels:
  L0     : zero-content probe blueprint (zeros, or constant mu*ones)
  L1     : semantic class template = blurred single-image latent repeated in time
  route1 : VAE_encode(T2V(prompt)) -- channel-strength ceiling diagnostic
           (route1 latents are produced by the runner via a T2V pass, not here)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_l0(shape_cTHW: Tuple[int, int, int, int], mu: float = 0.0) -> np.ndarray:
    """Zero-content blueprint: zeros (mu=0) or constant mu*ones."""
    c, t, h, w = shape_cTHW
    if mu == 0.0:
        return np.zeros((c, t, h, w), dtype=np.float32)
    return np.full((c, t, h, w), float(mu), dtype=np.float32)


def _gaussian_blur_2d(x_cHW: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur on the spatial dims of [C,H,W] (numpy only)."""
    if sigma <= 0:
        return x_cHW
    radius = max(1, int(3.0 * sigma + 0.5))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    pad = radius
    out = np.pad(x_cHW, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, out)
    out = np.pad(out, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 2, out)
    return out.astype(np.float32)


def build_l1_from_keyframe_latent(keyframe_latent_cHW: np.ndarray,
                                  t_len: int,
                                  blur_sigma: float = 2.0,
                                  std_clip: Optional[float] = 0.45) -> np.ndarray:
    """L1 semantic template: blur a single-frame latent, repeat across time.

    keyframe_latent_cHW: [C,H,W] normalized latent of a T2I keyframe
    std_clip: rescale so overall std does not exceed this (real-video std ~0.45)
    """
    blurred = _gaussian_blur_2d(keyframe_latent_cHW.astype(np.float32), blur_sigma)
    if std_clip is not None:
        cur = float(blurred.std())
        if cur > std_clip > 0:
            blurred = blurred * (std_clip / cur)
    return np.repeat(blurred[:, None], t_len, axis=1)  # [C,T,H,W]


def blueprint_stats(b_cTHW: np.ndarray) -> dict:
    return {
        "mean": float(b_cTHW.mean()),
        "std": float(b_cTHW.std()),
        "min": float(b_cTHW.min()),
        "max": float(b_cTHW.max()),
        "shape": list(b_cTHW.shape),
    }
