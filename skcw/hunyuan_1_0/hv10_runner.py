#!/usr/bin/env python3
"""HV10Runner: HunyuanVideo-1.0 wrapper mirroring BerniniRunner's interface.

Exposes the SAME three methods the Wan2.2 BerniniRunner does, so the existing
generation / training / eval code works on HV-1.0 by swapping the runner:
    generate(prompt, seed)        -> torch latents [1,16,17,60,60]
    latents_to_pixels_np(latents) -> np pixels   [3,65,480,480] in [-1,1]
    reencode_pixels(px_np)        -> np latents   [16,17,60,60]

Pipeline loading is the H0-proven recipe (bf16 transformer, fp16 VAE,
cpu_offload, vae tiling).  Latent space is the SCALED (model) space:
generate returns scaled latents; decode divides by scaling_factor; encode
multiplies by it -> round-trip stays in one consistent space (H0
roundtrip_residual_rel=0.0705).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HV_ROOT = "./pretrained_models/HunyuanVideo"
HEIGHT, WIDTH, NUM_FRAMES = 480, 480, 65       # -> latent (16,17,60,60)
NUM_STEPS = 30
GUIDANCE = 6.0
LATENT_SHAPE = (16, 17, 60, 60)
FPS = 15
NEG_PROMPT = ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HV10Runner:
    def __init__(self, device: str = "cuda:0", out_dir: Path = Path(".")):
        import torch
        from diffusers import (HunyuanVideoPipeline,  # noqa: E402
                               HunyuanVideoTransformer3DModel)
        self.torch = torch
        self.device = device
        log("loading HunyuanVideoPipeline (HV-1.0) ...")
        t0 = time.time()
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            HV_ROOT, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            HV_ROOT, transformer=transformer, torch_dtype=torch.float16)
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload(device=device)
        log(f"HV-1.0 pipeline loaded in {time.time() - t0:.1f}s")

    # --- generation ---
    def generate(self, prompt: str, seed: int,
                 num_frames: int = NUM_FRAMES, steps: int = NUM_STEPS):
        """Returns scaled latents [1,16,17,60,60] (torch, on cpu/float)."""
        torch = self.torch
        out = self.pipe(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_frames=num_frames, num_inference_steps=steps,
            guidance_scale=GUIDANCE,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="latent",
        ).frames
        # output_type="latent" -> frames is the scaled latent tensor.
        # Keep a 5-D batched view [1,16,17,60,60] so callers' [0] index yields
        # the canonical single latent [16,17,60,60].
        z = out if torch.is_tensor(out) else torch.as_tensor(out)
        if z.ndim == 4:
            z = z[None]
        return z.float().cpu()

    # --- vae ---
    def latents_to_pixels_np(self, latents) -> np.ndarray:
        """scaled latents [1,16,T,h,w] or [16,T,h,w] -> [3,T',H,W] in [-1,1]."""
        torch = self.torch
        vae = self.pipe.vae
        z = latents if torch.is_tensor(latents) else torch.from_numpy(latents)
        if z.ndim == 4:
            z = z[None]
        zt = z.to(self.device, vae.dtype) / vae.config.scaling_factor
        with torch.no_grad():
            px = vae.decode(zt, return_dict=False)[0]
        return px[0].float().clamp(-1, 1).cpu().numpy().astype(np.float32)

    def reencode_pixels(self, pixels_np: np.ndarray) -> np.ndarray:
        """pixels [3,T,H,W] in [-1,1] -> scaled latents [16,T',h,w] (np)."""
        torch = self.torch
        vae = self.pipe.vae
        x = torch.from_numpy(pixels_np[None]).to(self.device, vae.dtype)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
        return z[0].float().cpu().numpy().astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def save_video_mp4(pixels_np: np.ndarray, path: Path, fps: int = FPS) -> None:
    try:
        import imageio
        frames = ((pixels_np.transpose(1, 2, 3, 0) + 1.0) * 127.5
                  ).clip(0, 255).astype(np.uint8)
        imageio.mimwrite(str(path), list(frames), fps=fps,
                         codec="libx264", quality=8)
    except Exception as exc:  # noqa: BLE001
        log(f"WARN save_video failed ({path.name}): {exc}")
