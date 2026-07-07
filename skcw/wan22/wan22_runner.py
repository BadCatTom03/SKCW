#!/usr/bin/env python3
"""Wan2.2 adapter via BerniniRendererPipeline.

Exposes the same three-method interface as hv10_runner.HV10Runner:
    generate(prompt, guidance_mode, seed, ...)  -> torch latents
    latents_to_pixels_np(latents)               -> np pixels  [3,T,H,W] in [-1,1]
    reencode_pixels(px_np)                      -> np latents  [16,T_lat,h,w]

Configuration (set before constructing BerniniRunner):
    BERNINI_CODE_DIR  — path to Bernini-main repo
    BERNINI_WEIGHT_DIR — path to pretrained Bernini weights
    WAN22_BASE_DIR     — path to Wan2.2 base model
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

BERNINI_CODE_DIR = Path("./Bernini-main")
BERNINI_WEIGHT_DIR = Path("./pretrained_models/bernini")
WAN22_BASE_DIR = Path("./pretrained_models/Wan2.2")

NUM_FRAMES = 17
HEIGHT, WIDTH = 480, 832
NUM_STEPS = 40
OMEGA_TI = 4.0
FPS = 16
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    if mse <= 1e-20:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def save_video_mp4(pixels_np: np.ndarray, path: Path, fps: int = FPS) -> None:
    try:
        import imageio
        frames = ((pixels_np.transpose(1, 2, 3, 0) + 1.0) * 127.5
                  ).clip(0, 255).astype(np.uint8)
        imageio.mimwrite(str(path), list(frames), fps=fps,
                         codec="libx264", quality=7)
    except Exception as exc:
        log(f"WARN save_video failed ({path.name}): {exc}")


# ---- Bernini weight layout detection ----------------------------------------

def _looks_like_diffusers_bernini(path: Path) -> bool:
    return (path.is_dir()
            and ((path / "config.json").exists() or (path / "model_index.json").exists())
            and (path / "transformer").is_dir() and (path / "transformer_2").is_dir())


def _has_safetensors(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix == ".safetensors"
    return any(p.suffix == ".safetensors" for p in path.rglob("*"))


def detect_pipeline_kwargs(weight_dir: Path, out_dir: Path) -> dict:
    for cand in [weight_dir] + sorted(p for p in weight_dir.iterdir() if p.is_dir()):
        if _looks_like_diffusers_bernini(cand):
            return {"config_dir": str(cand), "load_ckpt_weights": False}
    def find(patterns):
        for pat in patterns:
            for p in weight_dir.iterdir():
                if p.is_dir() and pat in p.name.lower() and _has_safetensors(p):
                    return p
        return None
    high = find(["high", "transformer"])
    low = find(["low", "transformer_2"])
    if high is None or low is None or high == low:
        raise RuntimeError(f"cannot identify Bernini weight layout under {weight_dir}")
    src_cfg = BERNINI_CODE_DIR / "configs" / "bernini_renderer_wan22" / "config.json"
    cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
    cfg["wan22_base"] = str(WAN22_BASE_DIR)
    runtime_dir = out_dir / "config_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"config_dir": str(runtime_dir), "high_noise_ckpt": str(high),
            "low_noise_ckpt": str(low), "load_ckpt_weights": True}


# ---- BerniniRunner ----------------------------------------------------------

class BerniniRunner:
    """Thin wrapper: load once, then T2V / v2v_chain with self-built latents."""

    def __init__(self, device: str = "cuda:0", out_dir: Path = Path(".")):
        import torch
        if str(BERNINI_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(BERNINI_CODE_DIR))
        from bernini.pipeline import BerniniRendererPipeline
        kwargs = detect_pipeline_kwargs(BERNINI_WEIGHT_DIR, out_dir)
        log(f"loading BerniniRendererPipeline with {kwargs} ...")
        t0 = time.time()
        self.pipe = BerniniRendererPipeline.from_pretrained(
            device=device,
            use_unipc=True,
            use_src_id_rotary_emb=True,
            **kwargs,
        )
        self.device = device
        log(f"pipeline loaded in {time.time() - t0:.1f}s")
        self.torch = torch
        self.t_lat = (NUM_FRAMES - 1) // 4 + 1
        self.h_lat, self.w_lat = HEIGHT // 8, WIDTH // 8

    def tokenize(self, prompt: str):
        return self.pipe._tokenize(prompt)

    def _vae_to_device(self):
        self.pipe.model.to("cpu")
        self.torch.cuda.empty_cache()
        self.pipe.vae.to(self.device)

    def vae_encode_video(self, video_b3thw):
        import bernini.pipeline as pm
        self._vae_to_device()
        return pm._vae_encode(self.pipe.vae, video_b3thw.to(self.device))

    def vae_decode(self, latents):
        import bernini.pipeline as pm
        self._vae_to_device()
        return pm._vae_decode(self.pipe.vae, latents.to(self.device))

    def generate(self, prompt: str, guidance_mode: str, seed: int,
                 omega_v: float = 3.0, cond_latent=None):
        torch = self.torch
        dev = self.device
        ids, mask = self.tokenize(prompt)
        nids, nmask = self.tokenize(NEG_PROMPT)
        ids, mask = ids.to(dev), mask.to(dev)
        nids, nmask = nids.to(dev), nmask.to(dev)
        if cond_latent is not None and hasattr(cond_latent, "to"):
            cond_latent = cond_latent.to(dev)
        kwargs = dict(
            input_ids=ids, attention_mask=mask,
            uncond_input_ids=nids, uncond_attention_mask=nmask,
            guidance_mode=guidance_mode,
            omega_V=omega_v, omega_TI=OMEGA_TI,
            num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
            num_inference_steps=NUM_STEPS, seed=seed,
        )
        if cond_latent is not None:
            kwargs["multi_video_vae_latents"] = [cond_latent]
        with torch.no_grad():
            latents = self.pipe.model.sample(**kwargs)
        return latents

    def latents_to_pixels_np(self, latents) -> np.ndarray:
        video = self.vae_decode(latents)
        v = np.asarray(video, dtype=np.float32)
        if v.ndim == 5:
            v = v[0]
        v = v.transpose(3, 0, 1, 2)
        return (v * 2.0 - 1.0).clip(-1.0, 1.0)

    def reencode_pixels(self, pixels_np: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(pixels_np).unsqueeze(0).to(self.device)
        z = self.vae_encode_video(x)
        return z[0].float().cpu().numpy()
