#!/usr/bin/env python3
"""M5: pixel-level attack robustness baseline for E_theta v6 (last.pt).

Pipeline per evaluation point (held-out classes c28-31, NEVER trained on):
    z_gen (D1 clean generated latent)
      -> inject z_wm = z_gen + alpha*delta        (latent space, real channel g
                                                   comes from the VAE round-trip)
      -> VAE decode to pixels                     (BerniniRunner.latents_to_pixels_np)
      -> ATTACK at pixel level                    (H.264 CRF sweep, resize, noise)
      -> VAE re-encode                            (BerniniRunner.reencode_pixels)
      -> E_theta extract (C_hat = s*hf + r_net(hf), last.pt s=0.8962)
      -> PRC demod -> chip_acc / msg_bit_acc / score

Clean (non-watermarked) videos go through the SAME attacks -> score_clean (FPR side).
Reference cell: attack "none" = pure VAE round-trip (matches heldout_eval_v6 strong).

Cost: no generation needed (reuses D1 z_gen). ~24 VAE decodes + ~200 encodes.

Server:
  cd .
  nohup python eval_attacks.py > eval_attacks_v6.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import DEFAULT_N, prc_encode, prc_soft_decode, msg_bit_acc  # noqa: E402
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402
from skcw.runner.p0_channel_capacity import BerniniRunner, psnr, save_video_mp4, FPS  # noqa: E402

MASTER_SEED = "skcw-v0.3-p0"
LATENT_SHAPE = (16, 5, 60, 104)
D1_DIR = Path("./results/d1_latents")
CKPT = Path("./results/e_theta_b2/last.pt")
OUT_DIR = Path("./results/attack_eval_v6")

ALPHAS = [0.4, 0.25]          # strong-ish (g~0.35 -> g*a~0.14) / weak-ish (~0.09)
SEED_IDXS = [0, 1]            # 2 seeds per held-out class -> 8 latents total


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------- pixel-level attacks -----------------------------
# All attacks: px [3,T,H,W] float32 in [-1,1] -> same shape/range.

def attack_none(px: np.ndarray) -> np.ndarray:
    return px


def attack_h264(px: np.ndarray, crf: int) -> np.ndarray:
    """Encode with libx264 at the given CRF, decode back. Uses imageio+ffmpeg."""
    import imageio
    frames = ((px.transpose(1, 2, 3, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "att.mp4")
        w = imageio.get_writer(
            path, fps=FPS, codec="libx264", quality=None,
            ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"])
        for fr in frames:
            w.append_data(fr)
        w.close()
        rd = imageio.mimread(path, memtest=False)
    arr = np.stack(rd[:px.shape[1]]).astype(np.float32) / 127.5 - 1.0  # [T,H,W,3]
    return arr.transpose(3, 0, 1, 2).clip(-1.0, 1.0)


def attack_resize(px: np.ndarray, scale: float) -> np.ndarray:
    """Bilinear downscale then upscale back to original resolution."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(px).permute(1, 0, 2, 3)  # [T,3,H,W]
    h, w = x.shape[-2:]
    y = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y.permute(1, 0, 2, 3).numpy().astype(np.float32).clip(-1.0, 1.0)


def attack_noise(px: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (px + rng.normal(0, std, px.shape).astype(np.float32)).clip(-1.0, 1.0)


ATTACKS = {
    "none":        attack_none,
    "h264_crf18":  lambda px: attack_h264(px, 18),
    "h264_crf23":  lambda px: attack_h264(px, 23),
    "h264_crf28":  lambda px: attack_h264(px, 28),
    "h264_crf33":  lambda px: attack_h264(px, 33),
    "resize_075":  lambda px: attack_resize(px, 0.75),
    "resize_050":  lambda px: attack_resize(px, 0.5),
    "noise_002":   lambda px: attack_noise(px, 0.02),
    "noise_004":   lambda px: attack_noise(px, 0.04),
}


# ----------------------------- E_theta extractor -----------------------------

class Extractor:
    """last.pt (run-6 residual parameterization): C_hat = s*hf(z) + r_net(hf(z))."""

    def __init__(self, ckpt_path: Path, device: str):
        import torch
        from skcw.extractor.e_theta_model import ETheta
        from skcw.extractor.train_e_theta import highpass
        self.torch = torch
        self.highpass = highpass
        self.device = device
        ck = torch.load(ckpt_path, map_location=device)
        self.model = ETheta().to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.s = float(ck.get("s_param", 1.0))
        self.step = ck.get("step")
        log(f"E_theta loaded: step={self.step} s_param={self.s:.4f}")

    def extract(self, z_np: np.ndarray) -> np.ndarray:
        """z [16,5,60,104] -> C_hat same shape (fp32 numpy)."""
        torch = self.torch
        with torch.no_grad():
            z = torch.from_numpy(z_np[None].astype(np.float32)).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(self.device != "cpu")):
                hf = self.highpass(z)
                c_hat = self.s * hf + self.model(hf)
        return c_hat.float().cpu().numpy()[0]


def detect(c_hat: np.ndarray, car: PRCCarrier, key, codeword, message) -> dict:
    soft = car.demodulate(c_hat)
    dec = prc_soft_decode(soft, key, DEFAULT_N)
    out = {"score": round(float(dec["score"]), 4)}
    if codeword is not None:
        hard = (soft > 0).astype(np.uint8)
        out["chip_acc"] = round(float((hard == codeword).mean()), 4)
        out["msg_bit_acc"] = round(float(msg_bit_acc(dec, message)), 4)
    return out


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M5 attack robustness baseline")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_videos", action="store_true",
                    help="save one wm mp4 per (class, alpha) for visual check")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "attack_results.jsonl"

    extr = Extractor(Path(args.ckpt), args.device)
    runner = BerniniRunner(device=args.device, out_dir=out_dir)
    import torch

    # carriers per held-out class
    cars = {}
    for c in HELDOUT_CLASSES:
        key = derive_class_key(MASTER_SEED, c)
        cars[c] = (key, PRCCarrier(key, LATENT_SHAPE, DEFAULT_N))

    rng = np.random.default_rng(2026)
    cases = [(c, s) for c in HELDOUT_CLASSES for s in SEED_IDXS]
    log(f"cases: {len(cases)} latents x {len(ALPHAS)} alphas x "
        f"{len(ATTACKS)} attacks (+clean per attack)")

    n_ok = n_fail = 0
    for c, sidx in cases:
        f = D1_DIR / f"c{c:02d}_s{sidx}.npz"
        if not f.exists():
            log(f"SKIP missing {f.name}")
            continue
        z_gen = np.load(f)["z_gen"].astype(np.float32)

        # decode CLEAN pixels once per latent (shared across alphas)
        px_clean = runner.latents_to_pixels_np(
            torch.from_numpy(z_gen[None]).to(args.device))

        for alpha in ALPHAS:
            key, car = cars[c]
            msg = int(rng.integers(0, 2 ** 32))
            cw = prc_encode(msg, key, DEFAULT_N)
            delta = car.make_delta(cw)
            z_wm = z_gen + alpha * delta
            px_wm = runner.latents_to_pixels_np(
                torch.from_numpy(z_wm[None]).to(args.device))
            if args.save_videos and sidx == 0:
                save_video_mp4(px_wm, out_dir / f"c{c:02d}_a{alpha}_wm.mp4")

            base = {"class_id": c, "seed_idx": sidx, "alpha": alpha,
                    "message_hex": f"{msg:08x}",
                    "psnr_wm_vs_clean": round(psnr(px_wm, px_clean), 3)}

            for att_name, att_fn in ATTACKS.items():
                rec = dict(base, attack=att_name,
                           ts=time.strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    t0 = time.time()
                    z_obs_wm = runner.reencode_pixels(att_fn(px_wm))
                    rec["wm"] = detect(extr.extract(z_obs_wm), car, key, cw, msg)
                    z_obs_cl = runner.reencode_pixels(att_fn(px_clean))
                    rec["clean"] = detect(extr.extract(z_obs_cl), car, key,
                                          None, None)
                    rec["sec"] = round(time.time() - t0, 1)
                    rec["status"] = "ok"
                    n_ok += 1
                    log(f"c{c:02d}_s{sidx} a{alpha} {att_name}: "
                        f"chip={rec['wm']['chip_acc']} "
                        f"msg={rec['wm']['msg_bit_acc']} "
                        f"s_wm={rec['wm']['score']} "
                        f"s_clean={rec['clean']['score']}")
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc(limit=8)
                    n_fail += 1
                    log(f"    ERROR {att_name}: {rec['error']}")
                append_jsonl(jsonl, rec)
                torch.cuda.empty_cache()

    # ---------------- aggregate summary ----------------
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {}
    for att in ATTACKS:
        for alpha in ALPHAS:
            sel = [r for r in rows if r["attack"] == att and r["alpha"] == alpha]
            if not sel:
                continue
            agg = lambda k1, k2: round(float(np.mean(  # noqa: E731
                [r[k1][k2] for r in sel])), 4)
            summary[f"{att}|a{alpha}"] = {
                "n": len(sel),
                "chip_acc": agg("wm", "chip_acc"),
                "msg_bit_acc": agg("wm", "msg_bit_acc"),
                "score_wm": agg("wm", "score"),
                "score_clean": agg("clean", "score"),
            }
    report = {"ckpt": args.ckpt, "alphas": ALPHAS,
              "heldout_classes": list(HELDOUT_CLASSES),
              "n_ok": n_ok, "n_fail": n_fail, "summary": summary,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "attack_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE ok={n_ok} fail={n_fail} -> {out_dir / 'attack_summary.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
