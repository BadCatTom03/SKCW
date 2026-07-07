#!/usr/bin/env python3
"""P6 Capacity sweep: trade redundancy for message capacity.

Sweeps msg_bits in {32,64,96,128,192,256} at fixed n_chips in {512,1024}.
Redundancy r = n_chips // msg_bits drops from 16x -> 2x as capacity rises.

No video generation needed: reuses d1 clean latents from HV-1.0.
For Wan2.2: uses Bernini blueprint latents (set --backbone wan22).

Attacks evaluated per cell:
  - clean (no attack)
  - H.264 CRF=23
  - H.264 CRF=28
  - resize 75%

Usage:
  nohup python p6_capacity_sweep.py --backbone hv10 > p6_cap.log 2>&1 &
  nohup python p6_capacity_sweep.py --backbone hv10 --n_chips 1024 > p6_cap1024.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from skcw.core.prc_code import prc_encode, prc_soft_decode, msg_bit_acc, int_to_bits
from skcw.core.carriers_prc import PRCCarrier
from skcw.core.carriers import derive_class_key

# ---- Grids ----------------------------------------------------------------
MSG_BITS_GRID = [32, 64, 96, 128, 192, 256]
N_CHIPS_DEFAULT = 512
ALPHA = 0.4
MASTER_SEED = "skcw-v0.3-p0"
N_CLASSES = 4   # use first 4 classes (fast); or set to 32 for full run

ATTACKS = ["clean", "crf23", "crf28", "resize75"]


def apply_attack(pixels_np: np.ndarray, attack: str, runner) -> np.ndarray:
    """Apply attack to pixels [3,T,H,W] float32 in [-1,1], return re-encoded latent."""
    if attack == "clean":
        return runner.reencode_pixels(pixels_np)

    # convert to uint8 for codec
    import cv2, os
    T = pixels_np.shape[1]
    frames = [(((pixels_np[:, t] + 1.0) * 127.5).clip(0, 255)
               .astype(np.uint8).transpose(1, 2, 0)) for t in range(T)]
    H, W = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as td:
        inp_path = os.path.join(td, "input.mp4")
        out_path = os.path.join(td, "output.mp4")
        fps = 8

        # write input video
        vw = cv2.VideoWriter(inp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()

        if attack.startswith("crf"):
            crf = int(attack[3:])
            subprocess.run(
                ["ffmpeg", "-y", "-i", inp_path,
                 "-vcodec", "libx264", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", out_path],
                capture_output=True, check=True)
        elif attack == "resize75":
            nh, nw = int(H * 0.75), int(W * 0.75)
            tmp = os.path.join(td, "small.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", inp_path,
                 "-vf", f"scale={nw}:{nh}",
                 "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", tmp],
                capture_output=True, check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp,
                 "-vf", f"scale={W}:{H}",
                 "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out_path],
                capture_output=True, check=True)
        else:
            raise ValueError(f"Unknown attack: {attack}")

        # read back
        cap = cv2.VideoCapture(out_path)
        attacked_frames = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            attacked_frames.append(frm[:, :, ::-1])
        cap.release()

    # pad/trim to original T, convert back to [-1,1] float32
    while len(attacked_frames) < T:
        attacked_frames.append(attacked_frames[-1])
    attacked_frames = attacked_frames[:T]
    arr = np.stack(attacked_frames, axis=1).astype(np.float32) / 127.5 - 1.0  # [3,T,H,W]? no...
    arr = np.stack(attacked_frames, axis=0).astype(np.float32) / 127.5 - 1.0  # [T,H,W,3]
    arr = arr.transpose(3, 0, 1, 2)  # [3,T,H,W]

    return runner.reencode_pixels(arr)


def run_sweep(backbone: str, n_chips: int, out_root: Path, e_theta_ckpt: Path | None):
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / f"cap_sweep_{backbone}_n{n_chips}.jsonl"
    summary_path = out_root / f"cap_sweep_{backbone}_n{n_chips}_summary.json"

    if backbone == "hv10":
        from skcw.hunyuan_1_0.hv10_runner import HV10Runner, LATENT_SHAPE, log
        runner = HV10Runner(device="cuda:0", out_dir=out_root)
        d1_dir = Path("./results/d1_latents_hv10")
        latent_shape = LATENT_SHAPE
    else:
        raise NotImplementedError("Wan2.2 capacity sweep: use Bernini latents directly")

    # optionally load E_theta for extraction
    extractor = None
    if e_theta_ckpt and e_theta_ckpt.exists():
        import torch
        from skcw.extractor.e_theta_model import ETheta
        from skcw.extractor.train_e_theta import highpass as hp_fn
        ck = torch.load(e_theta_ckpt, map_location="cuda:0")
        m = ETheta().to("cuda:0").eval()
        m.load_state_dict(ck["model"])
        s_param = float(ck.get("s_param", 1.0))
        def extractor(z_np):
            import torch
            with torch.no_grad():
                z = torch.from_numpy(z_np[None].astype(np.float32)).cuda()
                hf = hp_fn(z)
                return (s_param * hf + m(hf)).float().cpu().numpy()[0]
        log("E_theta loaded for extraction")
    else:
        log("No E_theta checkpoint — using oracle high-pass extraction")
        from skcw.extractor.train_e_theta import highpass as hp_fn
        def extractor(z_np):
            import torch
            with torch.no_grad():
                z = torch.from_numpy(z_np[None].astype(np.float32)).cuda()
                return hp_fn(z).float().cpu().numpy()[0]

    rng = np.random.default_rng(42)
    msg_rng = random.Random(42)   # for arbitrary-width message ints (getrandbits)
    results = []

    for class_id in range(N_CLASSES):
        npz_path = d1_dir / f"c{class_id:02d}_s0.npz"
        if not npz_path.exists():
            log(f"[SKIP] {npz_path} not found")
            continue
        z_gen = np.load(npz_path)["z_gen"].astype(np.float32)
        key = derive_class_key(MASTER_SEED, class_id)

        # decode pixels once (shared across all configs)
        px_clean = runner.latents_to_pixels_np(z_gen)

        for mb in MSG_BITS_GRID:
            if n_chips < mb:
                log(f"[SKIP] n_chips={n_chips} < msg_bits={mb}")
                continue
            r = n_chips // mb
            message = msg_rng.getrandbits(mb)
            carrier = PRCCarrier(key, latent_shape, n_chips)
            cw = prc_encode(message, key, n_chips, mb)
            delta = carrier.make_delta(cw)
            z_wm = z_gen + ALPHA * delta
            px_wm = runner.latents_to_pixels_np(z_wm)

            for attack in ATTACKS:
                t0 = time.time()
                try:
                    z_re = apply_attack(px_wm, attack, runner)
                    c_hat = extractor(z_re)
                    soft = carrier.demodulate(c_hat)
                    dec = prc_soft_decode(soft, key, n_chips, mb)
                    mba = float(msg_bit_acc(dec, message))
                    rec = {
                        "class_id": class_id,
                        "msg_bits": mb,
                        "n_chips": n_chips,
                        "redundancy": r,
                        "attack": attack,
                        "msg_bit_acc": round(mba, 4),
                        "chip_acc": round(float(dec["chip_acc_proxy"]), 4),
                        "score": round(float(dec["score"]), 4),
                        "t_s": round(time.time() - t0, 2),
                    }
                    log(f"c{class_id} mb={mb:3d} r={r:2d} {attack:10s} "
                        f"bit_acc={mba:.4f} score={rec['score']:.2f}")
                except Exception as exc:
                    rec = {"class_id": class_id, "msg_bits": mb, "n_chips": n_chips,
                           "attack": attack, "error": str(exc)}
                    log(f"[ERROR] c{class_id} mb={mb} {attack}: {exc}")

                results.append(rec)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    # build summary: mean bit_acc per (msg_bits, attack)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        if "error" not in r:
            agg[(r["msg_bits"], r["attack"])].append(r["msg_bit_acc"])

    summary = {
        "backbone": backbone, "n_chips": n_chips, "alpha": ALPHA,
        "n_classes": N_CLASSES,
        "table": [
            {"msg_bits": mb, "redundancy": n_chips // mb, "attack": atk,
             "bit_acc_mean": round(float(np.mean(v)), 4),
             "bit_acc_std": round(float(np.std(v)), 4),
             "n": len(v)}
            for (mb, atk), v in sorted(agg.items())
        ],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Summary -> {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="hv10", choices=["hv10"])
    ap.add_argument("--n_chips", type=int, default=N_CHIPS_DEFAULT)
    ap.add_argument("--out_root", default="./results/capacity_sweep")
    ap.add_argument("--ckpt", default="./results/e_theta_hv10/last.pt")
    args = ap.parse_args()

    summary = run_sweep(
        backbone=args.backbone,
        n_chips=args.n_chips,
        out_root=Path(args.out_root),
        e_theta_ckpt=Path(args.ckpt),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
