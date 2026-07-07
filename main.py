#!/usr/bin/env python3
"""SKCW — Semantic-Keyed Content Watermarking: end-to-end demo.

Usage:
    # HV-1.0 (fast, smaller GPU footprint)
    python main.py --backend hv10 --prompt "A cat walks on a wooden floor" --class_id 0

    # Wan2.2 via Bernini (requires Bernini-main + pretrained weights)
    python main.py --backend wan22 --prompt "A red car drives down a street" --class_id 1

Pipeline (inversion-free):
    1. Key Derivation   — SHA-256(master_seed || class_id) → class key K_c
    2. Carrier Build    — PRCCarrier(K_c, latent_shape) → unit-L2 Gaussian blocks
    3. Payload Encode   — message + CRC16 → PRC codeword → delta = Σ s_i·b_i
    4. Embed            — z_wm = z_gen + α·delta  (VAE latent space)
    5. Generate         — diffusion denoise → VAE decode → watermarked video
    6. Extract          — VAE re-encode → highpass → E_θ 3D-UNet → carrier residual Ĉ
    7. Decode           — demodulate Ĉ with K_c → PRC soft decode → recovered message
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from skcw.core.carriers import derive_class_key
from skcw.core.carriers_prc import PRCCarrier
from skcw.core.prc_code import DEFAULT_N, prc_encode, prc_soft_decode, msg_bit_acc
from skcw.core.payload import encode_payload, verify_payload, PAYLOAD_BITS

MASTER_SEED = "skcw-v0.3-p0"
ALPHA = 0.4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_extractor_hv10(ckpt_path: str, device: str):
    """Load trained E_theta for HV-1.0."""
    import torch
    from skcw.extractor.e_theta_model import ETheta
    from skcw.extractor.train_e_theta import highpass
    ck = torch.load(ckpt_path, map_location=device)
    model = ETheta().to(device).eval()
    model.load_state_dict(ck["model"])
    s = float(ck.get("s_param", 1.0))
    log(f"E_theta-HV: step={ck.get('step')} s={s:.4f}")

    def extract(z_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.from_numpy(z_np[None].astype(np.float32)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                hf = highpass(z)
                c_hat = s * hf + model(hf)
        return c_hat.float().cpu().numpy()[0]

    return extract


def run_hv10(args) -> dict:
    """HV-1.0 pipeline: generate → embed → decode → re-encode → extract → detect."""
    import torch
    from skcw.hunyuan_1_0.hv10_runner import HV10Runner, LATENT_SHAPE, psnr

    runner = HV10Runner(device=args.device, out_dir=Path(args.out_dir))
    key = derive_class_key(MASTER_SEED, args.class_id)
    car = PRCCarrier(key, LATENT_SHAPE, DEFAULT_N)
    msg = args.message if args.message is not None else int(np.random.default_rng(args.seed).integers(0, 2**32))
    cw = prc_encode(msg, key, DEFAULT_N)
    extract_fn = load_extractor_hv10(args.ckpt, args.device) if args.ckpt else None

    result = {"backend": "hv10", "class_id": args.class_id, "message_hex": f"{msg:08x}",
              "alpha": ALPHA, "prompt": args.prompt[:200], "seed": args.seed}

    t0 = time.time()
    # 1. Generate clean latent
    log(f"generating: {args.prompt[:80]}...")
    z_gen = runner.generate(args.prompt, args.seed)[0].numpy().astype(np.float32)
    result["gen_sec"] = round(time.time() - t0, 1)
    log(f"  latent shape={z_gen.shape} std={z_gen.std():.3f} ({result['gen_sec']}s)")

    # 2. Embed watermark
    delta = car.make_delta(cw)
    z_wm = z_gen + ALPHA * delta
    delta_norm = float(np.linalg.norm(delta))
    result["delta_norm"] = round(delta_norm, 4)
    result["rel_energy"] = round(ALPHA * delta_norm / max(float(np.linalg.norm(z_gen)), 1e-12), 6)

    # 3. Decode to pixels
    px_wm = runner.latents_to_pixels_np(z_wm)
    px_clean = runner.latents_to_pixels_np(z_gen)
    result["psnr_wm_vs_clean"] = round(psnr(px_wm, px_clean), 2)
    log(f"  psnr(wm vs clean)={result['psnr_wm_vs_clean']} dB")

    # 4. Re-encode (simulates extraction channel)
    z_re = runner.reencode_pixels(px_wm)
    result["roundtrip_sec"] = round(time.time() - t0 - result["gen_sec"], 1)

    # 5. Extract & detect
    if extract_fn:
        c_hat = extract_fn(z_re)
        soft = car.demodulate(c_hat)
        dec = prc_soft_decode(soft, key, DEFAULT_N)
        result["score"] = round(float(dec["score"]), 4)
        result["chip_acc"] = round(float((soft > 0).astype(np.uint8) == cw).mean(), 4) if cw is not None else None
        result["msg_bit_acc"] = round(float(msg_bit_acc(dec, msg)), 4)
        result["crc_pass"] = verify_payload((soft > 0).astype(np.uint8))
        log(f"  detect: score={result['score']} bit_acc={result['msg_bit_acc']} crc={'PASS' if result['crc_pass'] else 'FAIL'}")

    result["total_sec"] = round(time.time() - t0, 1)
    return result


def run_wan22(args) -> dict:
    """Wan2.2/Bernini optional backend; requires Bernini-main and external weights."""
    from skcw.wan22.wan22_runner import BerniniRunner, log as wlog, psnr

    runner = BerniniRunner(device=args.device, out_dir=Path(args.out_dir))
    key = derive_class_key(MASTER_SEED, args.class_id)
    shape = (16, runner.t_lat, runner.h_lat, runner.w_lat)
    car = PRCCarrier(key, shape, DEFAULT_N)
    msg = args.message if args.message else int(np.random.default_rng(args.seed).integers(0, 2**32))
    cw = prc_encode(msg, key, DEFAULT_N)

    result = {"backend": "wan22", "class_id": args.class_id, "message_hex": f"{msg:08x}",
              "alpha": ALPHA, "prompt": args.prompt[:200], "seed": args.seed}

    t0 = time.time()
    # 1. Generate clean latent (T2V, no conditioning)
    wlog(f"generating T2V: {args.prompt[:80]}...")
    z_gen = runner.generate(args.prompt, "t2v", args.seed, omega_v=3.0)[0].float().cpu().numpy()
    result["gen_sec"] = round(time.time() - t0, 1)
    wlog(f"  latent shape={z_gen.shape} ({result['gen_sec']}s)")

    # 2. Embed → decode → re-encode (VAE roundtrip)
    z_wm = z_gen + ALPHA * car.make_delta(cw)
    px_wm = runner.latents_to_pixels_np(z_wm)
    px_clean = runner.latents_to_pixels_np(z_gen)
    result["psnr_wm_vs_clean"] = round(psnr(px_wm, px_clean), 2)
    z_re = runner.reencode_pixels(px_wm)

    # 3. Simple demodulation (no E_theta for Wan2.2 — direct correlation)
    from skcw.core.carriers import make_delta, demodulate
    r_vals = demodulate(z_re - z_gen, key, PAYLOAD_BITS)
    bits = encode_payload(msg)
    hard = [1 if v > 0 else 0 for v in r_vals]
    result["bit_acc"] = round(float(np.mean([int(a == b) for a, b in zip(hard, bits)])), 4)
    result["crc_pass"] = verify_payload(hard)
    result["total_sec"] = round(time.time() - t0, 1)
    wlog(f"  bit_acc={result['bit_acc']} crc={'PASS' if result['crc_pass'] else 'FAIL'}")
    return result


def main():
    ap = argparse.ArgumentParser(description="SKCW end-to-end demo")
    ap.add_argument("--backend", choices=["hv10", "wan22"], default="hv10",
                    help="generation backend")
    ap.add_argument("--prompt", required=True, help="text prompt for video generation")
    ap.add_argument("--class_id", type=int, default=0, help="semantic class id (0-31)")
    ap.add_argument("--seed", type=int, default=42, help="generation seed")
    ap.add_argument("--message", type=lambda x: int(x, 0), default=None,
                    help="32-bit message (hex ok: 0xA5...); random if unset")
    ap.add_argument("--ckpt", default=None,
                    help="path to E_theta checkpoint (HV-1.0 only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="./results")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    if args.backend == "hv10":
        result = run_hv10(args)
    else:
        result = run_wan22(args)

    result["wall_sec"] = round(time.time() - t_start, 1)
    out_path = Path(args.out_dir) / f"demo_{args.backend}_c{args.class_id}_s{args.seed}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done in {result['wall_sec']}s → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
