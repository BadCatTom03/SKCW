#!/usr/bin/env python3
"""P0 channel capacity experiment for Semantic-Keyed Content Watermarking (v0.3 SS10).

Measures how much of a carrier field alpha*Delta survives the Bernini v2v_chain
generation channel: B_wm = B + alpha*Delta -> generate -> decode -> re-encode -> z'.

Metrics per condition:
  M1 d_prime : per-bit correlation separability (bit=1 vs bit=0 groups)
  M2 band_survival : DCT radial-band energy ratio of observed vs injected delta
  M3 g_hat   : effective channel gain <z'_wm - z'_ctrl, Delta> / (alpha*||Delta||^2)
  M4 quality : PSNR (always) + LPIPS (if lpips installed) wm vs ctrl video

Each grid condition writes one JSONL line immediately (crash-safe).

Server usage (A100, python):
  cd .
  nohup python p0_channel_capacity.py --smoke \
      --out_root ./results \
      > p0_smoke.log 2>&1 &

Full grid: omit --smoke  (3 blueprint levels x 4 alpha x 4 omega x 4 prompts = 192).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


from skcw.core.carriers import derive_class_key, make_delta, demodulate  # noqa: E402
from skcw.core.payload import encode_payload, verify_payload, PAYLOAD_BITS  # noqa: E402
from skcw.core.blueprint import build_l0, build_l1_from_keyframe_latent, blueprint_stats  # noqa: E402
from skcw.wan22.wan22_runner import (  # noqa: E402
    BerniniRunner, psnr, save_video_mp4, log, append_jsonl, FPS,
    BERNINI_CODE_DIR, BERNINI_WEIGHT_DIR, WAN22_BASE_DIR,
    NUM_FRAMES, HEIGHT, WIDTH, NUM_STEPS, OMEGA_TI, NEG_PROMPT,
)

DEFAULT_OUT_ROOT = Path("./results")

MASTER_SEED = "skcw-v0.3-p0"

# class_id -> prompt (P0 oracle routing: class is known at extraction)
PROMPTS = {
    0: "A fluffy orange cat slowly walks across a sunlit wooden floor, tail swaying, soft afternoon light.",
    1: "A red vintage car drives down a quiet city street lined with trees, smooth camera tracking shot.",
    2: "A tall waterfall cascades into a clear mountain pool surrounded by green pine forest, gentle mist.",
    3: "A dancer in a flowing blue dress spins gracefully on an empty stage under a single warm spotlight.",
}

FULL_GRID = {
    "levels": ["L0", "L1", "route1"],
    "alphas": [0.05, 0.1, 0.2, 0.4],
    "omegas": [1.0, 2.0, 3.0, 4.0],
    "classes": [0, 1, 2, 3],
}
SMOKE_GRID = {
    "levels": ["L0"],
    "alphas": [0.1, 0.4],
    "omegas": [1.0, 3.0],
    "classes": [0, 2],
}
# P0.1 dual-blueprint diagnostic (2026-06-10): smoke proved L0 pathological
# (trajectory chaos: band_survival up to 5641x, g_hat <= 0.14, w3.0 collapses).
# route1 = content-matched real T2V latent -> channel-strength CEILING;
# L1     = blurred keyframe template      -> cheap prompt-derived mainline candidate.
# omega fixed at 1.0 (w3.0 unusable under strong video guidance), classes lead
# with c2 (waterfall: best visual quality in smoke), c0 (cat) as second class.
DIAG_GRID = {
    "levels": ["route1", "L1"],
    "alphas": [0.1, 0.4],
    "omegas": [1.0],
    "classes": [2, 0],
}

DCT_BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def radial_band_energy(x_cthw: np.ndarray, bands) -> list:
    """DCT radial band energies summed over C,T for [C,T,H,W]."""
    from scipy.fft import dctn
    c, t, h, w = x_cthw.shape
    coeff = dctn(x_cthw.astype(np.float32), axes=(-2, -1), norm="ortho")
    fy = np.arange(h)[:, None] / max(h - 1, 1)
    fx = np.arange(w)[None, :] / max(w - 1, 1)
    radius = np.sqrt(fy ** 2 + fx ** 2) / np.sqrt(2.0)
    energy = (coeff ** 2).sum(axis=(0, 1))  # [H,W]
    return [float(energy[(radius >= lo) & (radius < hi)].sum()) for lo, hi in bands]


def d_prime(r_values: np.ndarray, payload_bits) -> float:
    bits = np.asarray(payload_bits)
    r1, r0 = r_values[bits == 1], r_values[bits == 0]
    if len(r1) == 0 or len(r0) == 0:
        return float("nan")
    pooled = np.sqrt(0.5 * (r1.var() + r0.var()))
    return float((r1.mean() - r0.mean()) / max(pooled, 1e-12))


def try_lpips(a_np: np.ndarray, b_np: np.ndarray, device: str):
    """LPIPS over frames; returns mean or None if lpips unavailable."""
    try:
        import lpips
        import torch
        net = getattr(try_lpips, "_net", None)
        if net is None:
            net = lpips.LPIPS(net="alex").to(device).eval()
            try_lpips._net = net
        with torch.no_grad():
            ta = torch.from_numpy(a_np).permute(1, 0, 2, 3).to(device)  # [T,3,H,W]
            tb = torch.from_numpy(b_np).permute(1, 0, 2, 3).to(device)
            vals = []
            for i in range(0, ta.shape[0], 4):
                vals.append(net(ta[i:i + 4], tb[i:i + 4]).flatten())
            return float(torch.cat(vals).mean().cpu())
    except Exception as exc:  # noqa: BLE001
        log(f"WARN lpips unavailable: {exc}")
        return None

# ----------------------------- blueprint preparation -----------------------------


class BlueprintFactory:
    """Builds normalized-latent blueprints; caches the per-class T2V latent
    (used by both route1 directly and L1 via its first frame)."""

    def __init__(self, runner: BerniniRunner, cache_dir: Path):
        self.runner = runner
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _t2v_latent_np(self, class_id: int) -> np.ndarray:
        cache = self.cache_dir / f"t2v_latent_class{class_id}.npy"
        if cache.exists():
            return np.load(cache)
        log(f"route1 cache miss: running T2V for class {class_id} ...")
        t0 = time.time()
        latents = self.runner.generate(
            PROMPTS[class_id], guidance_mode="t2v", seed=1000 + class_id, omega_v=3.0)
        arr = latents[0].float().cpu().numpy()  # [16,T_lat,h,w]
        np.save(cache, arr)
        log(f"route1 T2V class {class_id} done in {time.time() - t0:.1f}s, std={arr.std():.3f}")
        return arr

    def build(self, level: str, class_id: int, shape_cthw) -> np.ndarray:
        if level == "L0":
            return build_l0(shape_cthw)
        if level == "L1":
            base = self._t2v_latent_np(class_id)
            return build_l1_from_keyframe_latent(base[:, 0], t_len=shape_cthw[1],
                                                 blur_sigma=2.0, std_clip=0.45)
        if level == "route1":
            return self._t2v_latent_np(class_id).copy()
        raise ValueError(f"unknown blueprint level: {level}")


# ----------------------------- one grid condition -----------------------------


def run_condition(runner: BerniniRunner, factory: BlueprintFactory,
                  level: str, class_id: int, alpha: float, omega_v: float,
                  seed: int, out_dir: Path, save_videos: bool) -> dict:
    torch = runner.torch
    shape = (16, runner.t_lat, runner.h_lat, runner.w_lat)
    prompt = PROMPTS[class_id]
    key = derive_class_key(MASTER_SEED, class_id)
    message = (0xA5A50000 | class_id) & 0xFFFFFFFF
    bits = encode_payload(message)

    rec = {
        "level": level, "class_id": class_id, "alpha": alpha, "omega_v": omega_v,
        "seed": seed, "prompt": prompt, "message_hex": f"{message:08x}",
        "num_steps": NUM_STEPS, "guidance_mode": "v2v_chain",
        "height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES,
        "status": "started", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t_start = time.time()

    # 1. blueprint + carrier field (normalized latent space, numpy)
    b_np = factory.build(level, class_id, shape)
    delta_np = make_delta(key, bits, shape)
    b_wm_np = b_np + alpha * delta_np
    rec["blueprint_stats"] = blueprint_stats(b_np)
    rec["delta_norm"] = float(np.linalg.norm(delta_np))

    def to_latent(arr: np.ndarray):
        return torch.from_numpy(arr).unsqueeze(0).to(runner.device)

    # 2. generate watermarked + control (same seed, alpha=0)
    lat_wm = runner.generate(prompt, "v2v_chain", seed, omega_v, to_latent(b_wm_np))
    lat_ctrl = runner.generate(prompt, "v2v_chain", seed, omega_v, to_latent(b_np))
    rec["gen_sec"] = round(time.time() - t_start, 1)

    # 3. decode to pixels, re-encode to latents (the real extraction channel)
    px_wm = runner.latents_to_pixels_np(lat_wm)
    px_ctrl = runner.latents_to_pixels_np(lat_ctrl)
    z_wm = runner.reencode_pixels(px_wm)      # [16,T_lat,h,w] normalized
    z_ctrl = runner.reencode_pixels(px_ctrl)

    if save_videos:
        tag = f"{level}_c{class_id}_a{alpha}_w{omega_v}_s{seed}"
        save_video_mp4(px_wm, out_dir / f"{tag}_wm.mp4")
        save_video_mp4(px_ctrl, out_dir / f"{tag}_ctrl.mp4")

    # 4. metrics ------------------------------------------------------------
    # M1 d': correlate (z_wm - B) against carriers
    r_vals = demodulate(z_wm - b_np, key, PAYLOAD_BITS)
    rec["R_values"] = [round(float(v), 6) for v in r_vals]
    rec["d_prime"] = round(d_prime(np.asarray(r_vals), bits), 4)
    hard = [1 if v > 0 else 0 for v in r_vals]
    rec["bit_acc"] = round(float(np.mean([int(a == b) for a, b in zip(hard, bits)])), 4)
    rec["crc_pass"] = verify_payload(hard) == message

    # paired demod (z_wm - z_ctrl): removes content, isolates injected delta
    r_paired = demodulate(z_wm - z_ctrl, key, PAYLOAD_BITS)
    hard_p = [1 if v > 0 else 0 for v in r_paired]
    rec["d_prime_paired"] = round(d_prime(np.asarray(r_paired), bits), 4)
    rec["bit_acc_paired"] = round(float(np.mean([int(a == b) for a, b in zip(hard_p, bits)])), 4)

    # M2 DCT band survival: observed diff vs injected delta, per radial band
    diff = z_wm - z_ctrl
    e_inj = radial_band_energy(alpha * delta_np, DCT_BANDS)
    e_obs = radial_band_energy(diff, DCT_BANDS)
    rec["band_energy_injected"] = [round(v, 4) for v in e_inj]
    rec["band_energy_observed"] = [round(v, 4) for v in e_obs]
    rec["band_survival"] = [round(o / max(i, 1e-12), 4) for o, i in zip(e_obs, e_inj)]

    # M3 effective channel gain
    denom = alpha * float(np.linalg.norm(delta_np)) ** 2
    rec["g_hat"] = round(float(np.vdot(diff, delta_np)) / max(denom, 1e-12), 6)

    # M4 quality: wm vs ctrl pixels
    rec["psnr_wm_vs_ctrl"] = round(psnr(px_wm, px_ctrl), 3)
    lp = try_lpips(px_wm, px_ctrl, runner.device)
    rec["lpips_wm_vs_ctrl"] = round(lp, 5) if lp is not None else None

    rec["total_sec"] = round(time.time() - t_start, 1)
    rec["status"] = "ok"
    return rec

# ----------------------------- main -----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P0 channel capacity grid (v2v_chain)")
    ap.add_argument("--smoke", action="store_true", help="8-condition reduced grid")
    ap.add_argument("--diag", action="store_true",
                    help="P0.1 dual-blueprint diagnostic grid (route1 vs L1)")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="multiple seeds per condition (overrides --seed)")
    ap.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--levels", nargs="*", default=None, help="override blueprint levels")
    ap.add_argument("--resume", action="store_true",
                    help="skip conditions already present in results.jsonl")
    return ap.parse_args()


def condition_id(level: str, c: int, a: float, w: float, seed: int) -> str:
    return f"{level}|c{c}|a{a}|w{w}|s{seed}"


def main() -> int:
    args = parse_args()
    if args.diag:
        grid, tag = DIAG_GRID, "diag"
    elif args.smoke:
        grid, tag = SMOKE_GRID, "smoke"
    else:
        grid, tag = FULL_GRID, "full"
    if args.levels:
        grid = dict(grid, levels=args.levels)
    seeds = args.seeds if args.seeds else [args.seed]

    out_dir = Path(args.out_root) / f"p0_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    log(f"output dir: {out_dir}")

    done_ids = set()
    if args.resume:
        for prev in sorted(Path(args.out_root).glob(f"p0_{tag}_*/results.jsonl")):
            for line in prev.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done_ids.add(condition_id(r["level"], r["class_id"],
                                                  r["alpha"], r["omega_v"], r["seed"]))
                except Exception:  # noqa: BLE001
                    pass
        log(f"resume: {len(done_ids)} conditions already done")

    conditions = [
        (lv, c, a, w, s)
        for lv in grid["levels"]
        for c in grid["classes"]
        for a in grid["alphas"]
        for w in grid["omegas"]
        for s in seeds
    ]
    log(f"grid = {len(conditions)} conditions "
        f"(levels={grid['levels']}, alphas={grid['alphas']}, "
        f"omegas={grid['omegas']}, classes={grid['classes']}, seeds={seeds})")

    runner = BerniniRunner(device=args.device, out_dir=out_dir)
    factory = BlueprintFactory(runner, cache_dir=Path(args.out_root) / "blueprint_cache")

    append_jsonl(out_dir / "run_meta.json.jsonl", {
        "grid": grid, "args": vars(args), "seeds": seeds,
        "num_conditions": len(conditions),
        "constants": {"master_seed": MASTER_SEED, "num_steps": NUM_STEPS,
                      "omega_TI": OMEGA_TI, "height": HEIGHT, "width": WIDTH,
                      "num_frames": NUM_FRAMES, "payload_bits": PAYLOAD_BITS},
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    n_ok = n_fail = n_skip = 0
    for idx, (lv, c, a, w, s) in enumerate(conditions, 1):
        cid = condition_id(lv, c, a, w, s)
        if cid in done_ids:
            n_skip += 1
            continue
        log(f"=== [{idx}/{len(conditions)}] {cid} ===")
        try:
            rec = run_condition(runner, factory, lv, c, a, w,
                                s, out_dir, args.save_videos)
            n_ok += 1
            log(f"    d'={rec['d_prime']} d'_paired={rec['d_prime_paired']} "
                f"bitacc={rec['bit_acc']} g_hat={rec['g_hat']} "
                f"psnr={rec['psnr_wm_vs_ctrl']} lpips={rec['lpips_wm_vs_ctrl']} "
                f"({rec['total_sec']}s)")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            rec = {"level": lv, "class_id": c, "alpha": a, "omega_v": w,
                   "seed": s, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=12),
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            log(f"    ERROR: {rec['error']}")
        append_jsonl(results_path, rec)
        try:
            runner.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    log(f"DONE: ok={n_ok} fail={n_fail} skip={n_skip} -> {results_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
