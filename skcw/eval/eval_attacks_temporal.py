#!/usr/bin/env python3
"""M5b: temporal attack robustness for E_theta v6 (last.pt).

Same pipeline as eval_attacks.py, but with TEMPORAL attacks:
  - frame shuffle (window 2 / 4 / global), reverse
  - frame drop (2 / 4 frames) + linear temporal resample back to 17
  - frame insert (2 / 4 interpolated frames) + resample back to 17

All attacks map [3,17,H,W] -> [3,17,H,W]: the detector normalizes length
to NUM_FRAMES=17 before VAE re-encode (reasonable detector preprocessing).

Server:
  cd .
  nohup python eval_attacks_temporal.py \
      > eval_attacks_temporal_v6.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import DEFAULT_N, prc_encode, prc_soft_decode, msg_bit_acc  # noqa: E402,F401
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402
from skcw.runner.p0_channel_capacity import BerniniRunner, psnr  # noqa: E402
from skcw.eval.eval_attacks import (  # noqa: E402
    MASTER_SEED, LATENT_SHAPE, D1_DIR, CKPT, ALPHAS, SEED_IDXS,
    Extractor, detect, append_jsonl, log,
)

OUT_DIR = Path("./results/attack_eval_v6_temporal")


# ----------------------- temporal attacks [3,T,H,W] -> [3,T,H,W] -----------------------

def _resample_to(px: np.ndarray, t_out: int) -> np.ndarray:
    """Linear temporal interpolation to t_out frames."""
    t_in = px.shape[1]
    if t_in == t_out:
        return px
    idx = np.linspace(0.0, t_in - 1.0, t_out)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t_in - 1)
    w = (idx - lo).astype(np.float32)[None, :, None, None]
    return (px[:, lo] * (1.0 - w) + px[:, hi] * w).astype(np.float32)


def attack_shuffle(px: np.ndarray, window: int, seed: int = 0) -> np.ndarray:
    """Shuffle frame order within windows of `window` (window>=T -> global)."""
    rng = np.random.default_rng(seed)
    t = px.shape[1]
    idx = np.arange(t)
    if window >= t:
        rng.shuffle(idx)
    else:
        for s in range(0, t, window):
            seg = idx[s:s + window].copy()
            rng.shuffle(seg)
            idx[s:s + window] = seg
    return px[:, idx].copy()


def attack_reverse(px: np.ndarray) -> np.ndarray:
    return px[:, ::-1].copy()


def attack_drop(px: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Randomly drop k frames, resample back to original length."""
    rng = np.random.default_rng(seed)
    t = px.shape[1]
    keep = np.sort(rng.choice(t, t - k, replace=False))
    return _resample_to(px[:, keep], t)


def attack_insert(px: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Insert k interpolated frames at random positions, resample back."""
    rng = np.random.default_rng(seed)
    t = px.shape[1]
    pos = set(np.sort(rng.choice(t - 1, k, replace=False)).tolist())
    out = []
    for i in range(t):
        out.append(px[:, i])
        if i in pos:
            out.append(0.5 * (px[:, i] + px[:, min(i + 1, t - 1)]))
    arr = np.stack(out, axis=1).astype(np.float32)
    return _resample_to(arr, t)


ATTACKS = {
    "tshuffle_w2":     lambda px: attack_shuffle(px, 2),
    "tshuffle_w4":     lambda px: attack_shuffle(px, 4),
    "tshuffle_global": lambda px: attack_shuffle(px, 999),
    "treverse":        attack_reverse,
    "tdrop_2":         lambda px: attack_drop(px, 2),
    "tdrop_4":         lambda px: attack_drop(px, 4),
    "tinsert_2":       lambda px: attack_insert(px, 2),
    "tinsert_4":       lambda px: attack_insert(px, 4),
}


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M5b temporal attack robustness")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "attack_results.jsonl"

    extr = Extractor(Path(args.ckpt), args.device)
    runner = BerniniRunner(device=args.device, out_dir=out_dir)
    import torch

    cars = {}
    for c in HELDOUT_CLASSES:
        key = derive_class_key(MASTER_SEED, c)
        cars[c] = (key, PRCCarrier(key, LATENT_SHAPE, DEFAULT_N))

    rng = np.random.default_rng(2027)
    cases = [(c, s) for c in HELDOUT_CLASSES for s in SEED_IDXS]
    log(f"cases: {len(cases)} latents x {len(ALPHAS)} alphas x "
        f"{len(ATTACKS)} temporal attacks (+clean per attack)")

    n_ok = n_fail = 0
    for c, sidx in cases:
        f = D1_DIR / f"c{c:02d}_s{sidx}.npz"
        if not f.exists():
            log(f"SKIP missing {f.name}")
            continue
        z_gen = np.load(f)["z_gen"].astype(np.float32)
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
