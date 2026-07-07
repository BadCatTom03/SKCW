#!/usr/bin/env python3
"""Held-out generalization evaluation for E_theta v6 (B2).

Question: did run-6 (chip~1.0 / msg=1.0 / probe=1.0 on the 224 TRAIN-class
latents) learn a real content-interference suppressor, or did it memorize the
high-frequency texture of those 224 specific latents?  The training probe
samples from the SAME pool, so it is blind to this failure mode.

Discriminator: classes 28-31 (HELDOUT_CLASSES in prompts32.py) were generated
in D1 but NEVER seen in training (`load_d1(d1_dir, set(TRAIN_CLASSES))`).
4 classes x 8 seeds = 32 genuinely unseen clean latents.

Protocol (mirrors training exactly -- same sample_batch, same highpass, same
demod/decode):
  pools     : heldout (c28-31, 32 latents) vs train_ref (c00-03, 32 latents,
              size-matched reference)
  detectors : e_theta  (C_hat = s*hf(z) + r_net(hf(z)), v6 best.pt)
              oracle   (C_hat = hf(z), zero-parameter baseline)
  conditions: strong  alpha~U[0.3,0.5]  g~U[0.3,0.45]  no aug   (probe regime)
              full+aug alpha~U[0.2,0.5] g~U[0.25,0.45] noise .015 jitter .02
              weak    alpha=0.2 g=0.25 (g*alpha=0.05 floor)     no aug

Decision rule (communicated 2026-06-12):
  heldout strong chip >= 0.9            -> real generalization, proceed to M4
  heldout strong chip ~ 0.7 (oracle)    -> net memorized beyond-oracle part;
                                           expand D1 or accept oracle level
  oracle reference points: chip 0.694 / strong msg 0.974 / weak msg 0.776

Server:
  cd .
  nohup python eval_heldout.py > heldout_eval_v6.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import TRAIN_CLASSES, HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import DEFAULT_N, prc_soft_decode, msg_bit_acc  # noqa: E402
from skcw.extractor.train_e_theta import (CarrierBank, load_d1, sample_batch,  # noqa: E402
                           highpass, LATENT_SHAPE)

DEFAULT_OUT = Path("./results/e_theta_b2")

CONDITIONS = {
    # name: (alpha_rng, g_rng, noise_std, gain_jitter)
    "strong":   ((0.3, 0.5),  (0.3, 0.45),  0.0,   0.0),
    "full_aug": ((0.2, 0.5),  (0.25, 0.45), 0.015, 0.02),
    "weak":     ((0.2, 0.2),  (0.25, 0.25), 0.0,   0.0),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def accumulate(c_hat_np: np.ndarray, meta, bank, acc: dict) -> None:
    """Per-sample metrics into acc lists (raw, not per-batch means)."""
    for b, m in enumerate(meta):
        key, car = bank.get(m["class_id"])
        soft = car.demodulate(c_hat_np[b])
        dec = prc_soft_decode(soft, key, DEFAULT_N)
        if m["clean"]:
            acc["score_clean"].append(float(dec["score"]))
        else:
            hard = (soft > 0).astype(np.uint8)
            acc["chip"].append(float((hard == m["codeword"]).mean()))
            acc["msg"].append(float(msg_bit_acc(dec, m["message"])))
            acc["score_wm"].append(float(dec["score"]))


def summarize(acc: dict) -> dict:
    out = {}
    for k, v in acc.items():
        if v:
            out[k] = round(float(np.mean(v)), 4)
            out[k + "_std"] = round(float(np.std(v)), 4)
            out["n_" + k] = len(v)
        else:
            out[k] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="held-out eval for E_theta v6")
    ap.add_argument("--ckpt", default=str(DEFAULT_OUT / "best.pt"))
    ap.add_argument("--d1_dir", default=str(
        Path("./results/d1_latents")))
    ap.add_argument("--out_json", default=str(DEFAULT_OUT / "heldout_eval_v6.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=16,
                    help="16 x batch 8 = 128 eval points per cell; each of "
                         "the 32 latents hits ~4 distinct (msg, alpha, g)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- data: held-out pool + size-matched train reference pool ----
    d1 = Path(args.d1_dir)
    pool_held = load_d1(d1, set(HELDOUT_CLASSES))
    pool_ref = load_d1(d1, set(TRAIN_CLASSES[:4]))  # c00-03, 32 latents
    n_h = sum(len(v) for v in pool_held.values())
    n_r = sum(len(v) for v in pool_ref.values())
    log(f"held-out pool: classes {sorted(pool_held)} ({n_h} latents); "
        f"train-ref pool: classes {sorted(pool_ref)} ({n_r} latents)")
    if n_h == 0:
        log("ERROR: no held-out latents found")
        return 1

    # ---- model ----
    bank = CarrierBank(dev)
    from skcw.extractor.e_theta_model import ETheta
    model = ETheta().to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    s_val = float(ck.get("s_param", 1.0))
    s_param = torch.tensor(s_val, device=dev)
    log(f"loaded {args.ckpt} (step {ck.get('step')}, "
        f"probe_chip {ck.get('probe_chip')}, s_param {s_val:.4f})")

    use_amp = dev != "cpu"

    def detect(z_np: np.ndarray, use_net: bool) -> np.ndarray:
        zt = torch.from_numpy(z_np).to(dev)
        with torch.no_grad():
            hf = highpass(zt)
            if not use_net:
                return hf.float().cpu().numpy()  # oracle
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                out = s_param * hf + model(hf)
            return out.float().cpu().numpy()

    # ---- sweep: pool x condition x detector ----
    report = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"),
              "s_param": s_val, "n_batches": args.n_batches,
              "batch": args.batch, "seed": args.seed, "cells": {}}
    for pool_name, pool in (("heldout", pool_held), ("train_ref", pool_ref)):
        for cond_name, (a_rng, g_rng, nz, jit) in CONDITIONS.items():
            # same rng per detector -> identical batches for fair comparison
            for det_name, use_net in (("e_theta", True), ("oracle", False)):
                rng = np.random.default_rng(args.seed)
                acc = {"chip": [], "msg": [], "score_wm": [], "score_clean": []}
                for _ in range(args.n_batches):
                    z_in, _t, meta = sample_batch(
                        pool, bank, args.batch, rng, p_clean=0.25,
                        alpha_rng=a_rng, g_rng=g_rng,
                        noise_std=nz, gain_jitter=jit)
                    accumulate(detect(z_in, use_net), meta, bank, acc)
                cell = summarize(acc)
                key = f"{pool_name}/{cond_name}/{det_name}"
                report["cells"][key] = cell
                log(f"{key}: chip={cell['chip']} msg={cell['msg']} "
                    f"s_wm={cell['score_wm']} s_clean={cell['score_clean']}")

    # ---- verdict ----
    c = report["cells"]
    hs = c["heldout/strong/e_theta"]["chip"]
    ts = c["train_ref/strong/e_theta"]["chip"]
    os_ = c["heldout/strong/oracle"]["chip"]
    gap = round(ts - hs, 4) if (hs is not None and ts is not None) else None
    if hs is None:
        verdict = "NO_DATA"
    elif hs >= 0.9:
        verdict = "GENERALIZES"   # -> proceed to M4
    elif os_ is not None and hs <= os_ + 0.05:
        verdict = "ORACLE_ONLY"   # memorized the beyond-oracle part
    else:
        verdict = "PARTIAL"
    report["summary"] = {
        "heldout_strong_chip": hs, "train_ref_strong_chip": ts,
        "generalization_gap": gap, "oracle_heldout_strong_chip": os_,
        "heldout_strong_msg": c["heldout/strong/e_theta"]["msg"],
        "heldout_weak_msg": c["heldout/weak/e_theta"]["msg"],
        "verdict": verdict,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(args.out_json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"VERDICT: {verdict}  heldout_strong_chip={hs} (train_ref {ts}, "
        f"gap {gap}, oracle {os_})")
    log(f"report -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
