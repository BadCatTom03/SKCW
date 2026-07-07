#!/usr/bin/env python3
"""M5e: FULL temporal-recovery ablation for the paper robustness table.

Factors (all crossed):
  attacks   : 10 temporal attacks (M5b 8 + SIGMark noise-frame-group 2)
              + a 5-attack subset additionally combined with H.264 CRF23
                (compression applied AFTER the temporal attack = realistic
                 re-encode-and-redistribute adversary)
  recovery  : none   -> linear resample to 17 (plain detector preprocessing)
              of_seg -> SGO optical-flow segment recovery (M5c)
              swdet  -> slice-restricted PRC scoring realignment (M5d)
              both   -> of_seg first, then swdet realignment
  alphas    : 0.4 / 0.25, held-out classes c28-31 x 2 seeds (8 latents)

Same messages as M5b/M5c/M5d (rng seed 2027).
Probe maps cached by frame sha1 ACROSS modes within one (latent,alpha,attack)
record (of_seg keeps original frames, so cache hits are high).

Server:
  cd .
  nohup python eval_attacks_ablation_full.py \
      > eval_attacks_ablation_full_v6.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import DEFAULT_N, prc_encode  # noqa: E402
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402
from skcw.runner.p0_channel_capacity import BerniniRunner, psnr  # noqa: E402
from skcw.eval.eval_attacks import (  # noqa: E402
    MASTER_SEED, LATENT_SHAPE, D1_DIR, CKPT, ALPHAS, SEED_IDXS,
    Extractor, detect, append_jsonl, log, attack_h264,
)
from skcw.eval.eval_attacks_temporal import (  # noqa: E402
    attack_shuffle, attack_reverse, attack_drop, attack_insert, _resample_to,
)
from skcw.eval.eval_attacks_temporal_sgo import sgo_recover, _insert_noise  # noqa: E402
from skcw.eval.eval_attacks_temporal_swdet import (  # noqa: E402
    SwDetAligner, assign_frames, realign, NUM_FRAMES,
)

OUT_DIR = Path("./results/attack_eval_v6_ablation_full")

MODES = ("none", "of_seg", "swdet", "both")

TEMPORAL_ATTACKS = {
    "tshuffle_w2":      lambda px: attack_shuffle(px, 2),
    "tshuffle_w4":      lambda px: attack_shuffle(px, 4),
    "tshuffle_global":  lambda px: attack_shuffle(px, 999),
    "treverse":         attack_reverse,
    "tdrop_2":          lambda px: attack_drop(px, 2),
    "tdrop_4":          lambda px: attack_drop(px, 4),
    "tinsert_2":        lambda px: attack_insert(px, 2),
    "tinsert_4":        lambda px: attack_insert(px, 4),
    "tinsert_noise_1g": lambda px: _insert_noise(px, 1, 2),
    "tinsert_noise_2g": lambda px: _insert_noise(px, 2, 2),
}

# subset additionally tested with H.264 CRF23 re-encode after the attack
H264_SUBSET = ("tshuffle_w4", "tshuffle_global", "treverse",
               "tdrop_4", "tinsert_noise_2g")
H264_CRF = 23


def build_attack_variants():
    """-> list of (variant_name, temporal_fn, with_h264: bool)."""
    var = [(name, fn, False) for name, fn in TEMPORAL_ATTACKS.items()]
    for name in H264_SUBSET:
        var.append((f"{name}+h264", TEMPORAL_ATTACKS[name], True))
    return var


# ----------------------------- recovery modes -----------------------------

def probe_maps_cached(px, runner, extr, cache: dict) -> list:
    """Per-frame spatial maps with a cache shared across modes of one record."""
    maps = []
    for i in range(px.shape[1]):
        fr = np.ascontiguousarray(px[:, i])
        h = hashlib.sha1(fr.tobytes()).hexdigest()
        if h not in cache:
            clip = np.repeat(fr[:, None], NUM_FRAMES, axis=1)
            z = runner.reencode_pixels(clip)
            cache[h] = extr.extract(z).mean(axis=1)
        maps.append(cache[h])
    return maps


def swdet_align(px, algn, runner, extr, cache):
    """px (any T) -> resample to 17 -> slice-score -> realign. Returns
    (aligned_px, slice_assign list)."""
    px17 = _resample_to(px, NUM_FRAMES)
    maps = probe_maps_cached(px17, runner, extr, cache)
    slice_of = assign_frames(algn.score_matrix(maps))
    return realign(px17, slice_of), slice_of.tolist()


def recover(mode, px_att, algn, runner, extr, cache):
    """Apply recovery mode -> (px_recovered [3,17,H,W], extra_info dict)."""
    if mode == "none":
        return _resample_to(px_att, NUM_FRAMES), {}
    if mode == "of_seg":
        return sgo_recover(px_att), {}
    if mode == "swdet":
        px, sa = swdet_align(px_att, algn, runner, extr, cache)
        return px, {"slice_assign": sa}
    if mode == "both":
        px_seg = sgo_recover(px_att)
        px, sa = swdet_align(px_seg, algn, runner, extr, cache)
        return px, {"slice_assign": sa}
    raise ValueError(mode)


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M5e full temporal ablation")
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

    cars, aligners = {}, {}
    for c in HELDOUT_CLASSES:
        key = derive_class_key(MASTER_SEED, c)
        car = PRCCarrier(key, LATENT_SHAPE, DEFAULT_N)
        cars[c] = (key, car)
        aligners[c] = SwDetAligner(car, key)

    variants = build_attack_variants()
    rng = np.random.default_rng(2027)   # same messages as M5b/c/d
    cases = [(c, s) for c in HELDOUT_CLASSES for s in SEED_IDXS]
    log(f"cases: {len(cases)} latents x {len(ALPHAS)} alphas x "
        f"{len(variants)} attack variants x {len(MODES)} modes")

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
            algn = aligners[c]
            msg = int(rng.integers(0, 2 ** 32))
            cw = prc_encode(msg, key, DEFAULT_N)
            z_wm = z_gen + alpha * car.make_delta(cw)
            px_wm = runner.latents_to_pixels_np(
                torch.from_numpy(z_wm[None]).to(args.device))

            base = {"class_id": c, "seed_idx": sidx, "alpha": alpha,
                    "message_hex": f"{msg:08x}",
                    "psnr_wm_vs_clean": round(psnr(px_wm, px_clean), 3)}

            for vname, tfn, with264 in variants:
                rec = dict(base, attack=vname,
                           ts=time.strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    t0 = time.time()
                    px_att_wm = tfn(px_wm)
                    px_att_cl = tfn(px_clean)
                    if with264:
                        px_att_wm = attack_h264(px_att_wm, H264_CRF)
                        px_att_cl = attack_h264(px_att_cl, H264_CRF)
                    cache_wm, cache_cl = {}, {}
                    for mode in MODES:
                        pw, info = recover(mode, px_att_wm, algn,
                                           runner, extr, cache_wm)
                        rec[f"{mode}_wm"] = detect(
                            extr.extract(runner.reencode_pixels(pw)),
                            car, key, cw, msg)
                        if info:
                            rec[f"{mode}_info"] = info
                        pc, _ = recover(mode, px_att_cl, algn,
                                        runner, extr, cache_cl)
                        rec[f"{mode}_clean"] = detect(
                            extr.extract(runner.reencode_pixels(pc)),
                            car, key, None, None)
                    rec["sec"] = round(time.time() - t0, 1)
                    rec["status"] = "ok"
                    n_ok += 1
                    log(f"c{c:02d}_s{sidx} a{alpha} {vname}: " + " ".join(
                        f"{m}={rec[f'{m}_wm']['msg_bit_acc']}"
                        for m in MODES))
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc(limit=8)
                    n_fail += 1
                    log(f"    ERROR {vname}: {rec['error']}")
                append_jsonl(jsonl, rec)
                torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {}
    for vname, _, _ in build_attack_variants():
        for alpha in ALPHAS:
            sel = [r for r in rows
                   if r["attack"] == vname and r["alpha"] == alpha]
            if not sel:
                continue
            for mode in MODES:
                agg = lambda k1, k2: round(float(np.mean(  # noqa: E731
                    [r[k1][k2] for r in sel])), 4)
                summary[f"{vname}|a{alpha}|{mode}"] = {
                    "n": len(sel),
                    "chip_acc": agg(f"{mode}_wm", "chip_acc"),
                    "msg_bit_acc": agg(f"{mode}_wm", "msg_bit_acc"),
                    "score_wm": agg(f"{mode}_wm", "score"),
                    "score_clean": agg(f"{mode}_clean", "score"),
                }
    report = {"ckpt": args.ckpt, "alphas": ALPHAS,
              "heldout_classes": list(HELDOUT_CLASSES),
              "modes": list(MODES), "h264_subset": list(H264_SUBSET),
              "h264_crf": H264_CRF,
              "n_ok": n_ok, "n_fail": n_fail, "summary": summary,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "attack_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE ok={n_ok} fail={n_fail} -> {out_dir / 'attack_summary.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
