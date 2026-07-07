#!/usr/bin/env python3
"""M5c: temporal attack robustness WITH SGO (of_seg path) ablation.

Comparison with eval_attacks_temporal.py (no-SGO baseline, M5b):
  - Same D1 held-out latents, alphas, attacks, plus SIGMark-style noise-frame
    insertion attacks (tinsert_noise_*).
  - Before re-encode, frames go through SGO recovery:
      optical_flow_segmentation -> outlier segment removal / gap padding
      -> pad_video_by_interpolation -> align to NUM_FRAMES.
    This is the SIGMark of_seg path (sw_det=False branch), adapted to SKCW
    pixel arrays. SKCW-specific sw_det (delta time-slice correlation) is M5d.

Server:
  cd .
  nohup python eval_attacks_temporal_sgo.py \
      > eval_attacks_temporal_sgo_v6.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


from skcw.core.prompts32 import HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import DEFAULT_N, prc_encode  # noqa: E402
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402
from skcw.runner.p0_channel_capacity import BerniniRunner, psnr  # noqa: E402
from skcw.eval.eval_attacks import (  # noqa: E402
    MASTER_SEED, LATENT_SHAPE, D1_DIR, CKPT, ALPHAS, SEED_IDXS,
    Extractor, detect, append_jsonl, log,
)
from skcw.eval.eval_attacks_temporal import (  # noqa: E402
    _resample_to, attack_shuffle, attack_reverse, attack_drop, attack_insert,
)
from skcw.eval.sgo_video_utils import (  # noqa: E402
    optical_flow_segmentation, pad_video_by_interpolation, align_frames,
)

OUT_DIR = Path("./results/attack_eval_v6_temporal_sgo")
NUM_FRAMES = 17


# ----------------- pixel array [-1,1] float <-> PIL frames -----------------

def px_to_frames(px: np.ndarray) -> list:
    """[3,T,H,W] float in [-1,1] -> list of T PIL RGB images."""
    arr = ((px.transpose(1, 2, 3, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def frames_to_px(frames: list) -> np.ndarray:
    """List of PIL RGB images -> [3,T,H,W] float in [-1,1]."""
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    return (arr / 127.5 - 1.0).transpose(3, 0, 1, 2).astype(np.float32)


# ------------------ SIGMark official noise-frame insertion ------------------

def _insert_noise(px: np.ndarray, groups: int, group_len: int,
                  seed: int = 0) -> np.ndarray:
    """Insert `groups` groups of `group_len` random-noise frames (SIGMark
    insert_frames style, scaled to 17-frame clips). Frame 0 stays first.
    Output T grows; detector side normalizes back to NUM_FRAMES."""
    rng = np.random.default_rng(seed)
    t, h, w = px.shape[1], px.shape[2], px.shape[3]
    pos = set(np.sort(rng.choice(np.arange(1, t), groups,
                                 replace=False)).tolist())
    cols = []
    for i in range(t):
        if i in pos:
            cols.append(rng.uniform(-1.0, 1.0,
                                    size=(3, group_len, h, w)).astype(np.float32))
        cols.append(px[:, i:i + 1])
    return np.concatenate(cols, axis=1)


# Attacks may change T; detector side handles normalization.
ATTACKS = {
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


# ----------------------------- SGO recovery (of_seg) -----------------------------

def sgo_recover(px: np.ndarray, tgt_num_frames: int = NUM_FRAMES) -> np.ndarray:
    """SIGMark segment_group_ordering, of_seg=True / sw_det=False branch,
    adapted to SKCW: no separate first conditioning frame is split off
    (SKCW watermarks all latent frames), so segmentation runs on all frames.

    min_segment_len lowered to 2: SKCW clips are 17 frames (SIGMark uses 81+),
    the default 8 would merge everything into one segment.
    """
    frames = px_to_frames(px)
    n = len(frames)
    segments_idx = optical_flow_segmentation(frames, min_segment_len=2)

    if n > tgt_num_frames:
        # Remove outlier-length segments (noise groups break flow continuity),
        # then align (tail-crop / pad) to target length.
        seg_lens = [end - start + 1 for start, end in segments_idx]
        most_common_len = max(set(seg_lens), key=seg_lens.count)
        kept = []
        for start, end in segments_idx:
            if end - start + 1 != most_common_len:
                kept.extend(frames[start:end + 1])
        if len(kept) == 0:  # degenerate: all segments same length
            kept = frames
        recovered = align_frames(kept, tgt_num_frames)
    elif n < tgt_num_frames:
        total_pad = tgt_num_frames - n
        num_segments = len(segments_idx)
        if num_segments == 1:
            recovered = [None] * tgt_num_frames
            off = total_pad // 2
            for i in range(n):
                recovered[off + i] = frames[i]
        else:
            base, rem = total_pad // num_segments, total_pad % num_segments
            gap_pads = [base + 1 if i < rem else base
                        for i in range(num_segments)]
            recovered = []
            for i, (start, end) in enumerate(segments_idx):
                recovered.extend([None] * gap_pads[i])
                recovered.extend(frames[start:end + 1])
        recovered = pad_video_by_interpolation(recovered)
    else:
        recovered = frames
    return frames_to_px(recovered)


def baseline_normalize(px: np.ndarray, tgt: int = NUM_FRAMES) -> np.ndarray:
    """No-SGO detector preprocessing: plain linear temporal resample."""
    return _resample_to(px, tgt)


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M5c temporal attacks with SGO")
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

    rng = np.random.default_rng(2027)  # same as M5b -> same messages
    cases = [(c, s) for c in HELDOUT_CLASSES for s in SEED_IDXS]
    log(f"cases: {len(cases)} latents x {len(ALPHAS)} alphas x "
        f"{len(ATTACKS)} attacks x 2 modes (nosgo/sgo)")

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
                att_wm = att_fn(px_wm)
                att_cl = att_fn(px_clean)
                for mode, recov in (("nosgo", baseline_normalize),
                                    ("sgo", sgo_recover)):
                    rec = dict(base, attack=att_name, mode=mode,
                               ts=time.strftime("%Y-%m-%d %H:%M:%S"))
                    try:
                        t0 = time.time()
                        z_obs_wm = runner.reencode_pixels(recov(att_wm))
                        rec["wm"] = detect(extr.extract(z_obs_wm), car, key,
                                           cw, msg)
                        z_obs_cl = runner.reencode_pixels(recov(att_cl))
                        rec["clean"] = detect(extr.extract(z_obs_cl), car,
                                              key, None, None)
                        rec["sec"] = round(time.time() - t0, 1)
                        rec["status"] = "ok"
                        n_ok += 1
                        log(f"c{c:02d}_s{sidx} a{alpha} {att_name} [{mode}]: "
                            f"chip={rec['wm']['chip_acc']} "
                            f"msg={rec['wm']['msg_bit_acc']} "
                            f"s_wm={rec['wm']['score']} "
                            f"s_clean={rec['clean']['score']}")
                    except Exception as exc:  # noqa: BLE001
                        rec["status"] = "error"
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                        rec["traceback"] = traceback.format_exc(limit=8)
                        n_fail += 1
                        log(f"    ERROR {att_name} [{mode}]: {rec['error']}")
                    append_jsonl(jsonl, rec)
                    torch.cuda.empty_cache()

    rows = [json.loads(l) for l in
            jsonl.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {}
    for att in ATTACKS:
        for alpha in ALPHAS:
            for mode in ("nosgo", "sgo"):
                sel = [r for r in rows if r["attack"] == att
                       and r["alpha"] == alpha and r["mode"] == mode]
                if not sel:
                    continue
                agg = lambda k1, k2: round(float(np.mean(  # noqa: E731
                    [r[k1][k2] for r in sel])), 4)
                summary[f"{att}|a{alpha}|{mode}"] = {
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
