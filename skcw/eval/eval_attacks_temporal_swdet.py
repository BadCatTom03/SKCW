#!/usr/bin/env python3
"""M5d: SKCW sw_det -- temporal realignment via slice-restricted PRC scoring.

Detection-side knows ONLY the class key (message unknown), so we cannot
correlate with the message-dependent delta directly.  Instead, per attacked
pixel frame i and candidate latent slice t:
  1. static-repeat frame i to a 17-frame clip -> VAE encode -> E_theta extract
  2. m_i = mean over latent time of C_hat  (spatial map [16,60,104])
  3. q_(i,t) = carrier demod with m_i placed ONLY at slice t (rest zeros)
  4. S[i,t] = prc_soft_decode(q_(i,t), key).score   (message-agnostic)
Assignment with capacities (1,4,4,4,4) (17 px frames -> 5 latent slices,
Wan VAE: first frame alone then 4 per latent slice) -> reorder groups by t
-> re-encode -> standard detect.  Within-group order kept (M5b: local
shuffle is harmless).

Modes per attack: noalign (M5b baseline) vs swdet (realigned).

Server:
  cd .
  nohup python eval_attacks_temporal_swdet.py \
      > eval_attacks_temporal_swdet_v6.log 2>&1 < /dev/null &
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
from skcw.eval.eval_attacks_temporal import attack_shuffle, attack_reverse  # noqa: E402

OUT_DIR = Path("./results/attack_eval_v6_temporal_swdet")

NUM_FRAMES = 17
N_SLICES = LATENT_SHAPE[1]            # 5 latent slices
CAPS = (1, 4, 4, 4, 4)                # px frames per latent slice (Wan VAE 1+4*4)

# focus on the attacks where M5b/M5c failed
ATTACKS = {
    "tshuffle_w4":     lambda px: attack_shuffle(px, 4),
    "tshuffle_global": lambda px: attack_shuffle(px, 999),
    "treverse":        attack_reverse,
}


# ----------------------------- sw_det aligner -----------------------------

class SwDetAligner:
    """Message-agnostic temporal localization for one class key.

    Precomputes per-slice demod tables: for slice t, demod restricted to the
    latent positions of slice t only.  Scoring a frame-vs-slice pair is then
    a single restricted demod + prc_soft_decode.
    """

    def __init__(self, car: PRCCarrier, key: bytes):
        self.car = car
        self.key = key
        c, t, h, w = car.shape
        self.slice_len = c * h * w
        # flat index ranges of each latent slice in [C,T,H,W] order:
        # axis order is (C,T,H,W) -> slice t occupies positions where the
        # T-index == t.  Build a boolean mask per slice once.
        idx = np.arange(c * t * h * w).reshape(c, t, h, w)
        self.slice_flat = [idx[:, tt].reshape(-1) for tt in range(t)]
        # per-slice block/g views for restricted demod
        self.block_s = [car.block[f] for f in self.slice_flat]
        self.g_s = [car.g[f] for f in self.slice_flat]

    def slice_score(self, spat_map: np.ndarray, t: int) -> float:
        """spat_map [C,H,W] (time-avg C_hat of a repeated-frame clip) placed
        at slice t -> PRC score (message-agnostic)."""
        r = spat_map.reshape(-1).astype(np.float64) * self.g_s[t]
        soft = np.bincount(self.block_s[t], weights=r, minlength=self.car.n)
        return float(prc_soft_decode(soft, self.key, DEFAULT_N)["score"])

    def score_matrix(self, spat_maps: list) -> np.ndarray:
        """spat_maps: per-pixel-frame [C,H,W] maps -> S [n_frames, N_SLICES]."""
        s = np.zeros((len(spat_maps), N_SLICES), dtype=np.float64)
        for i, m in enumerate(spat_maps):
            for t in range(N_SLICES):
                s[i, t] = self.slice_score(m, t)
        return s


def assign_frames(score: np.ndarray) -> np.ndarray:
    """S [17,5] -> slice id per frame, respecting CAPS, maximizing total score.
    Hungarian on capacity-expanded columns; greedy fallback if scipy missing."""
    n_f = score.shape[0]
    cols = []                      # expanded column -> slice id
    for t, cap in enumerate(CAPS):
        cols.extend([t] * cap)
    cols = np.array(cols)          # len == 17
    expanded = score[:, cols]      # [17,17]
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-expanded)
        out = np.empty(n_f, dtype=int)
        out[ri] = cols[ci]
        return out
    except Exception:              # noqa: BLE001  greedy fallback
        out = -np.ones(n_f, dtype=int)
        used = np.zeros(N_SLICES, dtype=int)
        order = np.dstack(np.unravel_index(
            np.argsort(score, axis=None)[::-1], score.shape))[0]
        for i, t in order:
            if out[i] == -1 and used[t] < CAPS[t]:
                out[i] = t
                used[t] += 1
        return out


def realign(px_att: np.ndarray, slice_of: np.ndarray) -> np.ndarray:
    """Reorder attacked frames so frames assigned to slice 0 come first, then
    slice 1, ... Within a slice keep the attacked-sequence order (local order
    is harmless per M5b)."""
    order = np.concatenate(
        [np.where(slice_of == t)[0] for t in range(N_SLICES)])
    return px_att[:, order].copy()


def probe_maps(px: np.ndarray, runner, extr) -> list:
    """Per pixel frame: static-repeat to 17 -> VAE encode -> E_theta ->
    time-avg -> spatial map [16,60,104]. Memoized by frame content hash
    (pure detector-side caching; identical input -> identical output)."""
    import hashlib
    maps, cache = [], {}
    for i in range(px.shape[1]):
        fr = px[:, i]
        h = hashlib.sha1(fr.tobytes()).hexdigest()
        if h not in cache:
            clip = np.repeat(fr[:, None], NUM_FRAMES, axis=1)
            z = runner.reencode_pixels(clip)
            cache[h] = extr.extract(z).mean(axis=1)   # [16,60,104]
        maps.append(cache[h])
    return maps


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M5d sw_det temporal realignment")
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

    rng = np.random.default_rng(2027)   # SAME seed as M5b/M5c -> same messages
    cases = [(c, s) for c in HELDOUT_CLASSES for s in SEED_IDXS]
    log(f"cases: {len(cases)} latents x {len(ALPHAS)} alphas x "
        f"{len(ATTACKS)} attacks x 2 modes (noalign/swdet)")

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
                    px_att_wm = att_fn(px_wm)
                    px_att_cl = att_fn(px_clean)
                    # --- noalign baseline ---
                    rec["noalign_wm"] = detect(
                        extr.extract(runner.reencode_pixels(px_att_wm)),
                        car, key, cw, msg)
                    rec["noalign_clean"] = detect(
                        extr.extract(runner.reencode_pixels(px_att_cl)),
                        car, key, None, None)
                    # --- sw_det realignment ---
                    maps = probe_maps(px_att_wm, runner, extr)
                    s_mat = algn.score_matrix(maps)
                    slice_of = assign_frames(s_mat)
                    rec["slice_assign"] = slice_of.tolist()
                    rec["swdet_wm"] = detect(
                        extr.extract(runner.reencode_pixels(
                            realign(px_att_wm, slice_of))),
                        car, key, cw, msg)
                    # clean side through the SAME procedure (FPR fairness)
                    maps_cl = probe_maps(px_att_cl, runner, extr)
                    slice_cl = assign_frames(algn.score_matrix(maps_cl))
                    rec["swdet_clean"] = detect(
                        extr.extract(runner.reencode_pixels(
                            realign(px_att_cl, slice_cl))),
                        car, key, None, None)
                    rec["sec"] = round(time.time() - t0, 1)
                    rec["status"] = "ok"
                    n_ok += 1
                    log(f"c{c:02d}_s{sidx} a{alpha} {att_name}: "
                        f"noalign msg={rec['noalign_wm']['msg_bit_acc']} "
                        f"swdet msg={rec['swdet_wm']['msg_bit_acc']} "
                        f"s_wm={rec['swdet_wm']['score']} "
                        f"s_cl={rec['swdet_clean']['score']}")
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc(limit=8)
                    n_fail += 1
                    log(f"    ERROR {att_name}: {rec['error']}")
                append_jsonl(jsonl, rec)
                torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {}
    for att in ATTACKS:
        for alpha in ALPHAS:
            for mode in ("noalign", "swdet"):
                sel = [r for r in rows
                       if r["attack"] == att and r["alpha"] == alpha]
                if not sel:
                    continue
                agg = lambda k1, k2: round(float(np.mean(  # noqa: E731
                    [r[k1][k2] for r in sel])), 4)
                summary[f"{att}|a{alpha}|{mode}"] = {
                    "n": len(sel),
                    "chip_acc": agg(f"{mode}_wm", "chip_acc"),
                    "msg_bit_acc": agg(f"{mode}_wm", "msg_bit_acc"),
                    "score_wm": agg(f"{mode}_wm", "score"),
                    "score_clean": agg(f"{mode}_clean", "score"),
                }
    report = {"ckpt": args.ckpt, "alphas": ALPHAS,
              "heldout_classes": list(HELDOUT_CLASSES),
              "caps": list(CAPS),
              "n_ok": n_ok, "n_fail": n_fail, "summary": summary,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "attack_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE ok={n_ok} fail={n_fail} -> {out_dir / 'attack_summary.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
