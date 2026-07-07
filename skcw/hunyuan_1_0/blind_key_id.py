#!/usr/bin/env python3
"""Blind key identification (no prompt at extraction time).

At extraction the verifier does NOT know which class key was used and has NO
original prompt -- only the video pixels.  We must recover the key from the
video alone.  Three measurements, each isolating a different guarantee:

  Stage A  Full-library argmax (CORRECTNESS LOWER BOUND)
    Extract the carrier residual C_hat ONCE (key-independent), then decode it
    with EVERY one of the K=32 class keys, pick argmax score -> predicted class.
    Build a K x K confusion matrix.  Top-1 accuracy here is the *provable*
    blind-ID correctness -- it does NOT depend on CLIP at all.

  Stage B  CLIP candidate shortlist (EARLY-STOP SPEED, not correctness)
    Embed the decoded video frames with CLIP image encoder, rank the 32 class
    *prompt* texts by cosine similarity, take Top-k.  Measures how often the
    true class is in CLIP's Top-k (k=1,3,5) -> how often we can skip the full
    sweep.  CLIP's image/text gap can only slow us down, never make us wrong,
    because the PRC score is itself the decisive discriminator and we fall back
    to the full library when no shortlisted key clears the threshold.

  Stage C  Calibrated FPR (FALSE-POSITIVE CONTROL)
    Round-trip the *clean* (un-watermarked) z_gen of every class, decode with
    all 32 keys -> H0 score pool.  Set a per-key threshold tau(K) by Bonferroni
    correction for K simultaneous keys, then report the fraction of clean videos
    where ANY key crosses tau (the realised blind FPR).

Reuses results/d1_latents_hv10/c{cc}_s0.npz (z_gen) -> NO video generation,
only one VAE decode+encode per class.

Server:
  cd .
  nohup python blind_key_id.py > blind_key_id.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


from skcw.core.prc_code import DEFAULT_N, prc_encode, prc_soft_decode, msg_bit_acc  # noqa: E402
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402
from skcw.hunyuan_1_0.hv10_runner import HV10Runner, LATENT_SHAPE, log  # noqa: E402
from skcw.core.prompts32 import PROMPTS32, HELDOUT_CLASSES  # noqa: E402

MASTER_SEED = "skcw-v0.3-p0"
CW = Path(".")
D1_DIR = CW / "results/d1_latents_hv10"
OUT_DIR = CW / "results/blind_key_id"
CKPT = CW / "results/e_theta_hv10/last.pt"
CLIP_DIR = Path("./pretrained_models/clip-vit-large-patch14")

ALPHA = 0.4
SEED_IDX = 0
ALL_CLASSES = list(range(32))


class Extractor:
    """E_theta carrier extractor.  C_hat is KEY-INDEPENDENT: compute once,
    decode against any candidate key for free."""

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
        log(f"E_theta-HV loaded: step={ck.get('step')} s={self.s:.4f}")

    def extract(self, z_np: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            z = torch.from_numpy(z_np[None].astype(np.float32)).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(self.device != "cpu")):
                hf = self.highpass(z)
                c_hat = self.s * hf + self.model(hf)
        return c_hat.float().cpu().numpy()[0]


class ClipRanker:
    """Rank 32 class prompts by CLIP cosine similarity to the decoded video.
    Image/text gap is tolerated: this only produces a *shortlist*."""

    def __init__(self, clip_dir: Path, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(str(clip_dir)).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(str(clip_dir))
        # pre-compute normalised text features for all 32 class prompts.
        # Use low-level text_model + projection (get_text_features differs across
        # transformers versions and may return a ModelOutput instead of a tensor).
        texts = [PROMPTS32[c] for c in ALL_CLASSES]
        with torch.no_grad():
            tin = self.proc(text=texts, return_tensors="pt", padding=True,
                            truncation=True).to(device)
            to = self.model.text_model(**tin)
            tf = self.model.text_projection(to.pooler_output)
            self.txt = (tf / tf.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        log(f"CLIP ranker ready: {self.txt.shape[0]} class-text features")

    def rank(self, pixels_np: np.ndarray, n_frames: int = 6):
        """pixels [3,T,H,W] in [-1,1] -> class indices sorted by similarity desc."""
        from PIL import Image
        torch = self.torch
        T = pixels_np.shape[1]
        idx = np.linspace(0, T - 1, num=min(n_frames, T)).astype(int)
        imgs = []
        for t in idx:
            arr = ((pixels_np[:, t] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            imgs.append(Image.fromarray(arr.transpose(1, 2, 0)))
        with torch.no_grad():
            pin = self.proc(images=imgs, return_tensors="pt").to(self.device)
            vo = self.model.vision_model(**pin)
            f = self.model.visual_projection(vo.pooler_output)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0)
            f = (f / f.norm()).float().cpu().numpy()
        sim = self.txt @ f                       # [32]
        order = np.argsort(-sim)                  # class indices, best first
        return order.tolist(), sim.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--alpha_fpr", type=float, default=1e-6,
                    help="target overall false-positive rate per video")
    ap.add_argument("--no_clip", action="store_true")
    args = ap.parse_args()

    import torch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(ALL_CLASSES)

    # per-class key / carrier / embedded message
    keys = {c: derive_class_key(MASTER_SEED, c) for c in ALL_CLASSES}
    cars = {c: PRCCarrier(keys[c], LATENT_SHAPE, DEFAULT_N) for c in ALL_CLASSES}
    rng = np.random.default_rng(31337)
    msgs = {c: int(rng.integers(0, 2 ** 32)) for c in ALL_CLASSES}
    cws = {c: prc_encode(msgs[c], keys[c], DEFAULT_N) for c in ALL_CLASSES}

    extr = Extractor(Path(args.ckpt), args.device)
    runner = HV10Runner(device=args.device, out_dir=out_dir)
    clip = None
    if not args.no_clip:
        try:
            clip = ClipRanker(CLIP_DIR, args.device)
        except Exception as exc:  # noqa: BLE001
            log(f"[WARN] CLIP unavailable ({exc}); skipping Stage B")

    # ---- Stage A & B over watermarked videos -------------------------------
    score_wm = np.zeros((K, K))      # [i][j] decode C_hat_i with key_j
    bitacc_wm = np.zeros((K, K))
    clip_order = {}                  # class -> CLIP-ranked class list
    t_extract = []                   # C_hat extraction wall time (once/video)
    t_decode = []                    # single key decode time

    # ---- Stage C: clean round-trips for the H0 / FPR pool ------------------
    score_h0 = np.zeros((K, K))      # [i][j] clean video i decoded with key j

    for i, ci in enumerate(ALL_CLASSES):
        npz = D1_DIR / f"c{ci:02d}_s{SEED_IDX}.npz"
        z_gen = np.load(npz)["z_gen"].astype(np.float32)

        # --- watermarked branch ---
        z_wm = z_gen + args.alpha * cars[ci].make_delta(cws[ci])
        px_wm = runner.latents_to_pixels_np(z_wm)
        z_re = runner.reencode_pixels(px_wm)
        t0 = time.time()
        c_hat = extr.extract(z_re)
        t_extract.append(time.time() - t0)
        for j, cj in enumerate(ALL_CLASSES):
            td = time.time()
            soft = cars[cj].demodulate(c_hat)
            dec = prc_soft_decode(soft, keys[cj], DEFAULT_N)
            t_decode.append(time.time() - td)
            score_wm[i][j] = round(float(dec["score"]), 4)
            bitacc_wm[i][j] = round(float(msg_bit_acc(dec, msgs[ci])), 4)
        if clip is not None:
            order, _ = clip.rank(px_wm)
            clip_order[ci] = order

        # --- clean branch (H0) ---
        px_cl = runner.latents_to_pixels_np(z_gen)
        z_cl = runner.reencode_pixels(px_cl)
        c_hat0 = extr.extract(z_cl)
        for j, cj in enumerate(ALL_CLASSES):
            soft0 = cars[cj].demodulate(c_hat0)
            dec0 = prc_soft_decode(soft0, keys[cj], DEFAULT_N)
            score_h0[i][j] = round(float(dec0["score"]), 4)

        pred = int(np.argmax(score_wm[i]))
        log(f"[c{ci:02d}] argmax-key={pred} (true={ci}) "
            f"match_score={score_wm[i][i]:.3f} max_other="
            f"{max(score_wm[i][j] for j in range(K) if j!=i):.3f} "
            f"bitacc={bitacc_wm[i][i]:.3f}"
            + (f" clip_rankOfTrue={clip_order[ci].index(ci)}" if clip else ""))
        torch.cuda.empty_cache()

    # ---- Stage A metrics: Top-1 / confusion --------------------------------
    pred_key = np.array([int(np.argmax(score_wm[i])) for i in range(K)])
    top1 = float(np.mean(pred_key == np.arange(K)))
    confusion = np.zeros((K, K), dtype=int)
    for i in range(K):
        confusion[i][pred_key[i]] += 1

    # ---- Stage B metrics: CLIP Top-k hit + early-stop ----------------------
    clip_metrics = None
    if clip is not None:
        ranks = np.array([clip_order[c].index(c) for c in ALL_CLASSES])  # 0-based
        clip_metrics = {
            "hit_at_1": round(float(np.mean(ranks < 1)), 4),
            "hit_at_3": round(float(np.mean(ranks < 3)), 4),
            "hit_at_5": round(float(np.mean(ranks < 5)), 4),
            "mean_rank_of_true": round(float(ranks.mean()), 3),
            "rank_of_true": ranks.tolist(),
        }

    # ---- Stage C metrics: Bonferroni threshold + FPR -----------------------
    h0 = score_h0.reshape(-1)
    h0_mean, h0_std = float(h0.mean()), float(h0.std())
    # per-key single-test alpha after Bonferroni for K simultaneous keys
    alpha_single = args.alpha_fpr / K
    # normal-tail quantile (score ~ approx Normal under H0 by CLT over 32 bits)
    z_q = math.sqrt(2.0) * _erfinv(1.0 - 2.0 * alpha_single)
    tau = h0_mean + z_q * h0_std
    # realised FPR: a CLEAN video is a false positive if ANY key exceeds tau
    fp_per_video = np.array([int(np.any(score_h0[i] >= tau)) for i in range(K)])
    fpr_emp = float(fp_per_video.mean())
    # detection power on watermarked: matched key clears tau?
    det_power = float(np.mean([score_wm[i][i] >= tau for i in range(K)]))

    diag = np.array([score_wm[i][i] for i in range(K)])
    off = np.array([score_wm[i][j] for i in range(K) for j in range(K) if i != j])

    report = {
        "experiment": "blind key identification (no prompt at extraction)",
        "model": "HunyuanVideo-1.0", "alpha": args.alpha,
        "n_classes": K, "heldout_classes": HELDOUT_CLASSES,
        "stage_A_full_library_argmax": {
            "top1_accuracy": round(top1, 4),
            "n_correct": int((pred_key == np.arange(K)).sum()),
            "matched_score": {"mean": round(float(diag.mean()), 4),
                              "min": round(float(diag.min()), 4)},
            "mismatched_score": {"mean": round(float(off.mean()), 4),
                                 "max": round(float(off.max()), 4),
                                 "std": round(float(off.std()), 4)},
            "separation_margin": round(float(diag.min() - off.max()), 4),
        },
        "stage_B_clip_shortlist": clip_metrics,
        "stage_C_fpr": {
            "target_alpha_fpr": args.alpha_fpr,
            "h0_score_mean": round(h0_mean, 4), "h0_score_std": round(h0_std, 4),
            "h0_score_max": round(float(h0.max()), 4),
            "bonferroni_alpha_single": alpha_single,
            "tau_threshold": round(float(tau), 4),
            "empirical_fpr_per_video": round(fpr_emp, 6),
            "detection_power_watermarked": round(det_power, 4),
            "matched_min_minus_tau": round(float(diag.min() - tau), 4),
        },
        "timing": {
            "c_hat_extract_s_mean": round(float(np.mean(t_extract)), 4),
            "single_key_decode_s_mean": round(float(np.mean(t_decode)), 6),
            "full_32key_sweep_s_est": round(float(np.mean(t_decode) * K), 4),
        },
        "score_matrix_wm": score_wm.tolist(),
        "bitacc_matrix_wm": bitacc_wm.tolist(),
        "score_matrix_h0": score_h0.tolist(),
        "confusion": confusion.tolist(),
        "pred_key": pred_key.tolist(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "blind_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE top1={top1:.4f} fpr={fpr_emp:.2e} tau={tau:.3f} "
        f"det_power={det_power:.4f} "
        + (f"clip_hit@1={clip_metrics['hit_at_1']:.3f} "
           f"hit@5={clip_metrics['hit_at_5']:.3f} " if clip_metrics else "")
        + f"-> {out_dir/'blind_summary.json'}")
    return 0


def _erfinv(y: float) -> float:
    """Inverse error function, stdlib-only.  scipy if present, else a Winitzki
    seed refined by Newton iteration on math.erf (accurate into the 1e-8 tail)."""
    try:
        from scipy.special import erfinv as _sci  # type: ignore
        return float(_sci(y))
    except Exception:  # noqa: BLE001
        pass
    a = 0.147
    ln = math.log(max(1.0 - y * y, 1e-300))
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    x = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)
    # Newton refinement: f(x)=erf(x)-y, f'(x)=2/sqrt(pi)*exp(-x^2)
    c = 2.0 / math.sqrt(math.pi)
    for _ in range(6):
        err = math.erf(x) - y
        if abs(err) < 1e-15:
            break
        x -= err / (c * math.exp(-x * x))
    return x


if __name__ == "__main__":
    sys.exit(main())
