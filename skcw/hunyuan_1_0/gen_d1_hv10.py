#!/usr/bin/env python3
"""H1: generate HV-1.0 D1 clean latents for E_theta-HV training + eval.

Mirrors gen_d1_clean_latents.py but on HunyuanVideo-1.0 (latent 16,17,60,60).
Each sample saves z_gen (scaled model latent) and z_obs (VAE round-trip).

Default small set (cost control; HV gen ~190s/clip per H0):
  train classes 0-27: 2 seeds each   = 56
  held-out 28-31    : 2 seeds each   = 8
  -> 64 clips, ~3.5 h. Use --seeds_per_class / --classes / --smoke to scale.

Output: results/d1_latents_hv10/c{cc}_s{s}.npz (+ manifest.jsonl)

Server:
  cd .
  nohup python gen_d1_hv10.py > d1_hv10_gen.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import PROMPTS32, TRAIN_CLASSES, HELDOUT_CLASSES  # noqa: E402
from skcw.hunyuan_1_0.hv10_runner import HV10Runner, LATENT_SHAPE, log  # noqa: E402

OUT_DIR = Path("./results/d1_latents_hv10")
SEED_BASE = 41000


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="H1: HV-1.0 D1 latent generation")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds_per_class", type=int, default=2)
    ap.add_argument("--heldout_seeds", type=int, default=2)
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    if args.smoke:
        jobs = [(0, 0), (28, 0)]
    else:
        classes = args.classes if args.classes else sorted(PROMPTS32.keys())
        jobs = []
        for c in classes:
            n = args.heldout_seeds if c in HELDOUT_CLASSES else args.seeds_per_class
            jobs += [(c, si) for si in range(n)]
    todo = [(c, si) for (c, si) in jobs
            if not (out_dir / f"c{c:02d}_s{si}.npz").exists()]
    log(f"H1 jobs total={len(jobs)} todo={len(todo)} (resume by file)")
    if not todo:
        log("nothing to do")
        return 0

    runner = HV10Runner(device=args.device, out_dir=out_dir)
    import torch

    n_ok = n_fail = 0
    for idx, (c, si) in enumerate(todo, 1):
        seed = SEED_BASE + c * 100 + si
        tag = f"c{c:02d}_s{si}"
        log(f"=== [{idx}/{len(todo)}] {tag} seed={seed} ===")
        t0 = time.time()
        try:
            z_gen = runner.generate(PROMPTS32[c], seed)[0].numpy()   # [16,17,60,60]
            px = runner.latents_to_pixels_np(z_gen)                  # [3,65,480,480]
            z_obs = runner.reencode_pixels(px)                       # [16,17,60,60]
            assert tuple(z_gen.shape) == LATENT_SHAPE, z_gen.shape
            np.savez_compressed(out_dir / f"{tag}.npz",
                                z_gen=z_gen.astype(np.float16),
                                z_obs=z_obs.astype(np.float16))
            rec = {"tag": tag, "class_id": c, "seed_idx": si, "seed": seed,
                   "z_gen_std": round(float(z_gen.std()), 4),
                   "z_obs_std": round(float(z_obs.std()), 4),
                   "rt_l1": round(float(np.abs(z_gen - z_obs).mean()), 4),
                   "shape": list(z_gen.shape),
                   "sec": round(time.time() - t0, 1), "status": "ok",
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            n_ok += 1
            log(f"    ok std={rec['z_gen_std']} rt_l1={rec['rt_l1']} ({rec['sec']}s)")
        except Exception as exc:  # noqa: BLE001
            rec = {"tag": tag, "class_id": c, "seed_idx": si, "seed": seed,
                   "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=8),
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            n_fail += 1
            log(f"    ERROR: {rec['error']}")
        append_jsonl(manifest, rec)
        torch.cuda.empty_cache()

    log(f"DONE ok={n_ok} fail={n_fail} -> {out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
