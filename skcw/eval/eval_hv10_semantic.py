#!/usr/bin/env python3
"""Semantic-stratified clean detection: OURS (E_theta-HV) on HV-1.0.

For each VBench2 semantic dimension we generate a few clips (different prompts
& seeds), embed our PRC carrier, decode->reencode (clean, NO attack), and run
the E_theta extractor.  bit_acc is reported per dimension so the curve shows
how detection accuracy varies across semantics.  This is the OURS side of the
OURS-vs-SIGMark semantic comparison (SIGMark side uses the SAME prompts/seeds).

Server:
  cd /LTX/ltx2.3/Semantic-STRR/condition_watermark
  nohup /opt/conda/bin/python eval_hv10_semantic.py \
      > eval_hv10_semantic.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from prompts32 import HELDOUT_CLASSES  # noqa: E402
from prc_code import DEFAULT_N, MSG_BITS, prc_encode, prc_soft_decode, msg_bit_acc  # noqa: E402
from carriers_prc import PRCCarrier  # noqa: E402
from carriers import derive_class_key  # noqa: E402
from hv10_runner import HV10Runner, LATENT_SHAPE, psnr, log  # noqa: E402

MASTER_SEED = "skcw-v0.3-p0"
PROMPT_SET_DIR = Path("/LTX/ltx2.3/Semantic-STRR/prompt_set/VBench2_aug_prompt")
OUT_DIR = Path("/LTX/ltx2.3/Semantic-STRR/condition_watermark/results/eval_hv10_semantic")
OUR_DIR = Path("/LTX/ltx2.3/Semantic-STRR/condition_watermark/results/eval_hv10_semantic_mb512")
CKPT = Path("/LTX/ltx2.3/Semantic-STRR/condition_watermark/results/e_theta_hv10/last.pt")

ALPHA = 0.4                 # main operating point
MSG_BITS_DEFAULT = MSG_BITS # 32 (from prc_code)
PROMPTS_PER_DIM = 2         # prompts sampled per semantic dimension
SEEDS_PER_PROMPT = 2        # seeds per prompt
SEED_BASE = 70000           # generation seed base (distinct from D1)


# ---- prompt sampling: mirror SIGMark VBench2PromptSet interval logic ----
# So OURS and SIGMark evaluate on the EXACT same prompts per dimension.
def list_dimensions(prompt_dir: Path) -> list[str]:
    dims = sorted(p.stem for p in prompt_dir.glob("*.txt"))
    # Match the 18 VBench2 semantic dimensions used in the paper table.
    # These two dimensions are present in some augmented prompt sets but were
    # not part of the reported 18-dimension comparison.
    excluded = {"Emotions_Expressions", "Medical_Health"}
    return [d for d in dims if d not in excluded]


def sample_prompts(prompt_dir: Path, dimension: str, num_prompts: int) -> list[str]:
    """Replicates VBench2PromptSet's subset selection (interval sampling)."""
    lines = (prompt_dir / f"{dimension}.txt").read_text(
        encoding="utf-8").splitlines()
    prompts = [l.strip() for l in lines if l.strip()]
    if num_prompts >= len(prompts):
        return prompts
    interval = len(prompts) // num_prompts
    idx, mod = [], 0
    while len(idx) < num_prompts:
        idx.extend([i for i in range(len(prompts)) if i % interval == mod])
        mod += 1
    idx = sorted(idx)[:num_prompts]
    return [prompts[i] for i in idx]


# ----------------------------- extractor -----------------------------
class Extractor:
    def __init__(self, ckpt_path: Path, device: str):
        import torch
        from e_theta_model import ETheta
        from train_e_theta import highpass
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


def detect(c_hat, car, key, codeword, message, msg_bits: int = MSG_BITS) -> dict:
    soft = car.demodulate(c_hat)
    dec = prc_soft_decode(soft, key, DEFAULT_N, msg_bits=msg_bits)
    hard = (soft > 0).astype(np.uint8)
    return {"score": round(float(dec["score"]), 4),
            "chip_acc": round(float((hard == codeword).mean()), 4),
            "msg_bit_acc": round(float(msg_bit_acc(dec, message)), 4)}


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------- main -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="OURS semantic-stratified clean eval")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--prompt_dir", default=str(PROMPT_SET_DIR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--msg_bits", type=int, default=MSG_BITS_DEFAULT,
                    help=f"message bits per clip (default {MSG_BITS_DEFAULT}); set 512 for "
                         f"fair-capacity comparison with SIGMark at 512-bit")
    ap.add_argument("--prompts_per_dim", type=int, default=PROMPTS_PER_DIM)
    ap.add_argument("--seeds_per_prompt", type=int, default=SEEDS_PER_PROMPT)
    ap.add_argument("--smoke", action="store_true",
                    help="only first 2 dimensions, 1 prompt, 1 seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "semantic_results.jsonl"
    prompt_dir = Path(args.prompt_dir)

    dims = list_dimensions(prompt_dir)
    n_pp, n_sp = args.prompts_per_dim, args.seeds_per_prompt
    if args.smoke:
        dims, n_pp, n_sp = dims[:2], 1, 1

    extr = Extractor(Path(args.ckpt), args.device)
    runner = HV10Runner(device=args.device, out_dir=out_dir)
    import torch

    # one fixed held-out class/key for the whole semantic sweep (semantics is the
    # variable under study, not the class).
    cls = HELDOUT_CLASSES[0]
    key = derive_class_key(MASTER_SEED, cls)
    car = PRCCarrier(key, LATENT_SHAPE, DEFAULT_N)
    rng = np.random.default_rng(7070)

    log(f"{len(dims)} dims x {n_pp} prompts x {n_sp} seeds | alpha={args.alpha} "
        f"class={cls} msg_bits={args.msg_bits}")
    n_ok = n_fail = 0
    for di, dim in enumerate(dims):
        prompts = sample_prompts(prompt_dir, dim, n_pp)
        for pi, prompt in enumerate(prompts):
            for si in range(n_sp):
                seed = SEED_BASE + di * 1000 + pi * 100 + si
                rec = {"dimension": dim, "prompt_idx": pi, "seed_idx": si,
                       "seed": seed, "prompt": prompt[:200],
                       "alpha": args.alpha, "class_id": cls,
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                try:
                    t0 = time.time()
                    z_gen = runner.generate(prompt, seed)[0].numpy().astype(
                        np.float32)
                    # generate a random message of msg_bits width
                    if args.msg_bits <= 64:
                        msg = int(rng.integers(0, 2 ** args.msg_bits))
                    else:
                        msg = int.from_bytes(
                            rng.bytes(args.msg_bits // 8), byteorder='big')
                    cw = prc_encode(msg, key, DEFAULT_N, msg_bits=args.msg_bits)
                    z_wm = z_gen + args.alpha * car.make_delta(cw)
                    px_wm = runner.latents_to_pixels_np(z_wm)
                    px_clean = runner.latents_to_pixels_np(z_gen)
                    rec["psnr_wm_vs_clean"] = round(psnr(px_wm, px_clean), 3)
                    # clean roundtrip: reencode the watermarked pixels, extract.
                    z_re = runner.reencode_pixels(px_wm)
                    rec["det"] = detect(extr.extract(z_re), car, key, cw, msg,
                                         msg_bits=args.msg_bits)
                    rec["sec"] = round(time.time() - t0, 1)
                    rec["status"] = "ok"
                    n_ok += 1
                    log(f"[{dim}] p{pi}s{si} bit_acc={rec['det']['msg_bit_acc']} "
                        f"chip={rec['det']['chip_acc']} score={rec['det']['score']} "
                        f"psnr={rec['psnr_wm_vs_clean']}")
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "error"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc(limit=8)
                    n_fail += 1
                    log(f"    ERROR [{dim}] p{pi}s{si}: {rec['error']}")
                append_jsonl(jsonl, rec)
                torch.cuda.empty_cache()

    # aggregate per dimension
    rows = [json.loads(l) for l in jsonl.read_text(
        encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok"]
    by_dim: dict = {}
    for dim in dims:
        sel = [r for r in rows if r["dimension"] == dim]
        if not sel:
            continue
        ba = np.array([r["det"]["msg_bit_acc"] for r in sel])
        ca = np.array([r["det"]["chip_acc"] for r in sel])
        sc = np.array([r["det"]["score"] for r in sel])
        by_dim[dim] = {
            "n": len(sel),
            "msg_bit_acc_mean": round(float(ba.mean()), 4),
            "msg_bit_acc_std": round(float(ba.std()), 4),
            "msg_bit_acc_min": round(float(ba.min()), 4),
            "chip_acc_mean": round(float(ca.mean()), 4),
            "score_mean": round(float(sc.mean()), 4),
        }
    overall = {
        "msg_bit_acc_mean": round(float(np.mean(
            [r["det"]["msg_bit_acc"] for r in rows])), 4) if rows else None,
        "msg_bit_acc_std": round(float(np.std(
            [r["det"]["msg_bit_acc"] for r in rows])), 4) if rows else None,
    }
    report = {"method": "OURS-E_theta-HV", "model": "HunyuanVideo-1.0",
              "alpha": args.alpha, "msg_bits": args.msg_bits,
              "ckpt": args.ckpt,
              "prompts_per_dim": n_pp, "seeds_per_prompt": n_sp,
              "n_ok": n_ok, "n_fail": n_fail,
              "overall": overall, "by_dimension": by_dim,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "semantic_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE ok={n_ok} fail={n_fail} -> {out_dir / 'semantic_summary.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
