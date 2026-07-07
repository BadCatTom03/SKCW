#!/usr/bin/env python3
"""M2: train E_theta (B2 direct carrier estimation) -- plan v2 Sections 6-7.

Task (run-6 residual parameterization):
    C_hat = s * hf(Z_obs) + r_net(hf(Z_obs)),  s learnable scalar (init 1)
  The zero-parameter hf+demod oracle (strong msg 0.974 / chip 0.694, diag
  2026-06-12) is the s=1, r_net=0 point of this space -- the zero-output
  basin that swallowed runs 2-5 is ELIMINATED: net collapse degrades to the
  oracle, not to chance.
Loss:   L = L_carrier + lambda1 * L_demod          (run-6: L_null deleted --
  it only dug the zero basin deeper; zero-output s_clean was 0.8 anyway, no
  real discrimination benefit)
  L_carrier : scale-invariant cosine alignment with g*alpha*delta (run-4:
              per-element L1 made zero-output the global optimum on real
              latents; only the keyed-subspace projection is recoverable)
  L_demod   : PRC soft-chip margin -- differentiable demod of C_hat, hinge on
              sign agreement with codeword (MAIN loss, lambda=1.0); run-6:
              margin SNR-scaled by ref/0.14 (weak-end per-chip ceiling
              Phi(0.2)~0.58 made fixed margin=1 unsatisfiable -> the
              never-extinguishing gradient caused the run-4/5 collapses)
Warm start (run-6): --resume best.pt --resume_model_only --lr 1e-4 loads the
  run-5 warmup solution (probe chip 0.910) with a FRESH optimizer.

Data:
  D1 npz files (z_gen [unused here], z_obs = VAE round-trip clean latent)
  D2 online synthetic: Z_in = z_obs + g_sim*alpha*delta (+ aug noise),
     g_sim ~ U[0.25,0.45], alpha ~ U[0.2,0.5] (run-5: weak end tightened so
     min g*alpha = 0.05 matches the hinge ref clamp; old 0.02 floor made the
     hinge target impossible and collapsed run-4 after a perfect warmup)
  Aug: gaussian noise std<=0.015 (run-5: halved), per-channel gain jitter
     <=0.02.  (Real-pipeline D3 pairs come later in M3.)
  Run-5 anti-forgetting: anneal stretched to 16000, 1/3 strong-regime replay
  after warmup, fixed strong probe each eval + best.pt on probe chip_acc.

Smoke (M2 gate, runs BEFORE D1 finishes):
  --smoke uses 8 synthetic random-content latents (std 0.45) and overfits;
  pass line: watermarked chip_acc(C_hat) >= 0.95 AND null score < 1.5.

Server:
  cd .
  nohup python train_e_theta.py --smoke > m2_smoke.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


from skcw.core.prompts32 import TRAIN_CLASSES, HELDOUT_CLASSES  # noqa: E402
from skcw.core.prc_code import (DEFAULT_N, prc_encode, prc_soft_decode,  # noqa: E402
                      msg_bit_acc)
from skcw.core.carriers_prc import PRCCarrier  # noqa: E402
from skcw.core.carriers import derive_class_key  # noqa: E402

MASTER_SEED = "skcw-v0.3-p0"  # same as P0 -> same class keys
LATENT_SHAPE = (16, 5, 60, 104)
DEFAULT_OUT = Path("./results/e_theta_b2")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------- data -----------------------------

class CarrierBank:
    """One PRCCarrier per class (lazy), torch buffers cached on device."""

    def __init__(self, device):
        self.device = device
        self._cars: dict = {}
        self._bufs: dict = {}

    def get(self, class_id: int) -> PRCCarrier:
        if class_id not in self._cars:
            key = derive_class_key(MASTER_SEED, class_id)
            self._cars[class_id] = (key, PRCCarrier(key, LATENT_SHAPE, DEFAULT_N))
        return self._cars[class_id]

    def bufs(self, class_id: int):
        if class_id not in self._bufs:
            _, car = self.get(class_id)
            self._bufs[class_id] = car.torch_buffers(self.device)
        return self._bufs[class_id]


def load_d1(d1_dir: Path, classes) -> dict:
    """class_id -> list of z_obs fp32 arrays. Empty dict if dir missing."""
    out: dict = {}
    if not d1_dir.exists():
        return out
    for f in sorted(d1_dir.glob("c*_s*.npz")):
        c = int(f.name[1:3])
        if c not in classes:
            continue
        try:
            z = np.load(f)["z_obs"].astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        if z.shape == LATENT_SHAPE:
            out.setdefault(c, []).append(z)
    return out


def make_synthetic_pool(n: int, rng: np.random.Generator) -> dict:
    """Smoke fallback: SMOOTH random 'content' latents, std 0.45.

    Round-3 root cause: iid-noise content is (a) per-element
    indistinguishable from the iid carrier and (b) unmemorizable by a
    translation-equivariant conv net, so the optimal local output is ~0 and
    the normalized carrier loss pins at exactly 1.0 (observed 1.10 plateau).
    Real VAE latents are spatially smooth; low-frequency content makes
    'suppress the locally-predictable content, keep the high-frequency
    carrier' a learnable local operation -- which is the actual task.
    """
    import torch
    import torch.nn.functional as F
    pool: dict = {}
    c, t, h, w = LATENT_SHAPE
    for i in range(n):
        coarse = rng.normal(0, 1.0, (1, c, max(t // 2, 1), h // 4, w // 4))
        z = F.interpolate(torch.from_numpy(coarse).float(),
                          size=(t, h, w), mode="trilinear",
                          align_corners=False)[0].numpy()
        z = (z / z.std()) * 0.45
        pool.setdefault(i % 4, []).append(z.astype(np.float32))
    return pool


def curriculum(step: int, warm_end: int = 4000, anneal_end: int = 16000,
               full_a=(0.2, 0.5), full_g=(0.25, 0.45),
               full_noise: float = 0.015, full_jitter: float = 0.02):
    """M3 run-3 fix: run-2 (defaults lr=2e-4, lam_d=0.1, warmup=1000) pinned
    at the 1.10 zero-output plateau on real D1 latents -- L_null pushed the
    output to zero during warmup, then the weak demod gradient could not pull
    it out, while aug noise (std 0.03 ~= 10x the ~3e-3 carrier amplitude)
    drowned the signal from step 1.

    Run-5 fix (run-4 collapsed AFTER a perfect warmup: chip 0.91/msg 1.0 at
    step 3000 -> chip 0.51 by step 15000 = catastrophic forgetting, not
    hard-sample difficulty):
      * weak end tightened: old min g*alpha = 0.1*0.2 = 0.02 < old hinge
        ref clamp 0.05 -> hinge demanded a physically impossible 2.5x
        amplification on weak samples, saturating the loss and steering the
        net away from the learned solution.  New min g*alpha = 0.2*0.25 =
        0.05 (matches the deployable operating floor).
      * full_noise 0.03 -> 0.015 (old value exceeded the weak-end signal),
        jitter 0.03 -> 0.02.
      * anneal stretched 10000 -> 16000 (lr has decayed further before the
        hardest distribution arrives).
      * (in train loop) 1/3 strong-regime replay + fixed strong probe with
        best.pt checkpointing.

    Curriculum: start in the smoke-proven strong/narrow injection regime with
    NO augmentation, then linearly anneal to the full operating range.
      p=0 (step<=warm_end): alpha in [0.3,0.5], g in [0.3,0.45], no noise/jitter
      p=1 (step>=anneal_end): full ranges + full augmentation
    Returns (alpha_rng, g_rng, noise_std, jitter_amt).
    """
    if step <= warm_end:
        p = 0.0
    elif step >= anneal_end:
        p = 1.0
    else:
        p = (step - warm_end) / float(anneal_end - warm_end)
    a_lo = 0.3 + p * (full_a[0] - 0.3)
    g_lo = 0.3 + p * (full_g[0] - 0.3)
    return ((a_lo, full_a[1]), (g_lo, full_g[1]),
            full_noise * p, full_jitter * p)


def sample_batch(pool: dict, bank: CarrierBank, batch: int, rng,
                 p_clean: float = 0.3, alpha_rng=(0.1, 0.5),
                 g_rng=(0.2, 0.45), noise_std: float = 0.03,
                 gain_jitter: float = 0.03):
    """Returns numpy arrays: z_in [B,...], target [B,...], meta list."""
    classes = list(pool.keys())
    z_in = np.empty((batch, *LATENT_SHAPE), dtype=np.float32)
    target = np.zeros_like(z_in)
    meta = []
    # deterministic clean count (random Bernoulli gave all-clean batches at
    # small batch sizes -> zero carrier supervision on those steps)
    n_clean = int(round(p_clean * batch))
    clean_flags = np.zeros(batch, dtype=bool)
    clean_flags[:n_clean] = True
    rng.shuffle(clean_flags)
    for b in range(batch):
        c = int(rng.choice(classes))
        z = pool[c][int(rng.integers(len(pool[c])))]
        clean = bool(clean_flags[b])
        if clean:
            zi = z.copy()
            meta.append({"clean": True, "class_id": c})
        else:
            key, car = bank.get(c)
            msg = int(rng.integers(0, 2 ** 32))
            cw = prc_encode(msg, key, DEFAULT_N)
            delta = car.make_delta(cw)
            alpha = float(rng.uniform(*alpha_rng))
            g = float(rng.uniform(*g_rng))
            cw_field = (g * alpha) * delta
            zi = z + cw_field
            target[b] = cw_field
            meta.append({"clean": False, "class_id": c, "message": msg,
                         "alpha": alpha, "g": g, "codeword": cw})
        if noise_std > 0:
            zi = zi + rng.normal(0, noise_std, LATENT_SHAPE).astype(np.float32)
        if gain_jitter > 0:
            # per-channel gain jitter (VAE re-encode proxy)
            gain = rng.uniform(1.0 - gain_jitter, 1.0 + gain_jitter,
                               (16, 1, 1, 1)).astype(np.float32)
            zi = zi * gain
        z_in[b] = zi
    return z_in, target, meta


# ----------------------------- losses / eval -----------------------------

def highpass(z, k: int = 5):
    """z [B,C,T,H,W] -> z - spatial avg blur (run-4 input prior).

    diag_real_learnability (2026-06-12): on real D1 latents the per-chip
    demod interference is 1.045 raw vs 0.256 after this high-pass, while the
    carrier (iid up to keyed permutation) passes through intact -- a
    zero-parameter hf+demod oracle already reaches msg 0.974 at the strong
    operating point.  Feeding hf(z) removes ~96% of the content nuisance
    energy and makes the oracle the network's starting point, not its target.
    """
    import torch.nn.functional as F
    p = k // 2
    blur = F.avg_pool3d(F.pad(z, (p, p, p, p, 0, 0), mode="replicate"),
                        kernel_size=(1, k, k), stride=1)
    return z - blur


def demod_loss(c_hat, meta, bank, torch, margin: float = 1.0):
    """Hinge on sign agreement of differentiable demod soft chips.

    Run-6: SNR-scaled margin.  Post-highpass per-chip content interference is
    ~0.256 (diag 2026-06-12); at the weak end (g*alpha=0.05, SNR~0.2) the
    per-chip accuracy ceiling is ~Phi(0.2)~0.58, i.e. ~40% of weak chips are
    INFORMATION-THEORETICALLY wrong -- a fixed margin=1 hinge on them never
    saturates and its never-extinguishing gradient is what dragged runs 4/5
    off the learned solution into the zero basin.  Scale the demanded margin
    by the achievable amplitude relative to the proven-achievable strong
    point (g*alpha=0.14, where warmup reached chip 0.91 with margin 1).
    """
    losses = []
    for b, m in enumerate(meta):
        if m["clean"]:
            continue
        block_t, g_t = bank.bufs(m["class_id"])
        _, car = bank.get(m["class_id"])
        soft = car.demodulate_torch(c_hat[b:b + 1], block_t, g_t)[0]  # [n]
        sgn = torch.from_numpy(
            (np.asarray(m["codeword"], np.float32) * 2 - 1)).to(soft.device)
        # normalize scale: per-block unit carrier, true |soft| = g*alpha.
        # run-5: curriculum min g*alpha raised to 0.05 == this clamp floor,
        # so the clamp no longer distorts the target.  Pure safety net.
        ref = max(m["g"] * m["alpha"], 0.05)
        marg = min(margin, ref / 0.14)  # run-6: weak samples demand less
        losses.append(torch.relu(marg - soft * sgn / ref).clamp_max(20.0).mean())
    if not losses:
        return torch.zeros((), device=c_hat.device)
    return torch.stack(losses).mean()


def eval_metrics(c_hat_np: np.ndarray, meta, bank) -> dict:
    accs, msg_accs, scores_wm, scores_clean = [], [], [], []
    for b, m in enumerate(meta):
        key, car = bank.get(m["class_id"])
        soft = car.demodulate(c_hat_np[b])
        dec = prc_soft_decode(soft, key, DEFAULT_N)
        if m["clean"]:
            scores_clean.append(dec["score"])
        else:
            hard = (soft > 0).astype(np.uint8)
            accs.append(float((hard == m["codeword"]).mean()))
            msg_accs.append(msg_bit_acc(dec, m["message"]))
            scores_wm.append(dec["score"])
    f = lambda v: round(float(np.mean(v)), 4) if v else None  # noqa: E731
    return {"chip_acc": f(accs), "msg_bit_acc": f(msg_accs),
            "score_wm": f(scores_wm), "score_clean": f(scores_clean)}


# ----------------------------- training -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="M2: train E_theta (B2)")
    ap.add_argument("--d1_dir", default=str(
        Path("./results/d1_latents")))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_demod", type=float, default=1.0,
                    help="run-4: demod hinge is the MAIN task loss (the only "
                         "loss aligned with the recoverable keyed projection)")
    ap.add_argument("--lambda_null", type=float, default=0.0,
                    help="run-6: DELETED by default (was 0.5).  L_null only "
                         "dug the zero-output basin deeper; zero output "
                         "scored s_clean~0.8 anyway, so it bought no real "
                         "clean/wm discrimination")
    ap.add_argument("--demod_warmup", type=int, default=0)
    ap.add_argument("--null_warmup", type=int, default=2000,
                    help="lambda_null=0 before this step (run-2 lesson: "
                         "L_null actively pushes output to zero, helping the "
                         "net settle into the zero plateau)")
    ap.add_argument("--curr_warm_end", type=int, default=4000,
                    help="curriculum: strong/narrow signal, no aug until here")
    ap.add_argument("--curr_anneal_end", type=int, default=16000,
                    help="curriculum: full range/aug reached at this step "
                         "(run-5: stretched from 10000)")
    ap.add_argument("--replay_frac", type=float, default=0.33,
                    help="run-5: fraction of post-warmup steps drawn from the "
                         "strong/no-aug regime (anti-forgetting replay)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="overfit 8 smooth synthetic samples, 4000 steps")
    ap.add_argument("--resume", default=None, help="checkpoint path")
    ap.add_argument("--resume_model_only", action="store_true",
                    help="run-6: load model weights only (fresh optimizer + "
                         "step counter) -- use when fine-tuning best.pt with "
                         "a new lr / new parameters")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / ("smoke_metrics.jsonl" if args.smoke else "metrics.jsonl")

    # data pool
    if args.smoke:
        pool = make_synthetic_pool(8, rng)
        # Round-4 smoke config: smooth content (see make_synthetic_pool),
        # bigger batch + higher lr (37M model, tiny data, normalized loss),
        # demod active from step 1 (it is the loss that directly aligns the
        # output with the keyed carrier subspace), no gain jitter.
        args.steps = min(args.steps, 4000)
        args.demod_warmup = 0
        args.batch = max(args.batch, 8)
        args.lr = max(args.lr, 1e-3)
        args.lambda_demod = max(args.lambda_demod, 0.5)
        log(f"SMOKE: 8 smooth synthetic latents, overfit, steps={args.steps}, "
            f"batch={args.batch}, lr={args.lr}, lambda_demod={args.lambda_demod}")
    else:
        pool = load_d1(Path(args.d1_dir), set(TRAIN_CLASSES))
        n_samp = sum(len(v) for v in pool.values())
        log(f"D1 loaded: {len(pool)} classes, {n_samp} samples "
            f"(held-out {HELDOUT_CLASSES} excluded)")
        if n_samp < 16:
            log("ERROR: too few D1 samples; run gen_d1_clean_latents.py first "
                "or use --smoke")
            return 1

    bank = CarrierBank(dev)
    from skcw.extractor.e_theta_model import ETheta, count_params
    model = ETheta().to(dev)
    # run-6 residual parameterization: C_hat = s*hf(z) + r_net(hf(z)).
    # s=1, r_net=0 IS the hf+demod oracle (strong msg 0.974) -- the
    # zero-output basin is no longer in the hypothesis space's attractor set:
    # to output zero the net must actively learn r_net = -s*hf(z), which the
    # demod hinge gradient opposes from step 1.
    s_param = torch.nn.Parameter(torch.tensor(1.0, device=dev))
    log(f"E_theta params: {count_params(model) / 1e6:.1f}M (+ residual "
        f"scalar s), device={dev}")
    opt = torch.optim.AdamW(list(model.parameters()) + [s_param],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    # AMP: fp16 autocast NaN'd in full training (M3 run-1, step ~1920, right
    # after demod activated) -- same underflow class as the smoke round-2 NaN:
    # this task regresses to ~3e-3 carrier amplitudes, below fp16 comfort.
    # Use bf16 (fp32 exponent range, native on A100), which needs NO GradScaler.
    use_amp = (dev != "cpu") and not args.smoke
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step0 = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "s_param" in ck:
            with torch.no_grad():
                s_param.copy_(torch.as_tensor(float(ck["s_param"]),
                                              device=dev))
        if args.resume_model_only:
            # run-6 warm start: best.pt (run-5 warmup, probe chip 0.910) has
            # no s_param and a stale Adam state; keep weights, restart opt at
            # the new fine-tune lr (cosine schedule over full --steps).
            log(f"warm start (model only) from {args.resume} "
                f"(ckpt step {ck.get('step')}, "
                f"probe_chip {ck.get('probe_chip')})")
        else:
            opt.load_state_dict(ck["opt"])
            step0 = ck["step"]
            log(f"resumed from {args.resume} @ step {step0}")

    append_jsonl(metrics_path, {"event": "start", "args": vars(args),
                                "n_chips": DEFAULT_N,
                                "ts": time.strftime("%Y-%m-%d %H:%M:%S")})

    t_last = time.time()
    best_probe = -1.0
    # smoke: fixed strong/narrow injection range. full training: curriculum()
    # schedules range + augmentation per step (run-3 fix, see curriculum doc).
    # run-5: after warmup, replay_frac of steps stay in the strong/no-aug
    # regime (anti catastrophic forgetting -- run-4 lost the warmup solution
    # entirely once the distribution shifted under it).
    for step in range(step0 + 1, args.steps + 1):
        if args.smoke:
            a_rng, g_rng, nz, jit = (0.3, 0.5), (0.3, 0.45), 0.0, 0.0
        else:
            a_rng, g_rng, nz, jit = curriculum(
                step, args.curr_warm_end, args.curr_anneal_end)
            if step > args.curr_warm_end and rng.random() < args.replay_frac:
                a_rng, g_rng, nz, jit = (0.3, 0.5), (0.3, 0.45), 0.0, 0.0
        z_in, target, meta = sample_batch(
            pool, bank, args.batch, rng,
            alpha_rng=a_rng, g_rng=g_rng,
            noise_std=nz, gain_jitter=jit)
        zt = torch.from_numpy(z_in).to(dev)
        tt = torch.from_numpy(target).to(dev)
        clean_mask = torch.tensor([m["clean"] for m in meta], device=dev)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            # run-4: high-pass input prior -- strips ~96% of content
            # interference (see highpass() doc) before the net sees it.
            # run-6: residual around the oracle -- C_hat = s*hf + r_net(hf).
            hf = highpass(zt)
            c_hat = s_param * hf + model(hf)
        c_hat = c_hat.float()
        wm = ~clean_mask
        # run-4 loss redesign.  runs 2/3 showed per-element (normalized) L1
        # regression makes "output zero" the GLOBAL optimum on real latents:
        # content hf texture is per-element indistinguishable from the keyed
        # carrier, so E[carrier | z_obs] ~= 0 pointwise.  The recoverable
        # quantity is the PROJECTION onto the keyed subspace, not the
        # per-element field.  So: scale-invariant cosine alignment (direction
        # only; zero output gets no reward) + demod hinge as the MAIN loss.
        if int(wm.sum()):
            ch = c_hat[wm].flatten(1)
            th = tt[wm].flatten(1)
            cos = (ch * th).sum(1) / (
                ch.norm(dim=1).clamp_min(1e-6) * th.norm(dim=1).clamp_min(1e-6))
            l_carrier = (1.0 - cos).mean()
        else:
            l_carrier = torch.zeros((), device=dev)
        # null loss (run-4): with a scale-free carrier loss the output scale
        # is set by the demod hinge, so penalize clean-output energy RELATIVE
        # to the watermarked-output energy (detached) -- "leak less on clean
        # than you emit on watermarked", scale-invariant by construction.
        lam_n = (args.lambda_null
                 if (args.smoke or step > args.null_warmup) else 0.0)
        if int(clean_mask.sum()) and int(wm.sum()):
            ref_mag = torch.abs(c_hat[wm]).mean().detach().clamp_min(1e-8)
            l_null = torch.abs(c_hat[clean_mask]).mean() / ref_mag
        else:
            l_null = torch.zeros((), device=dev)
        lam_d = args.lambda_demod if step > args.demod_warmup else 0.0
        l_demod = (demod_loss(c_hat, meta, bank, torch)
                   if lam_d > 0 else torch.zeros((), device=dev))
        loss = l_carrier + lam_d * l_demod + lam_n * l_null

        if not torch.isfinite(loss):
            # never let a bad step poison the weights
            log(f"step {step}: non-finite loss, skipping "
                f"(carrier={float(l_carrier.detach())}, "
                f"demod={float(l_demod.detach())}, null={float(l_null.detach())})")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [s_param], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                m = eval_metrics(c_hat.detach().cpu().numpy(), meta, bank)
                # run-5: fixed strong-regime probe (always the deployable
                # operating point, regardless of curriculum phase) -- this is
                # the health signal that exposed run-4's collapse only after
                # the fact.  best.pt tracks the best probe chip_acc.
                probe_chip = None
                if not args.smoke:
                    pz, pt, pmeta = sample_batch(
                        pool, bank, args.batch, rng, p_clean=0.25,
                        alpha_rng=(0.3, 0.5), g_rng=(0.3, 0.45),
                        noise_std=0.0, gain_jitter=0.0)
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16, enabled=use_amp):
                        phf = highpass(torch.from_numpy(pz).to(dev))
                        pc = s_param * phf + model(phf)
                    pm = eval_metrics(pc.float().cpu().numpy(), pmeta, bank)
                    probe_chip = pm["chip_acc"]
                    if probe_chip is not None and probe_chip > best_probe:
                        best_probe = probe_chip
                        torch.save({"model": model.state_dict(),
                                    "s_param": float(s_param.detach()),
                                    "opt": opt.state_dict(), "step": step,
                                    "probe_chip": probe_chip,
                                    "args": vars(args)}, out_dir / "best.pt")
            rec = {"step": step, "loss": round(float(loss.detach()), 5),
                   "l_carrier": round(float(l_carrier.detach()), 5),
                   "l_demod": round(float(l_demod.detach()), 5),
                   "l_null": round(float(l_null.detach()), 5),
                   "s_param": round(float(s_param.detach()), 4),
                   "probe_chip": probe_chip,
                   "lr": round(sched.get_last_lr()[0], 7), **m,
                   "sec": round(time.time() - t_last, 1),
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            t_last = time.time()
            append_jsonl(metrics_path, rec)
            log(f"step {step}: loss={rec['loss']} chip={m['chip_acc']} "
                f"msg={m['msg_bit_acc']} s_wm={m['score_wm']} "
                f"s_clean={m['score_clean']} probe={probe_chip} "
                f"best={best_probe:.3f} s={rec['s_param']}")

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "s_param": float(s_param.detach()),
                        "opt": opt.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / ("smoke_last.pt" if args.smoke else "last.pt"))

    # smoke verdict
    if args.smoke:
        with torch.no_grad():
            z_in, target, meta = sample_batch(pool, bank, 8, rng,
                                              p_clean=0.5, alpha_rng=(0.3, 0.5),
                                              g_rng=(0.3, 0.45), noise_std=0.0,
                                              gain_jitter=0.0)
            hf_s = highpass(torch.from_numpy(z_in).to(dev))
            c_hat = (s_param * hf_s + model(hf_s)).float().cpu().numpy()
        m = eval_metrics(c_hat, meta, bank)
        ok = (m["chip_acc"] is not None and m["chip_acc"] >= 0.95
              and (m["score_clean"] is None or m["score_clean"] < 1.5))
        verdict = {"event": "smoke_verdict", "pass": bool(ok), **m,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append_jsonl(metrics_path, verdict)
        log(f"SMOKE VERDICT: {'PASS' if ok else 'FAIL'} {m}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
