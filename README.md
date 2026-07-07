# SKCW — Semantic-Keyed Content Watermarking for Video Diffusion Models

> **Inversion-free, blind, semantically keyed generative video watermarking.**
> SKCW embeds a pseudorandom-error-correcting (PRC) carrier into the VAE latent space of a video
> diffusion model and recovers it with a lightweight residual extractor network, without diffusion
> inversion at detection time.

**Authors:** Haoyu Yang, Ke Niu\*, Shizhe Ru, Qiang Huang  
**Affiliation:** Engineering University of PAP, Xi'an, China  
**Corresponding author:** Ke Niu (niuke@163.com)

---

## Highlights

- **Inversion-free extraction.** Detection uses one VAE encode and one forward pass of a compact
  3D-UNet residual extractor. It does not require iterative diffusion inversion, so extraction is
  structurally decoupled from inversion quality.
- **Blind and constant-cost.** Only semantic class keys are stored. No per-video metadata is needed,
  and extraction cost is independent of the number of generated videos.
- **Semantic-keyed routing.** Each payload is associated with a semantic class through a class-derived
  key. The resulting keyed carrier spaces support class-level traceability, key separation, and fault
  isolation.
- **PRC error correction.** A spread-spectrum PRC layer with CRC provides redundancy over raw carrier
  chips and improves payload recovery under difficult semantic categories and mild distortions.
- **Temporal realignment support.** For frame-level temporal perturbations, SKCW provides an
  inversion-free temporal realignment module based on per-frame carrier evidence, slice-restricted
  PRC scoring, and capacity-constrained assignment before standard decoding.

---

## Why inversion-free extraction matters

Many generative-video watermarking pipelines recover watermarks by inverting the generated video back
through the diffusion trajectory. SKCW instead learns to read the watermark residual that survives the
full generation and VAE round trip. In practice, this gives a simpler detection path:

| | Inversion-based detection | **SKCW** |
|---|---|---|
| Detection path | VAE encode + iterative diffusion inversion + decode | VAE encode + residual extractor + demodulate |
| Depends on inversion quality | Yes | **No** |
| Per-video metadata | Not required | **Not required** |
| Scaling with number of videos | O(1) key storage | **O(1) key storage** |

---

## Repository layout

```text
.
├── main.py                    # end-to-end demo entry point
├── requirements.txt           # Python dependency list
├── CODE_INVENTORY.md          # file-by-file release inventory
├── skcw/
│   ├── core/                  # model-agnostic watermark kernel
│   │   ├── carriers.py        # class-key derivation and carrier patterns
│   │   ├── carriers_prc.py    # PRC-chip carrier: PRCCarrier, make_delta, demodulate
│   │   ├── prc_code.py        # PRC encoding and soft decoding
│   │   ├── payload.py         # message + CRC payload encoding
│   │   ├── blueprint.py       # zero-content blueprint base
│   │   └── prompts32.py       # 32-class prompt set
│   ├── extractor/             # residual extractor network E_theta
│   │   ├── e_theta_model.py   # 3D-UNet extractor
│   │   └── train_e_theta.py   # extractor training logic
│   ├── hunyuan_1_0/           # HunyuanVideo-1.0 adapter and evaluation scripts
│   │   ├── hv10_runner.py
│   │   ├── gen_d1_hv10.py
│   │   ├── train_e_theta_hv10.py
│   │   ├── eval_hv10_semantic.py
│   │   └── blind_key_id.py
│   ├── wan22/                 # optional Wan2.2 / Bernini adapter
│   │   └── wan22_runner.py
│   ├── runner/                # auxiliary experiment runners
│   └── eval/                  # robustness, temporal realignment, and generalization evaluation
```

See [`CODE_INVENTORY.md`](./CODE_INVENTORY.md) for a file-by-file description.

---

## Method overview

1. **Embedding.** For semantic class `c`, derive a class key `K_c` and build a PRC carrier
   `PRCCarrier(K_c, shape)`. The payload is PRC-encoded into a codeword, and the carrier produces a
   low-amplitude latent residual `Delta`.
2. **Generation.** The watermarked latent `z_wm = z + alpha * Delta` is decoded through the video
   diffusion and VAE pipeline.
3. **Extraction.** The received video is VAE-encoded, high-pass filtered, and passed through the
   residual extractor `E_theta` to estimate carrier evidence.
4. **Demodulation and decoding.** The estimated carrier evidence is demodulated with the semantic key
   and decoded by the PRC layer to recover the message.
5. **Temporal realignment.** Under frame drop/insert/clip attacks, the optional temporal realignment
   module scores each observed frame against latent temporal slices and solves a capacity-constrained
   assignment before standard decoding.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For full HunyuanVideo or Wan2.2 experiments, install the official dependencies required by the
corresponding generation backbone. Model weights are not included in this repository.

### 2. End-to-end HV-1.0 demo

```bash
python main.py \
  --backend hv10 \
  --prompt "A cat walks on a wooden floor" \
  --class_id 0 \
  --ckpt ./checkpoints/last.pt
```

### 3. Generate clean latents for extractor training/evaluation

```bash
python -m skcw.hunyuan_1_0.gen_d1_hv10 --out d1_hv10 --n 256
```

### 4. Train the residual extractor

```bash
python -m skcw.hunyuan_1_0.train_e_theta_hv10 --data d1_hv10 --out ckpt_hv10
```

### 5. Run semantic-stratified clean detection

```bash
python -m skcw.hunyuan_1_0.eval_hv10_semantic \
  --ckpt ckpt_hv10/last.pt \
  --out semantic_summary.json
```

### 6. Run robustness evaluation

```bash
python -m skcw.eval.eval_attacks --ckpt ckpt_hv10/last.pt
python -m skcw.eval.eval_attacks_ablation_full --ckpt ckpt_hv10/last.pt
```

Paths and flags may vary by local environment; check each script's `argparse` block for exact options.

---

## Model weights and checkpoints

This repository does **not** redistribute third-party video-generation weights or trained SKCW
checkpoints. Please obtain backbone weights from their official sources and place them under a local
path such as:

```text
pretrained_models/
├── HunyuanVideo/
├── Wan2.2/
└── bernini/
```

Extractor checkpoints can be placed under:

```text
checkpoints/
└── last.pt
```

Both directories are ignored by Git.

---

## Key configuration

| Item | Default |
|---|---|
| PRC chips `N` | 512 |
| Message length | 32 bits + CRC16 |
| Carrier amplitude `alpha` | 0.4 |
| HV-1.0 latent shape | `(16, 17, 60, 60)` for 480 x 480 x 65f |
| Extractor | 3D-UNet, `in_ch=16`, `base=96` |
| Semantic classes | 32 classes |
| Train / held-out split | classes 0-27 train, 28-31 held out |

---

## Optional Wan2.2 / Bernini backend

The `skcw/wan22/` adapter is provided for experiments with Wan2.2 through Bernini. It requires a local
Bernini checkout and model weights. Treat this path as an optional backend adapter rather than the
minimal reproducibility path.

---

## Citation

```bibtex
@misc{skcw2026,
  title  = {SKCW: Semantic-Keyed Content Watermarking for Video Diffusion Models via Inversion-Free Learned Residual Extraction},
  author = {Haoyu Yang and Ke Niu and Shizhe Ru and Qiang Huang},
  year   = {2026},
  note   = {Submitted to IEEE TCSVT}
}
```

## License

Released under the MIT License. See [`LICENSE`](./LICENSE).
