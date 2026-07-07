# SKCW Open-Source Code Inventory

> The repository is organized as `skcw/{core,extractor,hunyuan_1_0,wan22,runner,eval}/`.
> The primary reproducibility path is the HunyuanVideo-1.0 backend; the Wan2.2/Bernini backend is an optional adapter requiring external code and weights.

## 0. End-to-end entry point

| File | Role |
|---|---|
| `main.py` | End-to-end demo: key derivation, carrier construction, latent embedding, generation, extraction, and decoding. Supports `--backend hv10|wan22`. |

## 1. Core watermark library — `skcw/core/`

| File | Role |
|---|---|
| `carriers.py` | Semantic class-key derivation and baseline carrier utilities. |
| `carriers_prc.py` | PRC-chip carrier implementation: keyed permutation, Gaussian carrier blocks, `PRCCarrier`, `make_delta`, and demodulation. |
| `prc_code.py` | PRC encoding and soft decoding for message recovery. |
| `payload.py` | Message + CRC16 payload encoding and verification. |
| `blueprint.py` | Zero-content blueprint utilities. |
| `prompts32.py` | 32-class prompt set and train/held-out class split. |

## 2. Residual extractor — `skcw/extractor/`

| File | Role |
|---|---|
| `e_theta_model.py` | 3D-UNet residual extractor `ETheta`. |
| `train_e_theta.py` | Extractor training logic, high-pass preprocessing, residual parameterization, curriculum, and checkpointing. |

## 3. HunyuanVideo-1.0 backend — `skcw/hunyuan_1_0/`

| File | Role |
|---|---|
| `hv10_runner.py` | HunyuanVideo-1.0 generation / VAE encode-decode wrapper. |
| `gen_d1_hv10.py` | Clean latent generation for extractor training and evaluation. |
| `train_e_theta_hv10.py` | HV-1.0-specific extractor training entry point. |
| `eval_hv10_semantic.py` | Semantic-stratified clean detection evaluation. |
| `blind_key_id.py` | Blind semantic key identification and key-separation evaluation. |

## 4. Optional Wan2.2 / Bernini backend — `skcw/wan22/`

| File | Role |
|---|---|
| `wan22_runner.py` | Bernini/Wan2.2 adapter. Requires a local Bernini checkout and external model weights. |

## 5. Experiment runners — `skcw/runner/`

| File | Role |
|---|---|
| `p0_channel_capacity.py` | Channel-capacity and backbone adapter experiments. |
| `p6_capacity_sweep.py` | Capacity sweep utility. Keep only if the released paper uses this experiment; otherwise remove before public release. |

## 6. Robustness and generalization evaluation — `skcw/eval/`

| File | Role |
|---|---|
| `eval_attacks.py` | Spatial robustness evaluation. |
| `eval_attacks_temporal.py` | Temporal attack evaluation. |
| `eval_attacks_temporal_sgo.py` | Temporal segmentation / grouping ablation. |
| `eval_attacks_temporal_swdet.py` | Inversion-free temporal realignment evaluation. |
| `eval_attacks_ablation_full.py` | Main robustness ablation runner. |
| `eval_heldout.py` | Held-out semantic class generalization. |
| `sgo_video_utils.py` | Frame alignment, optical-flow segmentation, and temporal utility functions. |
| `run_sigmark_hv_semantic_512bit.sh` | Optional baseline helper script. Uses environment variables for local paths. |

## 7. Paper source — `skcw/paper/`

Contains the LaTeX manuscript and figures. Generated LaTeX build artifacts are ignored by Git.

## Release notes

- Third-party generation weights are not redistributed.
- Local checkpoints, generated videos, JSONL logs, and experiment outputs are ignored by `.gitignore`.
- Before public release, verify that all scripts referenced in the README are runnable in a clean environment and that optional baseline scripts contain no machine-specific absolute paths.
