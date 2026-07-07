#!/usr/bin/env python3
"""H2: train E_theta-HV for HunyuanVideo-1.0 (latent 16,17,60,60).

Thin shim over train_e_theta.py: the training logic, losses, curriculum,
run-6 residual parameterization (C_hat = s*hf + r_net(hf)) are model-agnostic
-- only the latent SHAPE and the IO directories differ.  ETheta is a 3D UNet
(stride on H,W only) so T=17 works unchanged; highpass is k=(1,5,5).

H0 showed raw highpass chip_acc=0.9863 on HV-1.0, so training should converge
fast.  Same gates as Wan: smoke pass = chip>=0.95 AND null<1.5.

Server:
  cd .
  # smoke gate first:
  nohup python train_e_theta_hv10.py --smoke \
      > m2_hv10_smoke.log 2>&1 < /dev/null &
  # full (after D1-HV ready):
  nohup python train_e_theta_hv10.py --steps 12000 \
      > m2_hv10.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import sys
from pathlib import Path


import skcw.extractor.train_e_theta as T  # noqa: E402

HV_LATENT_SHAPE = (16, 17, 60, 60)
HV_D1_DIR = "./results/d1_latents_hv10"
HV_OUT_DIR = "./results/e_theta_hv10"


def _inject_default(flag: str, value: str) -> None:
    """Set a default arg only if the user did not pass it."""
    if flag not in sys.argv:
        sys.argv += [flag, value]


def main() -> int:
    # patch the module-global shape used by CarrierBank / sample_batch /
    # load_d1 / make_synthetic_pool (all reference T.LATENT_SHAPE at runtime)
    T.LATENT_SHAPE = HV_LATENT_SHAPE
    _inject_default("--d1_dir", HV_D1_DIR)
    _inject_default("--out_dir", HV_OUT_DIR)
    T.log(f"[HV10 shim] LATENT_SHAPE={T.LATENT_SHAPE} d1={HV_D1_DIR}")
    return T.main()


if __name__ == "__main__":
    sys.exit(main())
