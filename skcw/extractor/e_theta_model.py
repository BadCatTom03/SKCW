"""E_theta (B2): direct carrier-estimation network in Wan VAE latent space.

Input : Z_obs  [B,16,T,H,W]   normalized latent of video under test
Output: C_hat  [B,16,T,H,W]   estimate of the attenuated carrier  g*alpha*C
        (NO residual add, NO blueprint estimation -- B2 mainline, plan v2 Sec.6)

3D UNet, ~45M params, 2x spatial bottleneck (stride only on H,W; T=5 too short
to downsample).  GroupNorm + SiLU.  Head takes [features, z_obs] and is
small-init (NOT zero-init -- zero-init killed all backbone gradients and froze
the net at C_hat=0; see head comment).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            gn(ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            gn(ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Down(nn.Module):
    """2x spatial downsample (T untouched)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """2x spatial upsample (T untouched)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)

    def forward(self, x, size_hw):
        x = nn.functional.interpolate(
            x, size=(x.shape[2], *size_hw), mode="nearest")
        return self.conv(x)


class ETheta(nn.Module):
    """B2 carrier estimator.  base=96 -> ~37M params."""

    def __init__(self, in_ch: int = 16, base: int = 96, depth_blocks: int = 2):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4          # 64,128,256
        self.inp = nn.Conv3d(in_ch, c1, 3, padding=1)
        self.enc1 = nn.Sequential(*[ResBlock3D(c1) for _ in range(depth_blocks)])
        self.down1 = Down(c1, c2)
        self.enc2 = nn.Sequential(*[ResBlock3D(c2) for _ in range(depth_blocks)])
        self.down2 = Down(c2, c3)
        self.mid = nn.Sequential(*[ResBlock3D(c3) for _ in range(depth_blocks + 1)])
        self.up2 = Up(c3, c2)
        self.dec2 = nn.Sequential(*[ResBlock3D(c2) for _ in range(depth_blocks)])
        self.up1 = Up(c2, c1)
        self.dec1 = nn.Sequential(*[ResBlock3D(c1) for _ in range(depth_blocks)])
        self.skip2 = nn.Conv3d(c2 * 2, c2, 1)
        self.skip1 = nn.Conv3d(c1 * 2, c1, 1)
        # head sees [features, z_obs]: the z_obs passthrough gives a direct
        # linear input->output path whose demod-loss gradient is COHERENT
        # (sum_pos delta*z_in = g*a*||delta||^2 > 0), guaranteeing escape
        # from the zero-output basin.
        self.head = nn.Sequential(gn(c1 + in_ch), nn.SiLU(),
                                  nn.Conv3d(c1 + in_ch, in_ch, 3, padding=1))
        # SMALL-init head (NOT zero): zero-init blocked ALL backbone grads
        # (dL/dbackbone = dL/dout * W_head = 0) and the head-only gradient
        # against a per-position pseudorandom target sums to ~0 -> the net
        # was permanently stuck at C_hat=0 (smoke rounds 3-4 loss plateau).
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z_obs: torch.Tensor) -> torch.Tensor:
        h1 = self.enc1(self.inp(z_obs))                  # [B,c1,T,H,W]
        h2 = self.enc2(self.down1(h1))                   # H/2
        h3 = self.mid(self.down2(h2))                    # H/4 bottleneck
        u2 = self.up2(h3, h2.shape[-2:])
        u2 = self.dec2(self.skip2(torch.cat([u2, h2], dim=1)))
        u1 = self.up1(u2, h1.shape[-2:])
        u1 = self.dec1(self.skip1(torch.cat([u1, h1], dim=1)))
        return self.head(torch.cat([u1, z_obs], dim=1))  # C_hat_w


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":  # shape smoke test (CPU)
    m = ETheta()
    print(f"params: {count_params(m) / 1e6:.1f}M")
    x = torch.randn(1, 16, 5, 60, 104)
    with torch.no_grad():
        y = m(x)
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}")
    assert y.shape == x.shape
    print(f"init |out| mean = {float(y.abs().mean()):.4g} (small but nonzero)")
