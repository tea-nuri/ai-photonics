"""Learned global-moment message; fixed statistics followed by a neural readout."""

import torch
from torch import nn


class GlobalMessage(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(5, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )
        self.register_buffer("feature_mean", torch.zeros(5))
        self.register_buffer("feature_scale", torch.ones(5))

    def features(self, a, t):
        ma = a.mean((-2, -1), keepdim=True)
        mt = t.mean((-2, -1), keepdim=True)
        sa = (a - ma).square().mean((-2, -1), keepdim=True).clamp_min(1e-10).sqrt()
        st = (t - mt).square().mean((-2, -1), keepdim=True).clamp_min(1e-10).sqrt()
        za = (a - ma) / sa
        zt = (t - mt) / st
        feat = torch.cat(
            (
                ma / sa,
                mt / st,
                sa / a.amax((-2, -1), keepdim=True).clamp_min(1e-8),
                st,
                (za * zt).mean((-2, -1), keepdim=True),
            ),
            -1,
        ).squeeze(-2)
        return feat, za, zt, sa

    def predict_coefficients(self, f):
        return self.net((f - self.feature_mean) / self.feature_scale)

    def forward(self, a, t):
        f, za, zt, sa = self.features(a, t)
        coef = self.predict_coefficients(f)
        return (coef[..., 0, None, None] * za + coef[..., 1, None, None] * zt) / sa
