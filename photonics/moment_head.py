"""Learned local-statistic coefficients and fixed Gaussian feature transport.

The inference head contains neural maps and moment normalization only.
Derivative labels are constructed separately by the offline training script.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MomentMessage(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        x = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-x.square() / (2 * 1.5**2))
        g /= g.sum()
        self.register_buffer("kernel", (g[:, None] * g[None, :])[None, None])
        self.net = nn.Sequential(
            nn.Linear(10, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        self.register_buffer("feature_mean", torch.zeros(10))
        self.register_buffer("feature_scale", torch.ones(10))
        self.register_buffer("beta_mean", torch.zeros(3))
        self.register_buffer("beta_scale", torch.ones(3))

    def moments(self, x, t):
        shape = x.shape
        x = x.reshape(-1, 1, *shape[-2:])
        t = t.reshape_as(x)
        u, v, aa, bb, ab = F.conv2d(
            torch.cat((x, t, x * x, t * t, x * t)), self.kernel
        ).chunk(5)
        vx = aa - u * u
        vy = bb - v * v
        cov = ab - u * v
        c = (u * u + v * v + 1e-4).clamp_min(1e-8)
        d = (vx + vy + 9e-4).clamp_min(1e-8)
        scale0 = c.rsqrt() + (u.abs() + v.abs()) / d
        features = torch.cat(
            (
                u * c.rsqrt(),
                v * c.rsqrt(),
                vx / d,
                vy / d,
                cov / d,
                1e-4 / c,
                9e-4 / d,
                u / d.sqrt(),
                v / d.sqrt(),
                (u - v) * c.rsqrt(),
            ),
            1,
        )
        return x, t, (u, v, vx, vy, cov, c, d, scale0), features, shape

    def predict_beta(self, features):
        return (
            self.net((features - self.feature_mean) / self.feature_scale)
            * self.beta_scale
            + self.beta_mean
        )

    def coefficients(self, features, c, d, scale0):
        beta = self.predict_beta(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return torch.cat((beta[:, :1] * scale0, beta[:, 1:] / d), 1)

    def transport(self, coefficients, x, t, shape):
        batch, _, h, w = coefficients.shape
        # The symmetric Gaussian permits full convolution instead of a backend-specific deconvolution.
        transported = F.conv2d(
            coefficients.reshape(batch * 3, 1, h, w),
            self.kernel,
            padding=self.kernel.shape[-1] - 1,
        ).reshape(batch, 3, *shape[-2:])
        message = transported[:, :1] + x * transported[:, 1:2] + t * transported[:, 2:3]
        return message.reshape(shape)

    def forward(self, x, t):
        x, t, values, features, shape = self.moments(x, t)
        *_, c, d, scale0 = values
        return self.transport(self.coefficients(features, c, d, scale0), x, t, shape)
