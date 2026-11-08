"""A learned residual network using fields of the actual frozen DOE predictor.

There is one learned message calculation, physical backpropagation, and a CNN
phase output. No amplitude substitution, phase optimizer, or synthesis loop.
"""

import math

import torch
from torch import nn

from .operator import FresnelOperatorDOE, SharedObjectMessages
from .physics import Fresnel, OpticalConfig, fft2c


class LearnedRefinerDOE(nn.Module):
    def __init__(
        self,
        base_spec,
        config=None,
        message_width=32,
        context_width=0,
        linear_projection=False,
    ):
        super().__init__()
        config = config or OpticalConfig()
        if base_spec["architecture"] != "operator":
            raise ValueError("Expected a frozen operator base")
        self.base_spec = dict(base_spec)
        self.width, self.seed = base_spec["width"], base_spec["seed"]
        self.architecture = "refiner"
        self.global_width = self.object_width = self.shared_width = 0
        self.block_count = 1
        self.message_width = message_width
        self.context_width = context_width
        self.physics = Fresnel(config)
        self.base = FresnelOperatorDOE(
            width=self.width,
            seed=self.seed,
            config=config,
            channels=base_spec.get("channels", 4),
            blocks=base_spec.get("blocks", 3),
            feature_kind=base_spec.get("feature_kind", "polar"),
            global_width=base_spec.get("global_width", 0),
            object_width=base_spec.get("object_width", 0),
            shared_width=base_spec.get("shared_width", 0),
        )
        self.base.requires_grad_(False)
        self.message = SharedObjectMessages(1, message_width, context_width)
        self.projection = nn.Conv2d(16, 1, 1, bias=False) if linear_projection else None
        if self.projection is not None:
            nn.init.zeros_(self.projection.weight)
        # Zero messages produce exactly zero correction. Nonzero random linear
        # weights preserve a useful first-step gradient into the message CNN.
        self.aperture = nn.Sequential(
            nn.Conv2d(16, 32, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1, bias=False),
        )

    def specification(self):
        return {
            "architecture": "refiner",
            "width": self.width,
            "seed": self.seed,
            "base": self.base_spec,
            "message_width": self.message_width,
            "context_width": self.context_width,
            "linear_projection": self.projection is not None,
        }

    def enable_message_context(self, width=32):
        self.message.enable_context(width)
        self.context_width = width

    def enable_linear_projection(self):
        if self.projection is None:
            self.projection = nn.Conv2d(16, 1, 1, bias=False).to(
                device=self.aperture[0].weight.device,
                dtype=self.aperture[0].weight.dtype,
            )
            nn.init.zeros_(self.projection.weight)

    def correction(self, targets, phase):
        wave = torch.complex(phase.cos(), phase.sin())
        fields = fft2c(wave * self.physics.chirp)[:, :, None]
        coordinates = self.base.base.coordinates.expand(targets.shape[0], -1, -1, -1)
        messages = self.message(fields, targets, coordinates).squeeze(2)
        back = self.physics.backproject_prechirp(messages)
        relative = back * wave.conj()
        features = torch.cat((back.real, back.imag, relative.real, relative.imag), 1)
        logits = self.aperture(features)
        if self.projection is not None:
            logits = logits + self.projection(features)
        correction = math.pi * torch.tanh(logits)
        return correction

    def forward(self, targets):
        phase = self.base(targets)
        result = phase + self.correction(targets, phase)
        return torch.atan2(result.sin(), result.cos())
