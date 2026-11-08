"""A learned gate between two frozen neural phase-correction experts."""

import math

import torch
from torch import nn

from .physics import OpticalConfig
from .refiner import LearnedRefinerDOE


class GatedDOE(nn.Module):
    def __init__(self, expert_specs, config=None):
        super().__init__()
        config = config or OpticalConfig()
        if len(expert_specs) != 2 or expert_specs[0]["base"] != expert_specs[1]["base"]:
            raise ValueError("Two experts must use the same frozen base specification")
        self.expert_specs = expert_specs
        self.architecture = "gated"
        self.width = expert_specs[0]["width"]
        self.seed = expert_specs[0]["seed"]

        def make(spec):
            return LearnedRefinerDOE(
                spec["base"],
                config,
                spec.get("message_width", 32),
                spec.get("context_width", 0),
                spec.get("linear_projection", False),
            )

        self.first, self.second = make(expert_specs[0]), make(expert_specs[1])
        self.base = self.first.base
        self.second.base = self.base
        # Every branch uses identical fixed Fresnel buffers. Share them as well
        # as the base network so the checkpoint stores each constant once.
        self.physics = self.base.physics
        self.base.base.physics = self.physics
        self.first.physics = self.second.physics = self.physics
        self.first.requires_grad_(False)
        self.second.requires_grad_(False)
        self.gate = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(0.2 / 0.8))

    def specification(self):
        return {
            "architecture": "gated",
            "width": self.width,
            "seed": self.seed,
            "experts": self.expert_specs,
        }

    @staticmethod
    def features(targets):
        mean = targets.mean((-2, -1))
        power = targets.square().mean((-2, -1))
        area = (targets > 0.1).to(targets.dtype).mean((-2, -1))
        rough = (targets[:, :, 1:, :] - targets[:, :, :-1, :]).abs().mean((-2, -1)) + (
            targets[:, :, :, 1:] - targets[:, :, :, :-1]
        ).abs().mean((-2, -1))
        return torch.cat((mean, power, area, rough), 1)

    def forward(self, targets):
        phase = self.base(targets)
        first = self.first.correction(targets, phase)
        second = self.second.correction(targets, phase)
        alpha = self.gate(self.features(targets)).sigmoid()[:, :, None, None]
        result = phase + (1 - alpha) * first + alpha * second
        return torch.atan2(result.sin(), result.cos())
