"""Fixed neural layers using learned local/global moment messages."""

import torch

from .dual_message import DualMessageDOE
from .global_message import GlobalMessage
from .moment_head import MomentMessage
from .physics import fft2c


class MomentMessageDOE(DualMessageDOE):
    def __init__(
        self,
        base_spec,
        config,
        depth=32,
        message_width=32,
        local_width=64,
        global_width=32,
    ):
        super().__init__(
            base_spec,
            config,
            depth,
            message_width,
            adaptive_readout=True,
            soft_peak=True,
        )
        self.moment_head = MomentMessage(local_width)
        self.global_head = GlobalMessage(global_width)
        self.architecture = "moment_message"

    def neural_maps(self, targets, phase):
        wave = torch.complex(phase.cos(), phase.sin())
        field = fft2c(wave * self.physics.chirp)
        amplitude = field.abs()
        ratio = amplitude / amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        messages = torch.stack(
            (self.moment_head(ratio, targets), self.global_head(amplitude, targets)), 1
        )
        return wave, field, amplitude, messages

    def specification(self):
        return {
            "architecture": self.architecture,
            "base": self.base_spec,
            "depth": self.depth,
            "message_width": self.message_width,
            "local_width": self.moment_head.width,
            "global_width": self.global_head.width,
            "width": self.width,
            "seed": self.seed,
        }
