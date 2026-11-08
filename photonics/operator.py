"""Learned global Fresnel feature processing, with a fixed inference graph.

The hidden fields are unconstrained learned features, not a sequence of
phase-only DOE estimates. There are no projection rules, gradients, or
optimization steps in this forward graph.
"""

import copy
import math

import torch
from torch import nn

from .model import DeepUNet, NeuralDOE
from .physics import Fresnel, OpticalConfig, fft2c, ifft2c


class SharedObjectMessages(nn.Module):
    """Shared learned bilinear features; no fixed amplitude projection rule."""

    def __init__(self, channels, width, context_width=0):
        super().__init__()
        self.channels, self.width = channels, width
        self.left = nn.Conv2d(6 * channels + 5, width, 1)
        self.right = nn.Conv2d(6 * channels + 5, width, 1)
        self.mix = nn.Sequential(
            nn.Conv2d(3 * width, 64, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 2 * channels, 1),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        self.context = None
        if context_width:
            self.enable_context(context_width)

    def enable_context(self, width=32):
        if self.context is not None:
            return
        self.context = nn.Sequential(
            nn.Conv2d(3 * self.width, width, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(width, 2 * self.channels, 1),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        self.context.to(device=self.left.weight.device, dtype=self.left.weight.dtype)

    def forward(self, fields, targets, coordinates):
        b, planes, c, n, _ = fields.shape
        amplitude = fields.abs()
        unit = fields / amplitude.clamp_min(1e-5)
        normalized = amplitude / amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-5)
        target_rms = (
            targets
            / targets.square().mean((-2, -1), keepdim=True).clamp_min(1e-8).sqrt()
        )
        depths = torch.linspace(
            -1, 1, planes, device=targets.device, dtype=targets.dtype
        )[None, :, None, None, None].expand(b, -1, -1, n, n)
        features = torch.cat(
            (
                fields.real,
                fields.imag,
                amplitude,
                unit.real,
                unit.imag,
                normalized,
                targets[:, :, None],
                target_rms[:, :, None],
                coordinates[:, None].expand(-1, planes, -1, -1, -1),
                depths,
            ),
            2,
        ).flatten(0, 1)
        left, right = self.left(features), self.right(features)
        gated = torch.cat((left, right, left * right), 1)
        output = self.mix(gated)
        if self.context is not None:
            output = output + self.context(gated)
        real, imag = output.chunk(2, 1)
        return torch.complex(
            real.reshape(b, planes, c, n, n), imag.reshape(b, planes, c, n, n)
        )


class FresnelFeatureBlock(nn.Module):
    def __init__(self, channels=4, polar=False, object_width=0, shared_width=0):
        super().__init__()
        self.channels = channels
        self.polar = polar
        self.shared_context = (
            SharedObjectMessages(channels, shared_width) if shared_width else None
        )
        self.object_context = (
            DeepUNet((20 if polar else 8) * channels + 6, 8 * channels, object_width)
            if object_width
            else None
        )
        # Complex features in four depth planes, plus target amplitudes
        # and coordinates. Spatial nonlinearities couple all Fourier modes.
        self.object_mix = nn.Sequential(
            nn.Conv2d((20 if polar else 8) * channels + 6, 48, 1),
            nn.GroupNorm(4, 48),
            nn.SiLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 8 * channels, 1),
        )
        self.aperture_mix = nn.Sequential(
            nn.Conv2d(4 * channels + 2, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 2 * channels, 3, padding=1),
        )
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, targets, coordinates, chirp):
        b, c, n, _ = z.shape
        fields = fft2c(z[:, None] * chirp[None, :, None])
        stacked = torch.cat(
            (
                fields.real.flatten(1, 2),
                fields.imag.flatten(1, 2),
                targets,
                coordinates,
            ),
            1,
        )
        if self.polar:
            amplitude = fields.abs()
            unit = fields / amplitude.clamp_min(1e-5)
            stacked = torch.cat(
                (
                    stacked,
                    amplitude.flatten(1, 2),
                    unit.real.flatten(1, 2),
                    unit.imag.flatten(1, 2),
                ),
                1,
            )
        mixed = self.object_mix(stacked)
        if self.object_context is not None:
            mixed = mixed + self.object_context(stacked)
        real, imag = mixed.chunk(2, 1)
        messages = torch.complex(
            real.reshape(b, 4, c, n, n), imag.reshape(b, 4, c, n, n)
        )
        if self.shared_context is not None:
            messages = messages + self.shared_context(fields, targets, coordinates)
        aperture = (ifft2c(messages) * chirp.conj()[None, :, None]).mean(1)
        features = torch.cat(
            (z.real, z.imag, aperture.real, aperture.imag, coordinates), 1
        )
        real, imag = self.aperture_mix(features).chunk(2, 1)
        return z + self.gain * torch.complex(real, imag)


class FresnelOperatorDOE(nn.Module):
    def __init__(
        self,
        width=16,
        seed=1729,
        config=None,
        channels=4,
        blocks=3,
        feature_kind="cartesian",
        global_width=0,
        object_width=0,
        shared_width=0,
    ):
        super().__init__()
        config = config or OpticalConfig()
        self.width, self.seed = width, seed
        self.architecture, self.carrier_init = "operator", "zero"
        self.channels, self.block_count = channels, blocks
        self.feature_kind = feature_kind
        self.global_width = global_width
        self.object_width = object_width
        self.shared_width = shared_width
        if feature_kind not in {"cartesian", "polar"}:
            raise ValueError("Unknown complex feature representation")
        self.physics = Fresnel(config)
        self.base = NeuralDOE(
            width=width,
            seed=seed,
            config=config,
            architecture="deep",
            carrier_init="zero",
        )
        self.base.requires_grad_(False)
        self.input_net = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2 * channels, 3, padding=1),
        )
        self.blocks = nn.ModuleList(
            [
                FresnelFeatureBlock(
                    channels, feature_kind == "polar", object_width, shared_width
                )
                for _ in range(blocks)
            ]
        )
        self.output_net = nn.Sequential(
            nn.Conv2d(2 * channels + 4, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)
        self.global_net = (
            DeepUNet(2 * channels + 4, 1, global_width) if global_width else None
        )

    def enable_global_context(self, width=8):
        """Add a zero-output global residual without changing the warm start."""
        if self.global_net is None:
            self.global_width = width
            self.global_net = DeepUNet(2 * self.channels + 4, 1, width)
            self.global_net.to(
                device=self.output_net[0].weight.device,
                dtype=self.output_net[0].weight.dtype,
            )

    def enable_object_context(self, width=16):
        """Learn multiscale complex messages in the four object planes."""
        if self.object_width:
            return
        if self.feature_kind != "polar":
            raise ValueError("Enable polar features before object context")
        self.object_width = width
        for block in self.blocks:
            block.object_context = DeepUNet(
                20 * self.channels + 6, 8 * self.channels, width
            )
            block.object_context.to(
                device=self.output_net[0].weight.device,
                dtype=self.output_net[0].weight.dtype,
            )

    def enable_shared_context(self, width=32):
        if self.shared_width:
            return
        self.shared_width = width
        for block in self.blocks:
            block.shared_context = SharedObjectMessages(self.channels, width)
            block.shared_context.to(
                device=self.output_net[0].weight.device,
                dtype=self.output_net[0].weight.dtype,
            )

    def enable_polar_features(self):
        """Add exact optical features while preserving the current function."""
        if self.feature_kind == "polar":
            return
        for block in self.blocks:
            old = block.object_mix[0]
            widened = nn.Conv2d(
                old.in_channels + 12 * self.channels, old.out_channels, 1
            )
            widened = widened.to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                widened.weight.zero_()
                widened.weight[:, : old.in_channels].copy_(old.weight)
                widened.bias.copy_(old.bias)
            block.object_mix[0] = widened
            block.polar = True
        self.feature_kind = "polar"

    def extend_depth(self, blocks):
        """Warm-start independent residual layers without changing predictions."""
        if blocks < self.block_count:
            raise ValueError("Depth expansion cannot remove trained blocks")
        while len(self.blocks) < blocks:
            layer = copy.deepcopy(self.blocks[-1])
            with torch.no_grad():
                layer.gain.zero_()
            self.blocks.append(layer)
        self.block_count = blocks

    def forward(self, targets):
        # The base is frozen by default. The training script can explicitly
        # unfreeze it for a final joint training stage; inference is unchanged.
        base_phase = self.base(targets)
        carrier = self.base.carrier
        field = torch.complex(targets * carrier.cos(), targets * carrier.sin())
        back = self.physics.backproject_prechirp(field)
        back = (
            back
            / back.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-8).sqrt()
        )
        xy = self.base.coordinates.expand(targets.shape[0], -1, -1, -1)
        features = torch.cat(
            (back.real, back.imag, base_phase.cos(), base_phase.sin(), xy), 1
        )
        real, imag = self.input_net(features).chunk(2, 1)
        z = 0.1 * torch.complex(real, imag)
        base_wave = torch.complex(base_phase.cos(), base_phase.sin())
        source = torch.cat((base_wave, torch.zeros_like(z[:, 1:])), 1)
        z = z + source
        for layer in self.blocks:
            z = layer(z, targets, xy, self.physics.chirp)
        features = torch.cat(
            (z.real, z.imag, base_phase.cos(), base_phase.sin(), xy), 1
        )
        logits = self.output_net(features)
        if self.global_net is not None:
            logits = logits + self.global_net(features)
        correction = math.pi * torch.tanh(logits)
        phase = base_phase + correction
        return torch.atan2(phase.sin(), phase.cos())
