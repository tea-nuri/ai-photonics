"""Fixed neural layers with two independently normalized optical channels.

CNN feature maps are supervised offline. Inference evaluates only the neural
graph and physical statistics, without a loss function or phase optimization.
"""

import copy
import math

import torch
from torch import nn

from .cascade import EquivariantCorrection


class DualMessageDOE(nn.Module):
    def __init__(
        self,
        base_spec,
        config,
        depth=16,
        message_width=32,
        stage_coefficients=False,
        adaptive_readout=False,
        soft_peak=False,
    ):
        super().__init__()
        from .model import build_model

        self.base_spec = base_spec
        self.base = build_model(base_spec, config)
        self.base.requires_grad_(False)
        self.physics = self.base.physics
        self.depth, self.message_width = depth, message_width
        self.architecture = "dual_message"
        self.width, self.seed = base_spec["width"], base_spec["seed"]
        self.encoder = EquivariantCorrection(
            message_width,
            context=True,
            peak_feature=True,
            statistics=True,
            unit_messages=True,
            local_statistics=True,
            joint_fields=True,
        )
        self.head = nn.Conv2d(message_width, 2, 1)
        self.context_head = nn.Sequential(
            nn.Conv2d(message_width, message_width, 3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Conv2d(message_width, 2, 1),
        )
        for layer in [self.head, self.context_head[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.peak_scale = nn.Parameter(torch.ones(()))
        self.branch_weights = nn.Parameter(torch.ones(2))
        self.step_scale = nn.Parameter(torch.tensor(0.04))
        self.stage_coefficients = False
        self.controller = None
        self.register_parameter("log_peak_temperature", None)
        self.register_parameter("stage_branch_weights", None)
        self.register_parameter("stage_step_scales", None)
        self.register_parameter("stage_peak_scales", None)
        if stage_coefficients:
            self.enable_stage_coefficients()
        if adaptive_readout:
            self.enable_adaptive_readout()
        if soft_peak:
            self.enable_soft_peak()
        self.freeze_unused()

    def enable_soft_peak(self, temperature=0.01):
        """Use a learned attention width instead of a discontinuous bright pixel."""
        if self.log_peak_temperature is None:
            self.log_peak_temperature = nn.Parameter(
                self.step_scale.new_tensor(math.log(temperature))
            )

    def peak_attention(self, amplitude, maximum):
        if self.log_peak_temperature is not None:
            temperature = self.log_peak_temperature.clamp(-9.0, -1.0).exp()
            logits = (amplitude / maximum / temperature).flatten(2)
            return logits.softmax(-1).reshape_as(amplitude)
        peak = (amplitude == maximum).to(amplitude.dtype)
        return peak / peak.sum((-2, -1), keepdim=True).clamp_min(1)

    def enable_adaptive_readout(self):
        """Learn scene-dependent neural channel strengths from physical features."""
        if self.controller is not None:
            return
        self.controller = nn.Sequential(
            nn.Linear(24, 32), nn.SiLU(), nn.Linear(32, 3)
        ).to(self.step_scale.device)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def readout_factors(self, targets, amplitude, tangent, stage):
        rms = amplitude.square().mean((-2, -1)).clamp_min(1e-10).sqrt()
        peaks = amplitude.flatten(2).topk(2, dim=-1).values
        margin = 10 * (peaks[..., 0] - peaks[..., 1]) / peaks[..., 0].clamp_min(1e-8)
        tangent_rms = tangent.square().mean((-2, -1)).clamp_min(1e-10).sqrt()
        cosine = (tangent[:, 0] * tangent[:, 1]).mean((-2, -1)) / tangent_rms.prod(1)
        features = torch.cat(
            (
                targets.mean((-2, -1)),
                targets.std((-2, -1), correction=0),
                torch.log1p(peaks[..., 0] / rms),
                margin,
                amplitude.std((-2, -1), correction=0) / rms,
                torch.log1p(1e4 * tangent_rms),
                cosine[:, None],
                targets.new_full((len(targets), 1), stage / max(1, self.depth - 1)),
            ),
            1,
        )
        logits = self.controller(features)
        return torch.cat(
            (2 * logits[:, :2].sigmoid(), logits[:, 2:].clamp(-3.0, 3.0).exp()), 1
        )

    def enable_stage_coefficients(self):
        if self.stage_coefficients:
            return
        self.stage_branch_weights = nn.Parameter(
            self.branch_weights.detach().repeat(self.depth, 1)
        )
        self.stage_step_scales = nn.Parameter(
            self.step_scale.detach().repeat(self.depth)
        )
        self.stage_peak_scales = nn.Parameter(
            self.peak_scale.detach().repeat(self.depth)
        )
        for parameter in [self.branch_weights, self.step_scale, self.peak_scale]:
            parameter.requires_grad_(False)
        self.stage_coefficients = True

    def freeze_unused(self):
        for layer in [
            self.encoder.object_net[-1],
            self.encoder.context_conv,
            self.encoder.aperture,
            self.encoder.linear,
        ]:
            layer.requires_grad_(False)

    @classmethod
    def from_source(cls, source, depth=16):
        model = cls(source.specification(), source.physics.config, depth).to(
            next(source.parameters()).device
        )
        model.base.load_state_dict(source.state_dict())
        encoder = copy.deepcopy(source.correction[-1])
        encoder.global_aperture = None
        encoder.global_aperture_width = 0
        encoder.enable_joint_fields()
        encoder.requires_grad_(True)
        model.encoder = encoder
        model.freeze_unused()
        return model

    def specification(self):
        return {
            "architecture": self.architecture,
            "width": self.width,
            "seed": self.seed,
            "base": self.base_spec,
            "depth": self.depth,
            "message_width": self.message_width,
            "stage_coefficients": self.stage_coefficients,
            "adaptive_readout": self.controller is not None,
            "soft_peak": self.log_peak_temperature is not None,
        }

    def neural_maps(self, targets, phase):
        wave, field, amplitude, _, encoded = self.encoder.encode_fields(
            targets, phase, self.physics
        )
        b, _, n, _ = targets.shape
        maps = (
            (self.head(encoded) + self.context_head(encoded))
            .reshape(b, 4, 2, n, n)
            .transpose(1, 2)
        )
        return wave, field, amplitude, maps

    def optical_features(self, wave, field, amplitude, maps, stage=0):
        maximum = amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        ratio = amplitude / maximum
        peak = self.peak_attention(amplitude, maximum)
        # A global attention readout of learned features at the bright peak.
        pooled = (maps[:, 0] * ratio).sum((-2, -1), keepdim=True)
        peak_scale = (
            self.stage_peak_scales[stage]
            if self.stage_coefficients
            else self.peak_scale
        )
        first = (maps[:, 0] - peak_scale * peak * pooled) / maximum
        amplitudes = torch.stack((first, maps[:, 1]), 1)
        unit = field / amplitude.clamp_min(1e-8)
        optical = self.physics.backproject_prechirp(amplitudes * unit[:, None]).sum(2)
        tangent = (optical * wave.conj()).imag
        return amplitudes, tangent

    def step(self, targets, phase, stage=0):
        wave, field, amplitude, maps = self.neural_maps(targets, phase)
        _, tangent = self.optical_features(wave, field, amplitude, maps, stage)
        rms = tangent.square().mean((-2, -1), keepdim=True).clamp_min(1e-10).sqrt()
        features = tangent / rms
        weights = (
            self.stage_branch_weights[stage]
            if self.stage_coefficients
            else self.branch_weights
        )
        scale = (
            self.stage_step_scales[stage]
            if self.stage_coefficients
            else self.step_scale
        )
        effective_weights = weights[None, :, None, None]
        if self.controller is not None:
            factors = self.readout_factors(targets, amplitude, tangent, stage)
            effective_weights = effective_weights * factors[:, :2, None, None]
            scale = scale * factors[:, 2, None, None, None]
        logit = (features * effective_weights).sum(1, keepdim=True)
        return phase - 0.5 * torch.tanh(scale * logit)

    def forward(self, targets):
        phase = self.base(targets)
        for stage in range(self.depth):
            phase = self.step(targets, phase, stage)
        return torch.atan2(phase.sin(), phase.cos())
