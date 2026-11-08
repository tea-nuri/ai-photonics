"""Fixed-depth neural residual layers with phase-equivariant optical features.

Every message coefficient and phase residual is predicted by a CNN. There is
no amplitude replacement, autograd, objective, or phase optimizer in inference.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .physics import fft2c


class QuadraticCarrier(nn.Module):
    """Fixed smooth optical seed; the subsequent learned CNN supplies the image."""

    def __init__(self, config, curvature=0.5):
        super().__init__()
        from .physics import Fresnel

        self.physics = Fresnel(config)
        n = torch.arange(config.size, dtype=torch.float64) - config.size // 2
        r2 = n[:, None].square() + n[None, :].square()
        z = sum(config.distances_m) / len(config.distances_m)
        phase = (
            math.pi * curvature / config.size
            - math.pi * config.pixel_pitch_m**2 / (config.wavelength_m * z)
        ) * r2
        self.register_buffer("phase", phase.float()[None, None])

    def forward(self, target):
        return self.phase.expand(len(target), -1, -1, -1)


class EquivariantCorrection(nn.Module):
    def __init__(
        self,
        width=32,
        context=False,
        peak_feature=False,
        statistics=False,
        unit_messages=False,
        local_statistics=False,
        joint_fields=False,
    ):
        super().__init__()
        self.width = width
        self.statistics = False
        self.peak_feature = False
        self.unit_messages = False
        self.local_statistics = False
        self.joint_fields = False
        self.other_fields = False
        self.global_aperture = None
        self.global_aperture_width = 0
        self.register_buffer("local_kernel", None)
        self.object_net = nn.Sequential(
            nn.Conv2d(9, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Conv2d(width, 2, 1),
        )
        nn.init.zeros_(self.object_net[-1].weight)
        nn.init.zeros_(self.object_net[-1].bias)
        self.aperture = nn.Sequential(
            nn.Conv2d(8, 32, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1, bias=False),
        )
        self.linear = nn.Conv2d(8, 1, 1, bias=False)
        self.context_film = self.context_conv = None
        if context:
            self.enable_context()
        if peak_feature:
            self.enable_peak_feature()
        if statistics:
            self.enable_statistics()
        if unit_messages:
            self.enable_unit_messages()
        if local_statistics:
            self.enable_local_statistics()
        if joint_fields:
            self.enable_joint_fields()

    def enable_joint_fields(self):
        if self.joint_fields:
            return
        previous = self.object_net[0]
        expanded = nn.Conv2d(previous.in_channels + 20, self.width, 3, padding=1).to(
            previous.weight.device
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : previous.in_channels].copy_(previous.weight)
            expanded.bias.copy_(previous.bias)
        self.object_net[0] = expanded
        self.joint_fields = True

    def enable_other_fields(self):
        if self.other_fields:
            return
        previous = self.object_net[0]
        expanded = nn.Conv2d(previous.in_channels + 4, self.width, 3, padding=1).to(
            previous.weight.device
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : previous.in_channels].copy_(previous.weight)
            expanded.bias.copy_(previous.bias)
        self.object_net[0] = expanded
        self.other_fields = True

    def enable_global_aperture(self, width=8):
        """Add a zero-initialized multiscale aperture CNN, preserving the seed."""
        if self.global_aperture is not None:
            if self.global_aperture_width != width:
                raise ValueError("Global aperture width is already fixed")
            return
        from .model import DeepUNet

        self.global_aperture = DeepUNet(10, 1, width).to(self.linear.weight.device)
        self.global_aperture_width = width

    def enable_local_statistics(self):
        if self.local_statistics:
            return
        previous = self.object_net[0]
        expanded = nn.Conv2d(previous.in_channels + 5, self.width, 3, padding=1).to(
            previous.weight.device
        )
        with torch.no_grad():
            expanded.weight.zero_()
            front = previous.in_channels - (20 if self.joint_fields else 0)
            expanded.weight[:, :front].copy_(previous.weight[:, :front])
            expanded.weight[:, front + 5 :].copy_(previous.weight[:, front:])
            expanded.bias.copy_(previous.bias)
        self.object_net[0] = expanded
        x = (
            torch.arange(11, device=previous.weight.device, dtype=previous.weight.dtype)
            - 5
        )
        g = torch.exp(-x.square() / (2 * 1.5**2))
        g = g / g.sum()
        self.local_kernel = (g[:, None] * g[None, :])[None, None]
        self.local_statistics = True

    def local_features(self, a, target):
        # Local image moments are neural input features. No phase derivative,
        # loss function, or optimization rule is evaluated by this layer.
        b, planes, n, _ = a.shape
        moments = F.conv2d(
            torch.cat((a, target, a * a, target * target, a * target)).reshape(
                -1, 1, n, n
            ),
            self.local_kernel.to(a.dtype),
            padding=5,
        )
        ma, mt, aa, tt, at = moments.reshape(5, b, planes, n, n).unbind(0)
        sa = (aa - ma * ma).clamp_min(1e-8).sqrt()
        st = (tt - mt * mt).clamp_min(1e-8).sqrt()
        cov = (at - ma * mt) / (sa * st + 0.03**2)
        return torch.stack((ma, mt, 10 * sa, 10 * st, cov), 2)

    def enable_unit_messages(self):
        if self.unit_messages:
            return

        # Amplitude losses act along the unit complex field. An additional
        # learned branch avoids requiring the CNN to approximate 1/amplitude
        # near dark pixels. Zero initialization preserves the previous graph.
        def expand(previous):
            expanded = nn.Conv2d(
                previous.in_channels,
                4,
                previous.kernel_size,
                padding=previous.padding,
                dilation=previous.dilation,
            ).to(previous.weight.device)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:2].copy_(previous.weight)
                expanded.bias[:2].copy_(previous.bias)
            return expanded

        self.object_net[-1] = expand(self.object_net[-1])
        if self.context_conv is not None:
            self.context_conv[-1] = expand(self.context_conv[-1])
        self.unit_messages = True

    def enable_peak_feature(self):
        previous = self.object_net[0]
        if self.peak_feature:
            return
        expanded = nn.Conv2d(previous.in_channels + 1, self.width, 3, padding=1).to(
            previous.weight.device
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :9].copy_(previous.weight[:, :9])
            if previous.in_channels > 9:
                expanded.weight[:, 10:].copy_(previous.weight[:, 9:])
            expanded.bias.copy_(previous.bias)
        self.object_net[0] = expanded
        self.peak_feature = True

    def enable_statistics(self):
        if self.statistics:
            return
        previous = self.object_net[0]
        expanded = nn.Conv2d(previous.in_channels + 6, self.width, 3, padding=1).to(
            previous.weight.device
        )
        with torch.no_grad():
            expanded.weight.zero_()
            front = 9 + int(self.peak_feature)
            expanded.weight[:, :front].copy_(previous.weight[:, :front])
            if previous.in_channels > front:
                expanded.weight[:, front + 6 :].copy_(previous.weight[:, front:])
            expanded.bias.copy_(previous.bias)
        self.object_net[0] = expanded
        self.statistics = True

    def enable_context(self):
        if self.context_film is not None:
            return
        w = self.width
        self.context_film = nn.Sequential(
            nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 2 * w)
        )
        self.context_conv = nn.Sequential(
            nn.Conv2d(w, w, 3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Conv2d(w, 4 if self.unit_messages else 2, 1),
        )
        for layer in (self.context_film[-1], self.context_conv[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.context_film.to(self.linear.weight.device)
        self.context_conv.to(self.linear.weight.device)

    def encode_fields(self, targets, phase, physics, other_phase=None):
        n = targets.shape[-1]
        wave = torch.complex(phase.cos(), phase.sin())
        field = fft2c(wave * physics.chirp)
        amplitude = field.abs()
        rms = amplitude.square().mean((-2, -1), keepdim=True).clamp_min(1e-8).sqrt()
        maximum = amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
        trms = targets.square().mean((-2, -1), keepdim=True).clamp_min(1e-8).sqrt()
        coord = torch.linspace(-1, 1, n, device=targets.device, dtype=targets.dtype)
        y, x = torch.meshgrid(coord, coord, indexing="ij")
        depth = torch.linspace(-1, 1, 4, device=targets.device, dtype=targets.dtype)
        features = torch.stack(
            (
                amplitude / rms,
                amplitude / maximum,
                targets,
                targets / trms,
                maximum.expand_as(targets),
                trms.expand_as(targets),
                x.expand_as(targets),
                y.expand_as(targets),
                depth[None, :, None, None].expand_as(targets),
            ),
            2,
        )
        if self.peak_feature:
            features = torch.cat(
                (features, (amplitude == maximum).to(targets.dtype)[:, :, None]), 2
            )
        if self.statistics:
            # Global image statistics let the learned message distinguish
            # brightness, contrast and covariance errors at otherwise equal
            # local patches. These are input features, not an optimizer step.
            a = amplitude / rms
            ma = a.mean((-2, -1), keepdim=True)
            mt = targets.mean((-2, -1), keepdim=True)
            sa = (a - ma).square().mean((-2, -1), keepdim=True).clamp_min(1e-8).sqrt()
            st = (
                (targets - mt)
                .square()
                .mean((-2, -1), keepdim=True)
                .clamp_min(1e-8)
                .sqrt()
            )
            corr = ((a - ma) * (targets - mt)).mean((-2, -1), keepdim=True) / (sa * st)
            stats = torch.stack(
                [
                    v.expand_as(targets)
                    for v in (ma, mt, sa, st, corr, maximum * trms / rms)
                ],
                2,
            )
            features = torch.cat((features, stats), 2)
        if self.local_statistics:
            if targets.device.type in {"cuda", "cpu"}:
                with torch.autocast(targets.device.type, enabled=False):
                    local = self.local_features(amplitude / maximum, targets)
            else:
                local = self.local_features(amplitude / maximum, targets)
            features = torch.cat((features, local), 2)
        if self.joint_fields:
            # Early exchange between depth planes and phase topology. All
            # products are invariant to a common optical phase piston.
            unit = field / torch.maximum(amplitude, 0.03 * rms)
            relative = unit[:, :, None] * unit[:, None].conj()
            dx = torch.roll(unit, -1, -1) * unit.conj()
            dy = torch.roll(unit, -1, -2) * unit.conj()
            joint = torch.cat(
                (
                    (amplitude / rms)[:, None].expand(-1, 4, -1, -1, -1),
                    targets[:, None].expand(-1, 4, -1, -1, -1),
                    relative.real,
                    relative.imag,
                    dx.real[:, :, None],
                    dx.imag[:, :, None],
                    dy.real[:, :, None],
                    dy.imag[:, :, None],
                ),
                2,
            )
            features = torch.cat((features, joint), 2)
        if self.other_fields:
            if other_phase is None:
                raise ValueError("This message encoder requires a second neural phase")
            other_wave = torch.complex(other_phase.cos(), other_phase.sin())
            other = fft2c(other_wave * physics.chirp)
            other_amplitude = other.abs()
            other_rms = (
                other_amplitude.square()
                .mean((-2, -1), keepdim=True)
                .clamp_min(1e-8)
                .sqrt()
            )
            other_max = other_amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
            relative = (other / torch.maximum(other_amplitude, 0.03 * other_rms)) * (
                field / torch.maximum(amplitude, 0.03 * rms)
            ).conj()
            additional = torch.stack(
                (
                    other_amplitude / other_rms,
                    other_amplitude / other_max,
                    relative.real,
                    relative.imag,
                ),
                2,
            )
            features = torch.cat((features, additional), 2)
        features = features.flatten(0, 1)
        encoded = self.object_net[:6](features)
        if self.context_film is not None:
            scale, bias = self.context_film(encoded.mean((-2, -1))).chunk(2, 1)
            encoded = (
                encoded * (1 + 0.1 * scale.tanh()[:, :, None, None])
                + 0.1 * bias.tanh()[:, :, None, None]
            )
        return wave, field, amplitude, rms, encoded

    def messages(self, targets, phase, physics, other_phase=None):
        b, _, n, _ = targets.shape
        wave, field, amplitude, rms, encoded = self.encode_fields(
            targets, phase, physics, other_phase
        )
        coefficients = self.object_net[-1](encoded)
        if self.context_conv is not None:
            coefficients = coefficients + self.context_conv(encoded)
        coefficients = coefficients.to(field.real.dtype).reshape(
            b, 4, 4 if self.unit_messages else 2, n, n
        )
        # Multiplication preserves the phase transformation of a complex field;
        # the two scalar coefficients are entirely learned, not projection rules.
        messages = (
            torch.complex(coefficients[:, :, 0], coefficients[:, :, 1]) * field / rms
        )
        if self.unit_messages:
            messages = messages + torch.complex(
                coefficients[:, :, 2], coefficients[:, :, 3]
            ) * field / torch.maximum(amplitude, 0.03 * rms)
        return wave, field, messages

    def decode(self, wave, messages, physics):
        relative = physics.backproject_prechirp(messages) * wave.conj()
        features = torch.cat((relative.real, relative.imag), 1)
        logit = self.linear(features) + self.aperture(features)
        if self.global_aperture is not None:
            n = features.shape[-1]
            coord = torch.linspace(
                -1, 1, n, device=features.device, dtype=features.dtype
            )
            y, x = torch.meshgrid(coord, coord, indexing="ij")
            coordinates = torch.stack((x, y))[None].expand(len(features), -1, -1, -1)
            logit = logit + self.global_aperture(torch.cat((features, coordinates), 1))
        return 0.5 * torch.tanh(logit)

    def forward(self, targets, phase, physics):
        wave, _, messages = self.messages(targets, phase, physics)
        return self.decode(wave, messages, physics)


class CascadedDOE(nn.Module):
    def __init__(
        self,
        base_spec,
        config,
        depth=8,
        width=32,
        context=False,
        peak_feature=False,
        statistics=False,
        pyramid=False,
        unit_messages=False,
        local_statistics=False,
        joint_fields=False,
    ):
        super().__init__()
        from .model import build_model

        self.base_spec = base_spec
        self.pyramid = pyramid
        self.depth, self.message_width = depth, width
        self.width, self.seed = base_spec["width"], base_spec["seed"]
        self.architecture = "cascade"
        self.base = build_model(base_spec, config)
        self.base.requires_grad_(False)
        self.physics = self.base.physics
        self.correction = EquivariantCorrection(
            width,
            context,
            peak_feature,
            statistics,
            unit_messages,
            local_statistics,
            joint_fields,
        )

    def specification(self):
        return {
            "architecture": "cascade",
            "width": self.width,
            "seed": self.seed,
            "base": self.base_spec,
            "depth": self.depth,
            "message_width": self.message_width,
            "context": self.correction.context_film is not None,
            "peak_feature": self.correction.peak_feature,
            "statistics": self.correction.statistics,
            "pyramid": self.pyramid,
            "unit_messages": self.correction.unit_messages,
            "local_statistics": self.correction.local_statistics,
            "joint_fields": self.correction.joint_fields,
            "global_aperture_width": self.correction.global_aperture_width,
        }

    def resolution(self, stage):
        if not self.pyramid:
            return self.physics.config.size
        sizes = [8, 16, 32, 64, 128]
        return min(
            self.physics.config.size,
            sizes[min(stage * len(sizes) // self.depth, len(sizes) - 1)],
        )

    def step(self, targets, phase, stage):
        delta = self.correction(targets, phase, self.physics)
        size = self.resolution(stage)
        if size < self.physics.config.size:
            delta = F.interpolate(
                F.adaptive_avg_pool2d(delta, size),
                size=phase.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
        return phase + delta

    def forward(self, targets):
        phase = self.base(targets)
        # Fixed depth is a checkpoint architecture parameter, never a convergence
        # condition or a per-input search. Like a recurrent CNN, weights are tied.
        for stage in range(self.depth):
            phase = self.step(targets, phase, stage)
        return torch.atan2(phase.sin(), phase.cos())


class UntiedCascadedDOE(CascadedDOE):
    """A fixed feedforward stack with independent learned parameters per layer."""

    def __init__(
        self,
        base_spec,
        config,
        depth=64,
        width=32,
        context=True,
        peak_feature=True,
        statistics=True,
        pyramid=False,
        unit_messages=True,
        local_statistics=False,
        joint_fields=False,
    ):
        if pyramid:
            raise ValueError("Untied layers currently use the full aperture grid")
        super().__init__(
            base_spec,
            config,
            depth,
            width,
            context,
            peak_feature,
            statistics,
            False,
            unit_messages,
            local_statistics,
            joint_fields,
        )
        prototype = self.correction
        self.correction = nn.ModuleList(
            [copy.deepcopy(prototype) for _ in range(depth)]
        )
        self.architecture = "untied_cascade"

    @classmethod
    def from_tied(cls, source):
        if source.pyramid or isinstance(source.correction, nn.ModuleList):
            raise ValueError("Expected a tied full-resolution source")
        cell = source.correction
        model = cls(
            source.base_spec,
            source.physics.config,
            source.depth,
            cell.width,
            cell.context_film is not None,
            cell.peak_feature,
            cell.statistics,
            False,
            cell.unit_messages,
            cell.local_statistics,
            cell.joint_fields,
        ).to(next(source.parameters()).device)
        model.base.load_state_dict(source.base.state_dict())
        for layer in model.correction:
            if cell.global_aperture_width:
                layer.enable_global_aperture(cell.global_aperture_width)
            layer.load_state_dict(cell.state_dict())
        return model

    def specification(self):
        cell = self.correction[0]
        return {
            "architecture": "untied_cascade",
            "width": self.width,
            "seed": self.seed,
            "base": self.base_spec,
            "depth": self.depth,
            "message_width": self.message_width,
            "context": cell.context_film is not None,
            "peak_feature": cell.peak_feature,
            "statistics": cell.statistics,
            "pyramid": False,
            "unit_messages": cell.unit_messages,
            "local_statistics": cell.local_statistics,
            "joint_fields": cell.joint_fields,
            "global_aperture_widths": [
                c.global_aperture_width for c in self.correction
            ],
        }

    def step(self, targets, phase, stage):
        return phase + self.correction[stage](targets, phase, self.physics)
