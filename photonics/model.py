"""Neural model construction and strict loading of bundled inference weights."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .physics import Fresnel, OpticalConfig


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.SiLU(),
        )


class SmallUNet(nn.Module):
    def __init__(self, cin, cout, width=12):
        super().__init__()
        self.enc1 = Block(cin, width)
        self.enc2 = Block(width, width * 2)
        self.bottom = Block(width * 2, width * 4)
        self.dec2 = Block(width * 6, width * 2)
        self.dec1 = Block(width * 3, width)
        self.out = nn.Conv2d(width, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bottom(F.avg_pool2d(b, 2))
        d = self.dec2(
            torch.cat(
                (
                    F.interpolate(
                        c, size=b.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    b,
                ),
                1,
            )
        )
        return self.out(
            self.dec1(
                torch.cat(
                    (
                        F.interpolate(
                            d, size=a.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        a,
                    ),
                    1,
                )
            )
        )


class DeepUNet(nn.Module):
    def __init__(self, cin, cout, width=16):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 6, width * 8]

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.GroupNorm(4, b),
                nn.SiLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.GroupNorm(4, b),
                nn.SiLU(),
            )

        self.encoders = nn.ModuleList(
            [block(cin if i == 0 else widths[i - 1], w) for i, w in enumerate(widths)]
        )
        self.decoders = nn.ModuleList(
            [block(widths[i] + widths[i - 1], widths[i - 1]) for i in range(4, 0, -1)]
        )
        self.out = nn.Conv2d(width, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(
                torch.cat(
                    (
                        F.interpolate(
                            x,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    1,
                )
            )
        return self.out(x)


class NeuralDOE(nn.Module):
    """Targets → learned object phases → adjoint → learned DOE → phase.

    Shared phase carriers and both CNNs are learned jointly on the training
    distribution. Carriers spread object information across the aperture.
    All parameters are frozen at inference; there are no test-time gradients.
    """

    def __init__(
        self,
        width=12,
        seed=1729,
        config=None,
        architecture="small",
        carrier_init="random",
    ):
        super().__init__()
        config = config or OpticalConfig()
        self.width, self.seed = width, seed
        self.architecture, self.carrier_init = architecture, carrier_init
        self.physics = Fresnel(config)
        network = DeepUNet if architecture == "deep" else SmallUNet
        self.object_net = network(14, 4, width)
        self.aperture_net = network(12, 1, width)
        gen = torch.Generator().manual_seed(seed)
        carrier = (
            torch.rand((1, 4, config.size, config.size), generator=gen) * 2 - 1
        ) * math.pi
        if carrier_init == "zero":
            carrier.zero_()
        elif carrier_init == "quadratic":
            n = torch.arange(config.size) - config.size // 2
            r2 = (n[:, None].square() + n[None, :].square()).float()[None, None]
            carrier = (
                r2
                * math.pi
                / config.size
                * torch.tensor([1.7, 1.9, 2.1, 2.3])[None, :, None, None]
            )
        self.carrier = nn.Parameter(carrier)
        n = torch.linspace(-1, 1, config.size)
        yy, xx = torch.meshgrid(n, n, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy))[None])
        self.plane_logits = nn.Parameter(torch.zeros(4))

    def forward(self, targets):
        coordinates = self.coordinates.expand(targets.shape[0], -1, -1, -1)
        carrier_features = torch.cat(
            (self.carrier.cos(), self.carrier.sin()), 1
        ).expand(targets.shape[0], -1, -1, -1)
        phase = self.carrier + math.pi * torch.tanh(
            self.object_net(torch.cat((targets, coordinates, carrier_features), 1))
        )
        fields = torch.complex(targets * phase.cos(), targets * phase.sin())
        back = self.physics.backproject_prechirp(fields)
        # Per-plane energy balance prevents a bright input from suppressing others.
        scale = back.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-5)
        back = back / scale
        combined = (back * self.plane_logits.softmax(0)[None, :, None, None]).sum(
            1, keepdim=True
        )
        features = torch.cat(
            (back.real, back.imag, combined.real, combined.imag, coordinates), 1
        )
        correction = math.pi * torch.tanh(self.aperture_net(features))
        # atan2(0, 0) is undefined in backward. Keep both components nonzero
        # for the degenerate all-black target, which is handled in CLI too.
        base = torch.atan2(combined.imag + 1e-7, combined.real + 1e-7)
        return torch.atan2(torch.sin(base + correction), torch.cos(base + correction))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    cfg = dict(checkpoint["optics"])
    cfg["distances_m"] = tuple(cfg["distances_m"])
    model = build_model(checkpoint["model"], OpticalConfig(**cfg))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def build_model(spec, config):
    if spec.get("architecture") == "moment_message":
        from .moment_message import MomentMessageDOE

        return MomentMessageDOE(
            spec["base"],
            config,
            spec["depth"],
            spec.get("message_width", 32),
            spec.get("local_width", 64),
            spec.get("global_width", 32),
        )
    if spec.get("architecture") == "dual_message":
        from .dual_message import DualMessageDOE

        return DualMessageDOE(
            spec["base"],
            config,
            spec["depth"],
            spec.get("message_width", 32),
            spec.get("stage_coefficients", False),
            spec.get("adaptive_readout", False),
            spec.get("soft_peak", False),
        )
    if spec.get("architecture") in {"cascade", "untied_cascade"}:
        from .cascade import CascadedDOE, UntiedCascadedDOE

        cls = (
            UntiedCascadedDOE
            if spec["architecture"] == "untied_cascade"
            else CascadedDOE
        )
        model = cls(
            spec["base"],
            config,
            spec.get("depth", 8),
            spec.get("message_width", 32),
            spec.get("context", False),
            spec.get("peak_feature", False),
            spec.get("statistics", False),
            spec.get("pyramid", False),
            spec.get("unit_messages", False),
            spec.get("local_statistics", False),
            spec.get("joint_fields", False),
        )
        if isinstance(model.correction, nn.ModuleList):
            widths = spec.get("global_aperture_widths", [0] * model.depth)
            if len(widths) != model.depth:
                raise ValueError("Global aperture metadata must match cascade depth")
            for cell, width in zip(model.correction, widths):
                if width:
                    cell.enable_global_aperture(width)
        elif spec.get("global_aperture_width", 0):
            model.correction.enable_global_aperture(spec["global_aperture_width"])
    elif spec.get("architecture") == "gated":
        from .gated import GatedDOE

        model = GatedDOE(spec["experts"], config)
    elif spec.get("architecture") == "refiner":
        from .refiner import LearnedRefinerDOE

        model = LearnedRefinerDOE(
            spec["base"],
            config,
            spec.get("message_width", 32),
            spec.get("context_width", 0),
            spec.get("linear_projection", False),
        )
    elif spec.get("architecture") == "operator":
        from .operator import FresnelOperatorDOE

        model = FresnelOperatorDOE(
            width=spec["width"],
            seed=spec["seed"],
            config=config,
            channels=spec.get("channels", 4),
            blocks=spec.get("blocks", 3),
            feature_kind=spec.get("feature_kind", "cartesian"),
            global_width=spec.get("global_width", 0),
            object_width=spec.get("object_width", 0),
            shared_width=spec.get("shared_width", 0),
        )
    else:
        model = NeuralDOE(
            width=spec["width"],
            seed=spec["seed"],
            config=config,
            architecture=spec.get("architecture", "small"),
            carrier_init=spec.get("carrier_init", "random"),
        )
    return model
