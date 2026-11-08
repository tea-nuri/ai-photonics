"""Centered, single-FFT Fresnel propagation specified by the hackathon.

The global factor exp(ikz)/(i lambda z) is omitted: every evaluated
amplitude is independently normalized, so this factor cancels exactly.
The output quadratic phase is retained in field(), but cancels in abs().
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class OpticalConfig:
    size: int = 128
    wavelength_m: float = 532e-9
    pixel_pitch_m: float = 10e-6
    distances_m: tuple[float, ...] = (0.0241, 0.0253, 0.0265, 0.0277)

    def to_dict(self):
        return asdict(self)


def fft2c(x):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(x):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


class Fresnel(nn.Module):
    def __init__(self, config=OpticalConfig()):
        super().__init__()
        self.config = config
        # Calculate chirps in float64 before casting to avoid phase error.
        n = torch.arange(config.size, dtype=torch.float64) - config.size // 2
        r2 = n[:, None].square() + n[None, :].square()
        z = torch.tensor(config.distances_m, dtype=torch.float64)[:, None, None]
        angle_in = math.pi * config.pixel_pitch_m**2 / (config.wavelength_m * z) * r2
        output_pitch = config.wavelength_m * z / (config.size * config.pixel_pitch_m)
        angle_out = math.pi * output_pitch.square() / (config.wavelength_m * z) * r2
        self.register_buffer(
            "chirp",
            torch.polar(torch.ones_like(angle_in), angle_in).to(torch.complex64),
        )
        self.register_buffer(
            "output_chirp",
            torch.polar(torch.ones_like(angle_out), angle_out).to(torch.complex64),
        )

    def field(self, phase):
        """[B,1,N,N] phase → [B,4,N,N] complex output fields."""
        wave = torch.complex(phase.cos(), phase.sin())
        return fft2c(wave * self.chirp) * self.output_chirp

    def forward(self, phase):
        return self.field(phase).abs()

    def backproject_prechirp(self, fields):
        """Adjoint for fields parameterized before output quadratic phase.

        A trainable phase can absorb the output chirp without affecting the
        desired amplitude. This removes a redundant fixed multiplication.
        """
        return ifft2c(fields) * self.chirp.conj()

    def adjoint(self, fields):
        return self.backproject_prechirp(fields * self.output_chirp.conj())


def normalize_max(x):
    return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def phase_to_uint8(phase):
    p = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Phase contains non-finite values")
    # Optical phase is periodic. Keep the established encoding of values
    # already in range (including both endpoints); wrap other representatives.
    p = np.where(
        (p >= -math.pi) & (p <= math.pi), p, (p + math.pi) % (2 * math.pi) - math.pi
    )
    return np.rint((p + math.pi) * (255.0 / (2 * math.pi))).astype(np.uint8)


def uint8_to_phase(pixels):
    return np.asarray(pixels, dtype=np.float64) / 255.0 * (2 * math.pi) - math.pi


def numpy_reconstruct(pixels, config=OpticalConfig()):
    """Independent float64 reference, reconstructs the EXPORTED 8-bit DOE."""
    phase = uint8_to_phase(pixels)
    n = np.arange(config.size, dtype=np.float64) - config.size // 2
    r2 = (n[:, None] ** 2 + n[None, :] ** 2) * config.pixel_pitch_m**2
    result = []
    for z in config.distances_m:
        wave = np.exp(1j * (phase + math.pi * r2 / (config.wavelength_m * z)))
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave), norm="ortho"))
        amplitude = np.abs(field)
        result.append(amplitude / max(float(amplitude.max()), 1e-12))
    return np.stack(result)
