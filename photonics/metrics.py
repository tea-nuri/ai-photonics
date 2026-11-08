"""Full-image Pearson and Gaussian SSIM using NumPy float64."""

import numpy as np


def image_metrics(target, reconstructed):
    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(reconstructed, dtype=np.float64)
    a = a / max(float(a.max()), 1e-12)
    b = b / max(float(b.max()), 1e-12)
    x, y = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    pearson = float(np.sum(x * y) / denominator) if denominator > 1e-12 else None
    coords = np.arange(11, dtype=np.float64) - 5
    kernel = np.exp(-(coords * coords) / (2 * 1.5**2))
    kernel /= kernel.sum()
    values = np.stack((a, b, a * a, b * b, a * b))
    values = np.sum(
        np.lib.stride_tricks.sliding_window_view(values, 11, axis=-1) * kernel, axis=-1
    )
    values = np.sum(
        np.lib.stride_tricks.sliding_window_view(values, 11, axis=-2) * kernel, axis=-1
    )
    ma, mb, aa, bb, ab = values
    va, vb, cov = aa - ma * ma, bb - mb * mb, ab - ma * mb
    ssim = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return {"ssim": float(ssim.mean()), "pearson": pearson}
