"""Jointly map four grayscale images to one 8-bit phase DOE."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from photonics.metrics import image_metrics
from photonics.model import load_model
from photonics.physics import numpy_reconstruct, phase_to_uint8


def load_inputs(paths):
    if len(paths) != 4:
        raise ValueError(f"Expected exactly four input images, got {len(paths)}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
                raise ValueError(f"{path}: expected PNG, 128x128, 8-bit grayscale (L)")
            pixels = np.array(image, dtype=np.float32) / 255.0
        images.append(pixels / max(float(pixels.max()), 1e-8))
    return np.stack(images)


def directory_inputs(folder):
    if not folder.is_dir():
        raise ValueError(f"Input directory does not exist: {folder}")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )


def select_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Metal is not available; use --device cpu")
    return name


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def predict(model, paths, output_dir, device):
    targets = load_inputs(paths)
    tensor = torch.from_numpy(targets[None]).to(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        phase = model(tensor)
    synchronize(device)
    elapsed = time.perf_counter() - start
    pixels = phase_to_uint8(phase[0, 0].cpu().numpy())
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "doe.png"
    Image.fromarray(pixels).save(destination)
    # Reconstruct from the actual exported DOE, including its 8-bit quantization.
    with Image.open(destination) as exported:
        pixels = np.array(exported)
    reconstructions = numpy_reconstruct(pixels, model.physics.config)
    planes = []
    for index, (target, restored, distance) in enumerate(
        zip(targets, reconstructions, model.physics.config.distances_m), 1
    ):
        filename = f"recon_{index}.png"
        restored_pixels = np.rint(np.clip(restored, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(restored_pixels).save(output_dir / filename)
        with Image.open(output_dir / filename) as exported:
            reopened = np.array(exported, dtype=np.float64) / 255.0
        planes.append(
            {
                "plane": index,
                "distance_mm": distance * 1000,
                "input": str(paths[index - 1]),
                "reconstruction": filename,
                **image_metrics(target, reopened),
            }
        )
    defined = [item["pearson"] for item in planes if item["pearson"] is not None]
    report = {
        "doe": "doe.png",
        "optics": model.physics.config.to_dict(),
        "phase_encoding": "pixel / 255 * 2*pi - pi",
        "reconstructed_from": "saved_and_reopened_8bit_doe_png",
        "metric_inputs": "saved_and_reopened_8bit_reconstruction_pngs; independent maximum normalization",
        "ssim_parameters": {
            "window": 11,
            "gaussian_sigma": 1.5,
            "data_range": 1.0,
            "sample_covariance": False,
        },
        "planes": planes,
        "mean": {
            "ssim": float(np.mean([item["ssim"] for item in planes])),
            "pearson": float(np.mean(defined)) if defined else None,
        },
        "inference_seconds": elapsed,
        "device": device,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return {
        "output": str(destination),
        "inference_seconds": elapsed,
        "device": device,
        "mean": report["mean"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--inputs", type=Path, nargs=4, metavar=("Z24_1", "Z25_3", "Z26_5", "Z27_7")
    )
    inputs.add_argument(
        "--input-dir",
        type=Path,
        help="One directory with four PNGs, sorted by filename",
    )
    inputs.add_argument(
        "--batch-dir",
        type=Path,
        help="Directory of sets; each subdirectory contains four PNGs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "weights/model.pt",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.batch_dir:
            if not args.batch_dir.is_dir():
                raise ValueError(f"Batch directory does not exist: {args.batch_dir}")
            folders = sorted(p for p in args.batch_dir.iterdir() if p.is_dir())
            if not folders:
                raise ValueError("Batch directory contains no image-set subdirectories")
            jobs = [(directory_inputs(p), args.output_dir / p.name) for p in folders]
        else:
            jobs = [
                (
                    args.inputs if args.inputs else directory_inputs(args.input_dir),
                    args.output_dir,
                )
            ]
        # Validate all image sets before inference starts.
        for paths, _ in jobs:
            load_inputs(paths)
        device = select_device(args.device)
        model, _ = load_model(args.weights, device)
        model.requires_grad_(False)
        for paths, output in jobs:
            print(json.dumps(predict(model, paths, output, device)), flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
