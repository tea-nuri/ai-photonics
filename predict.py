"""Jointly map four grayscale images to one 8-bit phase DOE."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from photonics.model import load_model
from photonics.physics import phase_to_uint8


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
    tensor = torch.from_numpy(load_inputs(paths)[None]).to(device)
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
    return {"output": str(destination), "inference_seconds": elapsed, "device": device}


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
