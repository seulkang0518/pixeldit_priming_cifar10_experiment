#!/usr/bin/env python3
"""CIFAR-10 FID using the TensorFlow-compatible 2048-dimensional Inception.

Standalone: python fid.py --samples generated.npy --data-root /path/to/data
Samples must be NCHW (N, 3, 32, 32) or flattened (N, 3072), uint8 or
floating point in [0, 1]. Each row is one image, never a rendered grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

ASSETS = Path(__file__).resolve().parent / "assets"
WEIGHTS = ASSETS / "pt_inception-2015-12-05-6726825d.pth"


def load_inception(device):
    from _fid_inception import InceptionV3
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing FID weights: {WEIGHTS}")
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    expected = (ASSETS / "inception.sha256").read_text().split()[0]
    if digest != expected:
        raise RuntimeError("FID Inception weight checksum mismatch")
    return InceptionV3([3]).eval().to(device)


def feature_statistics(batches, inception, device):
    """Accumulate float64 mean/covariance without storing all image features."""
    count, mean, scatter = 0, None, None
    with torch.inference_mode():
        for images in batches:
            if images.ndim == 2 and images.shape[1] == 3072:
                images = images.reshape(-1, 3, 32, 32)
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
                raise ValueError("Expected CIFAR-10 NCHW images or flattened RGB rows")
            byte_pixels = images.dtype == torch.uint8
            images = images.to(device=device, dtype=torch.float32)
            if byte_pixels:
                images = images / 255
            if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
                raise ValueError("FID pixels must be finite and in [0, 1]")
            features = inception(images)[0].flatten(1).cpu().numpy().astype(np.float64)
            n = len(features)
            if not n:
                continue
            batch_mean = features.mean(axis=0)
            centered = features - batch_mean
            batch_scatter = centered.T @ centered
            if count == 0:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                scatter += batch_scatter + np.outer(delta, delta) * (count * n / (count + n))
                mean += delta * n / (count + n)
            count += n
    if count < 2:
        raise ValueError("FID requires at least two images per distribution")
    return mean, scatter / (count - 1), count


def frechet_distance(mean1, cov1, mean2, cov2):
    """Gaussian Wasserstein distance; symmetric PSD form handles singular covariances."""
    values, vectors = linalg.eigh(cov1)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    middle = root @ cov2 @ root
    eigenvalues = linalg.eigvalsh((middle + middle.T) / 2)
    value = float(np.sum((mean1 - mean2) ** 2) + np.trace(cov1) + np.trace(cov2)
                  - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum())
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite FID")
    return max(value, 0.0)


class CIFAR10FID:
    """One reference computation per run; independent fixed latents for comparisons."""
    def __init__(self, real_images, *, device, batch_size=64):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device, self.batch_size = device, batch_size
        self.inception = load_inception(device)
        print(f"Computing FID reference statistics: {len(real_images)} CIFAR-10 images", flush=True)
        self.reference = self.statistics(real_images)

    def statistics(self, images):
        return feature_statistics((images[i:i + self.batch_size]
                                   for i in range(0, len(images), self.batch_size)),
                                  self.inception, self.device)

    def score(self, stats):
        return frechet_distance(*self.reference[:2], *stats[:2])

    def evaluate_generator(self, model, latent_dim, *, count=10000, seed=12345):
        rng = torch.Generator().manual_seed(seed)
        training = model.training
        model.eval()
        def batches():
            for start in range(0, count, self.batch_size):
                z = torch.randn(min(self.batch_size, count - start), latent_dim, generator=rng)
                yield model(z.to(self.device))
        try:
            stats = feature_statistics(batches(), self.inception, self.device)
            return self.score(stats)
        finally:
            model.train(training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="Generated images as a .npy array")
    parser.add_argument("--data-root", type=Path, default=Path("/home/zongchen/data"))
    parser.add_argument("--reference-split", choices=("train", "test"), default="train")
    parser.add_argument("--class-id", type=int, choices=range(10))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("fid.json"))
    args = parser.parse_args()
    from data import download_cifar10, load_cifar10_split
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    archive = download_cifar10(args.data_root)
    images, labels = load_cifar10_split(archive, args.reference_split == "train")
    if args.class_id is not None:
        images = images[labels == args.class_id]
    evaluator = CIFAR10FID(images, device=device, batch_size=args.batch_size)
    samples = torch.from_numpy(np.load(args.samples, allow_pickle=False))
    stats = evaluator.statistics(samples)
    result = dict(fid=evaluator.score(stats), generated_count=stats[2], reference_count=len(images),
                  reference_split=args.reference_split, class_id=args.class_id, feature_dim=2048,
                  inception="pytorch-fid-v0.3.0", weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                  samples=str(args.samples.resolve()), samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  dataset_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
                  preprocessing="float [0,1]; bilinear 299x299 align_corners=False; scale to [-1,1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
