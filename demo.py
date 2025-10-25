"""Command line demo for the texture-aware α-shape pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from texture_alpha_shape import AlphaStabilityAnalyzer, PointSampler, TextureAwareAlphaShape, TextureFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input grayscale image")
    parser.add_argument(
        "--alpha-min", type=float, default=5.0, help="Minimum α for the sweep"
    )
    parser.add_argument(
        "--alpha-max", type=float, default=25.0, help="Maximum α for the sweep"
    )
    parser.add_argument(
        "--alpha-steps", type=int, default=10, help="Number of α samples"
    )
    parser.add_argument(
        "--beta", type=float, default=4.0, help="Texture weighting parameter β"
    )
    parser.add_argument(
        "--lambda", dest="lam", type=float, default=0.7, help="CSI balance λ"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=4000,
        help="Maximum number of sampled points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory where results will be stored",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Persist stability analysis as JSON",
    )
    parser.add_argument(
        "--contour-color",
        type=str,
        default="0,0,255",
        help="BGR colour for overlaying the contour",
    )
    return parser.parse_args()


def parse_color(value: str) -> tuple:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("Colour must be specified as B,G,R")
    return tuple(int(p) for p in parts)


def overlay_contours(image: np.ndarray, contours: List[np.ndarray], color: tuple) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        pts = np.round(contour).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=2)
    return overlay


def main() -> None:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)

    extractor = TextureFeatureExtractor()
    features = extractor.extract(image)

    sampler = PointSampler(max_points=args.max_points)
    samples = sampler.sample(image, features)

    tas = TextureAwareAlphaShape(beta=args.beta)
    results = [tas.build(samples, alpha=float(alpha)) for alpha in alpha_values]

    analyzer = AlphaStabilityAnalyzer(lam=args.lam)
    analysis = analyzer.evaluate(results)

    best = analysis.best_result
    color = parse_color(args.contour_color)
    overlay = overlay_contours(image, best.contours(), color)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / f"{image_path.stem}_texture_alpha.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Saved contour overlay to {overlay_path}")
    print(f"Selected α*: {analysis.best_record.alpha:.3f}")
    print(f"Texture consistency: {analysis.best_record.texture_consistency:.4f}")
    print(f"Geometry CSI: {analysis.best_record.csi_geometry:.4f}")
    print(f"Texture CSI: {analysis.best_record.csi_texture:.4f}")

    if args.save_metrics:
        metrics_path = args.output_dir / f"{image_path.stem}_stability.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(
                [dataclasses.asdict(record) for record in analysis.records],
                fh,
                indent=2,
            )
        print(f"Saved stability metrics to {metrics_path}")


if __name__ == "__main__":
    main()

