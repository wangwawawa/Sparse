"""Command line demo for the texture-aware α-shape pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import zlib
from pathlib import Path

import numpy as np

from texture_alpha_shape import (
    AlphaStabilityAnalyzer,
    PointSampler,
    TextureAwareAlphaShape,
    TextureFeatureExtractor,
    TextureAwareAlphaShapeResult,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = len(data).to_bytes(4, "big")
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return length + chunk_type + data + crc.to_bytes(4, "big")


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_unfilter(filter_type: int, row: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    result = np.empty_like(row)
    if filter_type == 0:
        result[:] = row
    elif filter_type == 1:
        for idx in range(row.size):
            left = result[idx - bpp] if idx >= bpp else 0
            result[idx] = (row[idx] + left) & 0xFF
    elif filter_type == 2:
        result[:] = (row + prev) & 0xFF
    elif filter_type == 3:
        for idx in range(row.size):
            left = result[idx - bpp] if idx >= bpp else 0
            up = prev[idx]
            result[idx] = (row[idx] + ((left + up) // 2)) & 0xFF
    elif filter_type == 4:
        for idx in range(row.size):
            left = result[idx - bpp] if idx >= bpp else 0
            up = prev[idx]
            up_left = prev[idx - bpp] if idx >= bpp else 0
            predictor = _paeth_predictor(left, up, up_left)
            result[idx] = (row[idx] + predictor) & 0xFF
    else:
        raise ValueError(f"Unsupported PNG filter type {filter_type}")
    return result.astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Only 8-bit PNG images are supported")

    offset = len(PNG_SIGNATURE)
    width = height = None
    color_type = None
    bit_depth = None
    idat = bytearray()

    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        chunk_type = data[offset : offset + 4]
        offset += 4
        chunk_data = data[offset : offset + length]
        offset += length
        offset += 4  # CRC (ignored)

        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            if bit_depth != 8:
                raise ValueError("Only 8-bit PNG images are supported")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None:
        raise ValueError("Invalid PNG file")

    decompressed = zlib.decompress(bytes(idat))
    bytes_per_pixel = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bytes_per_pixel is None:
        raise ValueError("Unsupported PNG colour type")

    stride = width * bytes_per_pixel
    rows = []
    pos = 0
    prev = np.zeros(stride, dtype=np.uint8)
    for _ in range(height):
        filter_type = decompressed[pos]
        pos += 1
        row = np.frombuffer(decompressed[pos : pos + stride], dtype=np.uint8)
        pos += stride
        recon = _png_unfilter(filter_type, row, prev, bytes_per_pixel)
        rows.append(recon)
        prev = recon

    image = np.vstack(rows).reshape(height, stride)
    if color_type == 0:
        return image.reshape(height, width)
    if color_type == 2:
        rgb = image.reshape(height, width, 3)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        return gray
    if color_type == 4:
        rgba = image.reshape(height, width, 2)
        return rgba[..., 0]
    if color_type == 6:
        rgba = image.reshape(height, width, 4)
        rgb = rgba[..., :3]
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        return gray
    raise ValueError("Unsupported PNG format")


def write_image(path: Path, image: np.ndarray) -> None:
    if image.ndim == 2:
        colour_type = 0
        payload = image.astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 3:
        colour_type = 2
        payload = image.astype(np.uint8)
    else:
        raise ValueError("Only grayscale or RGB images can be saved")

    height, width = payload.shape[:2]
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, colour_type, 0, 0, 0])
    )

    if colour_type == 0:
        rows = [b"\x00" + row.tobytes() for row in payload]
    else:
        rows = [b"\x00" + row.reshape(-1).tobytes() for row in payload]
    compressed = zlib.compress(b"".join(rows))

    with path.open("wb") as fh:
        fh.write(PNG_SIGNATURE)
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"IDAT", compressed))
        fh.write(_png_chunk(b"IEND", b""))


def overlay_contours(
    image: np.ndarray,
    result: TextureAwareAlphaShapeResult,
    color: tuple,
    thickness: int = 2,
) -> np.ndarray:
    overlay = np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
    mask = result.render_mask(thickness=thickness)
    mask = mask.astype(bool)
    for channel, value in enumerate(color):
        overlay[..., channel] = np.where(mask, value, overlay[..., channel])
    return overlay


def main() -> None:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = read_image(image_path)
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    if image.ndim != 2:
        raise RuntimeError("Input image must be grayscale or convertible to grayscale")
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = (image * 255.0).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

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
    overlay = overlay_contours(image, best, color)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / f"{image_path.stem}_texture_alpha.png"
    write_image(overlay_path, overlay)

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

