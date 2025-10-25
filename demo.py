"""Command line demo for the texture-aware α-shape pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import zlib
from pathlib import Path
from typing import List, Sequence, Tuple

from texture_alpha_shape import (
    AlphaStabilityAnalyzer,
    PointSampler,
    TextureAwareAlphaShape,
    TextureAwareAlphaShapeResult,
    TextureFeatureExtractor,
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
        default=1500,
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
        help="B,G,R colour for overlaying the contour",
    )
    return parser.parse_args()


def parse_color(value: str) -> Tuple[int, int, int]:
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


def _png_unfilter(filter_type: int, row: bytes, prev: bytes, bpp: int) -> bytes:
    row_vals = list(row)
    prev_vals = list(prev)
    result = [0] * len(row_vals)

    if filter_type == 0:
        return bytes(row_vals)
    if filter_type == 1:
        for idx, value in enumerate(row_vals):
            left = result[idx - bpp] if idx >= bpp else 0
            result[idx] = (value + left) & 0xFF
        return bytes(result)
    if filter_type == 2:
        for idx, value in enumerate(row_vals):
            result[idx] = (value + prev_vals[idx]) & 0xFF
        return bytes(result)
    if filter_type == 3:
        for idx, value in enumerate(row_vals):
            left = result[idx - bpp] if idx >= bpp else 0
            up = prev_vals[idx]
            result[idx] = (value + ((left + up) // 2)) & 0xFF
        return bytes(result)
    if filter_type == 4:
        for idx, value in enumerate(row_vals):
            left = result[idx - bpp] if idx >= bpp else 0
            up = prev_vals[idx]
            up_left = prev_vals[idx - bpp] if idx >= bpp else 0
            predictor = _paeth_predictor(left, up, up_left)
            result[idx] = (value + predictor) & 0xFF
        return bytes(result)
    raise ValueError(f"Unsupported PNG filter type {filter_type}")


def read_image(path: Path) -> List[List[int]]:
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
        offset += 4  # Skip CRC

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
    rows: List[bytes] = []
    pos = 0
    prev = bytes([0] * stride)
    for _ in range(height):
        filter_type = decompressed[pos]
        pos += 1
        row = decompressed[pos : pos + stride]
        pos += stride
        recon = _png_unfilter(filter_type, row, prev, bytes_per_pixel)
        rows.append(recon)
        prev = recon

    pixels: List[List[int]] = []
    if color_type == 0:
        for y in range(height):
            row_data = list(rows[y][:width])
            pixels.append(row_data)
        return pixels
    if color_type == 2:
        for y in range(height):
            row = rows[y]
            gray_row: List[int] = []
            for x in range(width):
                r = row[3 * x]
                g = row[3 * x + 1]
                b = row[3 * x + 2]
                gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
                gray_row.append(gray)
            pixels.append(gray_row)
        return pixels
    if color_type == 4:
        for y in range(height):
            row = rows[y]
            gray_row = [row[2 * x] for x in range(width)]
            pixels.append(gray_row)
        return pixels
    if color_type == 6:
        for y in range(height):
            row = rows[y]
            gray_row: List[int] = []
            for x in range(width):
                r = row[4 * x]
                g = row[4 * x + 1]
                b = row[4 * x + 2]
                gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
                gray_row.append(gray)
            pixels.append(gray_row)
        return pixels
    raise ValueError("Unsupported PNG format")


def write_image(path: Path, image: List[List[int]] | List[List[List[int]]]) -> None:
    if not image:
        raise ValueError("Image is empty")

    if isinstance(image[0][0], list):  # type: ignore[index]
        colour_type = 2
        height = len(image)
        width = len(image[0])
        rows: List[bytes] = []
        for row in image:  # type: ignore[assignment]
            flat: List[int] = []
            for pixel in row:  # type: ignore[assignment]
                if len(pixel) != 3:
                    raise ValueError("RGB rows must contain triplets")
                flat.extend(int(max(0, min(255, channel))) for channel in pixel)
            rows.append(bytes([0] + flat))
    else:
        colour_type = 0
        height = len(image)
        width = len(image[0])
        rows = [bytes([0] + [int(max(0, min(255, value))) for value in row]) for row in image]

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, colour_type, 0, 0, 0])
    )

    compressed = zlib.compress(b"".join(rows))

    with path.open("wb") as fh:
        fh.write(PNG_SIGNATURE)
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"IDAT", compressed))
        fh.write(_png_chunk(b"IEND", b""))


def overlay_contours(
    image: List[List[int]],
    result: TextureAwareAlphaShapeResult,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> List[List[List[int]]]:
    height = len(image)
    width = len(image[0]) if height else 0
    overlay: List[List[List[int]]] = []
    for y in range(height):
        row: List[List[int]] = []
        for x in range(width):
            value = int(image[y][x])
            row.append([value, value, value])
        overlay.append(row)

    mask = result.render_mask(thickness=thickness)
    for y in range(min(height, len(mask))):
        row_mask = mask[y]
        for x in range(min(width, len(row_mask))):
            if row_mask[x]:
                overlay[y][x] = [color[0], color[1], color[2]]
    return overlay


def linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def main() -> None:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = read_image(image_path)
    if not image or not image[0]:
        raise RuntimeError("Input image is empty")

    alpha_values = linspace(args.alpha_min, args.alpha_max, args.alpha_steps)

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
