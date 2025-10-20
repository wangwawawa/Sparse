"""Gradient-guided binary segmentation followed by sparse contour recovery."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from sparse import edge as sparse_edge


EPS = 1e-8


def anisotropic_gaussian_derivative_kernel(
    order: int,
    sigma_major: float,
    sigma_minor: float,
    theta: float,
    size: int | None = None,
) -> np.ndarray:
    """Create a rotated anisotropic Gaussian derivative kernel."""
    if order not in (1, 2):
        raise ValueError("Only first and second order derivatives are supported")

    sigma_max = max(sigma_major, sigma_minor)
    if size is None:
        size = int(np.ceil(6 * sigma_max)) | 1  # ensure odd size
    half = size // 2

    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta

    gaussian = np.exp(
        -0.5 * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2)
    )

    if order == 1:
        kernel = -(x_rot / (sigma_major**2)) * gaussian
    else:
        kernel = ((x_rot**2 - sigma_major**2) / (sigma_major**4)) * gaussian

    kernel -= kernel.mean()
    norm = np.sqrt((kernel**2).sum())
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.float32)


@dataclass
class GradientResponses:
    magnitude: np.ndarray
    direction: np.ndarray
    first_order: np.ndarray
    second_order: np.ndarray


def compute_multi_direction_gradients(
    image: np.ndarray,
    num_orientations: int = 8,
    sigma_major: float = 3.0,
    sigma_minor: float = 1.5,
) -> GradientResponses:
    """Compute directional gradients using anisotropic Gaussian derivatives."""
    if image.ndim != 2:
        raise ValueError("Expected a single-channel image")

    thetas = np.linspace(0.0, np.pi, num=num_orientations, endpoint=False)
    first_responses = []
    second_responses = []

    for theta in thetas:
        k1 = anisotropic_gaussian_derivative_kernel(1, sigma_major, sigma_minor, theta)
        k2 = anisotropic_gaussian_derivative_kernel(2, sigma_major, sigma_minor, theta)
        response_first = cv2.filter2D(image, cv2.CV_32F, k1, borderType=cv2.BORDER_REFLECT)
        response_second = cv2.filter2D(image, cv2.CV_32F, k2, borderType=cv2.BORDER_REFLECT)
        first_responses.append(response_first)
        second_responses.append(response_second)

    first_stack = np.stack(first_responses, axis=0)
    second_stack = np.stack(second_responses, axis=0)

    first_abs = np.abs(first_stack)
    best_idx = np.argmax(first_abs, axis=0)
    direction = thetas[best_idx]

    first_selected = np.take_along_axis(first_stack, best_idx[np.newaxis, ...], axis=0)[0]
    second_selected = np.take_along_axis(second_stack, best_idx[np.newaxis, ...], axis=0)[0]

    magnitude = np.sqrt(first_selected**2 + second_selected**2)

    return GradientResponses(
        magnitude=magnitude.astype(np.float32),
        direction=direction.astype(np.float32),
        first_order=first_selected.astype(np.float32),
        second_order=second_selected.astype(np.float32),
    )


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Suppress non-maximum responses along gradient directions."""
    mag = magnitude.astype(np.float32, copy=False)
    angle = np.degrees(direction)
    angle = (angle + 180.0) % 180.0
    quantized = np.round(angle / 45.0).astype(np.int32) % 4

    suppressed = np.zeros_like(mag)

    shift_left = np.roll(mag, 1, axis=1)
    shift_right = np.roll(mag, -1, axis=1)
    shift_up = np.roll(mag, 1, axis=0)
    shift_down = np.roll(mag, -1, axis=0)

    shift_upleft = np.roll(shift_left, 1, axis=0)
    shift_upright = np.roll(shift_right, 1, axis=0)
    shift_downleft = np.roll(shift_left, -1, axis=0)
    shift_downright = np.roll(shift_right, -1, axis=0)

    masks = [
        quantized == 0,
        quantized == 1,
        quantized == 2,
        quantized == 3,
    ]
    neighbors = [
        (shift_left, shift_right),
        (shift_upright, shift_downleft),
        (shift_up, shift_down),
        (shift_upleft, shift_downright),
    ]

    for mask, (pos, neg) in zip(masks, neighbors):
        comparison = (mag >= pos) & (mag >= neg)
        selected = mask & comparison
        suppressed[selected] = mag[selected]

    suppressed[[0, -1], :] = 0
    suppressed[:, [0, -1]] = 0
    return suppressed


def adaptive_threshold(magnitude: np.ndarray, window: int = 21, k: float = 0.7) -> np.ndarray:
    """Adaptive threshold based on local mean and deviation."""
    if window % 2 == 0:
        window += 1

    mag = magnitude.copy()
    mag[mag < 0] = 0

    mean = cv2.blur(mag, (window, window))
    mean_sq = cv2.blur(mag**2, (window, window))
    variance = np.clip(mean_sq - mean**2, a_min=0, a_max=None)
    std = np.sqrt(variance)

    threshold_map = mean + k * std
    global_threshold = mag.mean() + 0.5 * mag.std()

    binary = (mag > threshold_map) & (mag > global_threshold)
    return binary.astype(np.uint8)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    flood = binary.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, flood)
    return filled


def largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    sizes = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(sizes)) + 1
    best_area = int(sizes[best_idx - 1])
    if best_area < min_area:
        return mask

    result = np.zeros_like(mask)
    result[labels == best_idx] = 255
    return result


def build_binary_mask(
    image: np.ndarray,
    responses: GradientResponses,
    adaptive_window: int,
    adaptive_k: float,
    min_area_ratio: float,
) -> np.ndarray:
    suppressed = non_maximum_suppression(responses.magnitude, responses.direction)
    adaptive = adaptive_threshold(suppressed, adaptive_window, adaptive_k)

    suppressed_norm = cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX)
    suppressed_norm = suppressed_norm.astype(np.uint8)

    adaptive_u8 = (adaptive * 255).astype(np.uint8)
    _, otsu = cv2.threshold(suppressed_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.bitwise_or(adaptive_u8, otsu)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    filled = fill_holes(dilated)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    _, global_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    global_mask = cv2.bitwise_not(global_mask)

    refined = cv2.bitwise_and(filled, global_mask)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_area = max(int(min_area_ratio * image.shape[0] * image.shape[1]), 1)
    refined = largest_component(refined, min_area)

    refined = cv2.GaussianBlur(refined, (0, 0), sigmaX=0.8)
    _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
    return refined


def process_image(
    image_path: str,
    output_dir: str,
    num_orientations: int = 8,
    sigma_major: float = 3.0,
    sigma_minor: float = 1.5,
    adaptive_window: int = 21,
    adaptive_k: float = 0.7,
    min_area_ratio: float = 5e-4,
    eps: float = 1e-2,
) -> Tuple[str, str]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    responses = compute_multi_direction_gradients(
        image,
        num_orientations=num_orientations,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
    )
    initial_mask = build_binary_mask(
        image,
        responses,
        adaptive_window=adaptive_window,
        adaptive_k=adaptive_k,
        min_area_ratio=min_area_ratio,
    )

    sparse_mask = sparse_edge(initial_mask, eps)

    if sparse_mask.ndim == 3:
        sparse_mask = cv2.cvtColor(sparse_mask, cv2.COLOR_BGR2GRAY)

    _, sparse_mask = cv2.threshold(sparse_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        sparse_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = np.zeros_like(sparse_mask)
    if contours:
        cv2.drawContours(contour_image, contours, -1, 255, thickness=1)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    contour_path = os.path.join(output_dir, f"{base}_contour.png")
    binary_path = os.path.join(output_dir, f"{base}_binary.png")

    cv2.imwrite(contour_path, contour_image)
    cv2.imwrite(binary_path, sparse_mask)

    return contour_path, binary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient guided contour extraction")
    parser.add_argument("image", help="Path to the input grayscale image")
    parser.add_argument(
        "--output",
        default="data/output",
        help="Directory where the contour and binary images will be stored",
    )
    parser.add_argument("--orientations", type=int, default=8, help="Number of filter orientations")
    parser.add_argument("--sigma-major", type=float, default=3.0, help="Major-axis sigma for anisotropic Gaussian")
    parser.add_argument("--sigma-minor", type=float, default=1.5, help="Minor-axis sigma for anisotropic Gaussian")
    parser.add_argument("--window", type=int, default=21, help="Adaptive threshold window size")
    parser.add_argument("--k", type=float, default=0.7, help="Adaptive threshold balancing term")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=5e-4,
        help="Minimum kept component area as a fraction of the image",
    )
    parser.add_argument("--eps", type=float, default=1e-2, help="Alpha selection tolerance for sparse contour extractor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_image(
        args.image,
        args.output,
        num_orientations=args.orientations,
        sigma_major=args.sigma_major,
        sigma_minor=args.sigma_minor,
        adaptive_window=args.window,
        adaptive_k=args.k,
        min_area_ratio=args.min_area_ratio,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
