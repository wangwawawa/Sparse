"""Gradient-based contour extraction using anisotropic Gaussian derivatives."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def anisotropic_diffusion(
    image: np.ndarray,
    num_iter: int = 15,
    kappa: float = 20.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion to smooth the image."""

    if image.ndim != 2:
        raise ValueError("Anisotropic diffusion expects a single channel image")

    if not (0.0 < gamma <= 0.25):
        raise ValueError("gamma must be in the range (0, 0.25]")

    diffused = image.astype(np.float32).copy()

    for _ in range(num_iter):
        nabla_north = np.zeros_like(diffused)
        nabla_south = np.zeros_like(diffused)
        nabla_east = np.zeros_like(diffused)
        nabla_west = np.zeros_like(diffused)

        nabla_north[1:, :] = diffused[1:, :] - diffused[:-1, :]
        nabla_south[:-1, :] = diffused[:-1, :] - diffused[1:, :]
        nabla_east[:, :-1] = diffused[:, :-1] - diffused[:, 1:]
        nabla_west[:, 1:] = diffused[:, 1:] - diffused[:, :-1]

        if option == 1:
            c_n = np.exp(-(nabla_north / kappa) ** 2)
            c_s = np.exp(-(nabla_south / kappa) ** 2)
            c_e = np.exp(-(nabla_east / kappa) ** 2)
            c_w = np.exp(-(nabla_west / kappa) ** 2)
        elif option == 2:
            c_n = 1.0 / (1.0 + (nabla_north / kappa) ** 2)
            c_s = 1.0 / (1.0 + (nabla_south / kappa) ** 2)
            c_e = 1.0 / (1.0 + (nabla_east / kappa) ** 2)
            c_w = 1.0 / (1.0 + (nabla_west / kappa) ** 2)
        else:
            raise ValueError("option must be either 1 or 2")

        diffused += gamma * (
            c_n * nabla_north
            + c_s * nabla_south
            + c_e * nabla_east
            + c_w * nabla_west
        )

    diffused = np.clip(diffused, 0.0, 1.0)
    return diffused


def anisotropic_gaussian_derivative_kernel(
    order: int,
    sigma_major: float,
    sigma_minor: float,
    theta: float,
    size: int | None = None,
) -> np.ndarray:
    """Create an anisotropic Gaussian derivative kernel rotated by ``theta``.
    Parameters
    ----------
    order:
        Order of derivative along the major axis (1 or 2).
    sigma_major:
        Standard deviation along the major axis of the Gaussian.
    sigma_minor:
        Standard deviation along the minor axis of the Gaussian.
    theta:
        Orientation angle in radians.
    size:
        Optional kernel size. If ``None`` a size large enough to cover three
        standard deviations of the anisotropic Gaussian is used.
    """
    if order not in (1, 2):
        raise ValueError("Only first and second order derivatives are supported")

    sigma_max = max(sigma_major, sigma_minor)
    if size is None:
        size = int(np.ceil(6 * sigma_max)) | 1  # ensure odd size
    half = size // 2

    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta

    gaussian = np.exp(
        -0.5
        * ((x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2)
    )

    if order == 1:
        kernel = -(x_rot / (sigma_major**2)) * gaussian
    else:  # order == 2
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
        raise ValueError("Gradient computation expects a single-channel image")

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

    magnitude = np.abs(second_selected)

    return GradientResponses(
        magnitude=magnitude.astype(np.float32),
        direction=direction.astype(np.float32),
        first_order=first_selected.astype(np.float32),
        second_order=second_selected.astype(np.float32),
    )


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Suppress non-maximum responses along the gradient direction."""
    if magnitude.dtype != np.float32:
        mag = magnitude.astype(np.float32)
    else:
        mag = magnitude

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


def clean_edge_map(edge_map: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def extract_inner_outer_contours(
    edge_map: np.ndarray,
    image_shape: Tuple[int, int],
    min_area_ratio: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract both external and internal contours from the edge map."""

    h, w = image_shape
    edge_u8 = (edge_map > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edge_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    filled = dilated.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled = cv2.bitwise_not(filled)
    region = cv2.bitwise_or(dilated, filled)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    min_area = max(int(min_area_ratio * h * w), 1)

    region_filtered = np.zeros_like(region)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            region_filtered[labels == label] = 255

    if np.count_nonzero(region_filtered) == 0:
        region_filtered = region

    contours, hierarchy = cv2.findContours(
        region_filtered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    object_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask = np.zeros((h, w), dtype=np.uint8)
    inner_boundary = np.zeros((h, w), dtype=np.uint8)
    outer_boundary = np.zeros((h, w), dtype=np.uint8)

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for idx, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue

            parent = hierarchy[idx][3]
            if parent == -1:
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
                cv2.drawContours(outer_boundary, [contour], -1, 255, thickness=1)
            else:
                cv2.drawContours(hole_mask, [contour], -1, 255, thickness=cv2.FILLED)
                cv2.drawContours(inner_boundary, [contour], -1, 255, thickness=1)
    else:
        object_mask = region_filtered.copy()
        outer_boundary = cv2.Canny(object_mask, 50, 150)

    outer_mask = cv2.subtract(object_mask, hole_mask)

    return inner_boundary, outer_boundary, outer_mask, region_filtered


def process_image(
    image_path: str,
    output_dir: str,
    num_orientations: int = 8,
    sigma_major: float = 3.0,
    sigma_minor: float = 1.5,
    threshold_window: int = 21,
    threshold_k: float = 0.7,
    min_area_ratio: float = 1e-3,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    gray_norm = cv2.normalize(
        gray.astype(np.float32), None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX
    )

    diffused = anisotropic_diffusion(gray_norm, num_iter=20, kappa=25.0, gamma=0.2, option=1)

    gradients = compute_multi_direction_gradients(
        diffused,
        num_orientations=num_orientations,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
    )

    suppressed = non_maximum_suppression(gradients.magnitude, gradients.direction)
    edges = adaptive_threshold(suppressed, window=threshold_window, k=threshold_k)
    edges = clean_edge_map(edges)

    inner_boundary, outer_boundary, object_mask, region = extract_inner_outer_contours(
        edges,
        image_shape=gray.shape,
        min_area_ratio=min_area_ratio,
    )

    overlay = image.copy()
    overlay[outer_boundary > 0] = (0, 0, 255)
    overlay[inner_boundary > 0] = (0, 255, 0)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_gray.png"), (gray_norm * 255).astype(np.uint8)
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_diffused.png"), (diffused * 255).astype(np.uint8)
    )
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_magnitude.png"), cv2.normalize(gradients.magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_nms.png"), cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), object_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_inner.png"), inner_boundary)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_outer.png"), outer_boundary)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract inner and outer contours using anisotropic gradients")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--output",
        default="data/output",
        help="Directory to store intermediate results",
    )
    parser.add_argument("--orientations", type=int, default=8, help="Number of filter orientations")
    parser.add_argument("--sigma-major", type=float, default=3.0, help="Gaussian sigma along the major axis")
    parser.add_argument("--sigma-minor", type=float, default=1.5, help="Gaussian sigma along the minor axis")
    parser.add_argument("--window", type=int, default=21, help="Adaptive threshold window size")
    parser.add_argument("--k", type=float, default=0.7, help="Adaptive threshold sensitivity factor")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=1e-3,
        help="Minimum area ratio used to discard small components",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_image(
        image_path=args.image,
        output_dir=args.output,
        num_orientations=args.orientations,
        sigma_major=args.sigma_major,
        sigma_minor=args.sigma_minor,
        threshold_window=args.window,
        threshold_k=args.k,
        min_area_ratio=args.min_area_ratio,
    )


if __name__ == "__main__":
    main()
