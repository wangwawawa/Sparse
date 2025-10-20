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
    """Suppress non-maximum responses along the gradient direction via interpolation."""

    if magnitude.dtype != np.float32:
        mag = magnitude.astype(np.float32)
    else:
        mag = magnitude

    # Ensure directions are wrapped to [0, pi)
    theta = np.mod(direction, np.pi).astype(np.float32)

    rows, cols = mag.shape
    yy, xx = np.indices((rows, cols), dtype=np.float32)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    pos_x = xx + cos_theta
    pos_y = yy + sin_theta
    neg_x = xx - cos_theta
    neg_y = yy - sin_theta

    mag_float = mag.astype(np.float32)
    pos_vals = cv2.remap(
        mag_float,
        pos_x,
        pos_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    neg_vals = cv2.remap(
        mag_float,
        neg_x,
        neg_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    suppressed = np.zeros_like(mag_float)
    keep_mask = (mag_float >= pos_vals) & (mag_float >= neg_vals)
    suppressed[keep_mask] = mag_float[keep_mask]

    suppressed[[0, -1], :] = 0
    suppressed[:, [0, -1]] = 0
    return suppressed


def histogram_adaptive_threshold(
    magnitude: np.ndarray, s: float = 0.8, tau: float = 0.65
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute strong/weak edge maps using histogram-based thresholds."""

    mag = magnitude.astype(np.float32)
    mag = np.clip(mag, a_min=0.0, a_max=None)

    mag_min = float(mag.min())
    mag_max = float(mag.max())
    if mag_max > mag_min:
        mag_norm = (mag - mag_min) / (mag_max - mag_min)
    else:
        mag_norm = np.zeros_like(mag)

    hist, _ = np.histogram((mag_norm * 255).astype(np.uint8), bins=256, range=(0, 255))
    cumulative = np.cumsum(hist)
    total = mag_norm.size
    threshold_count = s * total
    high_index = int(np.searchsorted(cumulative, threshold_count))
    high_index = min(max(high_index, 0), 255)
    high_threshold = min(high_index / 255.0, 1.0)
    low_threshold = tau * high_threshold

    strong_edges = np.zeros_like(mag_norm, dtype=np.uint8)
    strong_edges[mag_norm >= high_threshold] = 255

    weak_edges = np.zeros_like(mag_norm, dtype=np.uint8)
    mask = (mag_norm >= low_threshold) & (mag_norm < high_threshold)
    weak_edges[mask] = 255

    return mag_norm, strong_edges, weak_edges


def connect_weak_edges(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Connect weak edges to strong edges within an 8-neighborhood."""

    kernel = np.ones((3, 3), dtype=np.uint8)
    connected = strong.copy()
    remaining = weak.copy()

    while True:
        expanded = cv2.dilate(connected, kernel, iterations=1)
        attach = cv2.bitwise_and(remaining, expanded)
        if not np.any(attach):
            break
        connected = cv2.bitwise_or(connected, attach)
        remaining[attach > 0] = 0

    return connected


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
    threshold_s: float = 0.8,
    threshold_tau: float = 0.65,
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

    esm_norm, strong_edges, weak_edges = histogram_adaptive_threshold(
        suppressed, s=threshold_s, tau=threshold_tau
    )
    edges = connect_weak_edges(strong_edges, weak_edges)

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
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_magnitude.png"),
        cv2.normalize(gradients.magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_nms.png"),
        cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_esm.png"),
        (esm_norm * 255).astype(np.uint8),
    )
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
    parser.add_argument("--s", type=float, default=0.8, help="Histogram proportion for high threshold")
    parser.add_argument("--tau", type=float, default=0.65, help="Low-to-high threshold ratio")
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
        threshold_s=args.s,
        threshold_tau=args.tau,
        min_area_ratio=args.min_area_ratio,
    )


if __name__ == "__main__":
    main()
