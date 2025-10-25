"""Texture-aware alpha shape contour extraction without external dependencies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Image = List[List[int]]
FloatImage = List[List[float]]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def image_shape(image: Image | FloatImage) -> Tuple[int, int]:
    return len(image), len(image[0]) if image else 0


def zeros_like(image: FloatImage, value: float = 0.0) -> FloatImage:
    h, w = image_shape(image)
    return [[value for _ in range(w)] for _ in range(h)]


def to_float_image(image: Image) -> FloatImage:
    if not image:
        return []
    max_value = max(max(row) for row in image) or 1
    scale = 1.0 / max_value
    return [[pixel * scale for pixel in row] for row in image]


def reflect_index(idx: int, size: int) -> int:
    if size <= 1:
        return 0
    while idx < 0 or idx >= size:
        if idx < 0:
            idx = -idx - 1
        if idx >= size:
            idx = (2 * size - idx) - 1
    return idx


def convolve2d(image: FloatImage, kernel: FloatImage) -> FloatImage:
    kh = len(kernel)
    kw = len(kernel[0]) if kh else 0
    pad_y = kh // 2
    pad_x = kw // 2
    height, width = image_shape(image)
    output = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(kh):
                iy = reflect_index(y + ky - pad_y, height)
                row = image[iy]
                kernel_row = kernel[ky]
                for kx in range(kw):
                    ix = reflect_index(x + kx - pad_x, width)
                    acc += row[ix] * kernel_row[kx]
            output[y][x] = acc
    return output


def convolve_separable(image: FloatImage, kernel: List[float], axis: int) -> FloatImage:
    height, width = image_shape(image)
    radius = len(kernel) // 2
    output = [[0.0 for _ in range(width)] for _ in range(height)]
    if axis == 0:
        for y in range(height):
            for x in range(width):
                acc = 0.0
                for k, weight in enumerate(kernel):
                    offset = k - radius
                    iy = reflect_index(y + offset, height)
                    acc += image[iy][x] * weight
                output[y][x] = acc
    else:
        for y in range(height):
            for x in range(width):
                acc = 0.0
                for k, weight in enumerate(kernel):
                    offset = k - radius
                    ix = reflect_index(x + offset, width)
                    acc += image[y][ix] * weight
                output[y][x] = acc
    return output


def gaussian_kernel1d(sigma: float) -> List[float]:
    radius = max(1, int(round(3 * sigma)))
    kernel: List[float] = []
    norm = 0.0
    for x in range(-radius, radius + 1):
        value = math.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel.append(value)
        norm += value
    if norm == 0.0:
        return [1.0]
    return [value / norm for value in kernel]


def gaussian_blur(image: FloatImage, sigma: float) -> FloatImage:
    kernel = gaussian_kernel1d(sigma)
    temp = convolve_separable(image, kernel, axis=0)
    return convolve_separable(temp, kernel, axis=1)


def downsample_image(image: FloatImage, factor: int) -> FloatImage:
    height, width = image_shape(image)
    if factor <= 1 or height == 0 or width == 0:
        return [row[:] for row in image]
    new_height = max(1, (height + factor - 1) // factor)
    new_width = max(1, (width + factor - 1) // factor)
    result = [[0.0 for _ in range(new_width)] for _ in range(new_height)]
    for y in range(new_height):
        for x in range(new_width):
            acc = 0.0
            count = 0
            for dy in range(factor):
                src_y = y * factor + dy
                if src_y >= height:
                    break
                for dx in range(factor):
                    src_x = x * factor + dx
                    if src_x >= width:
                        break
                    acc += image[src_y][src_x]
                    count += 1
            result[y][x] = acc / max(1, count)
    return result


def upsample_image(image: FloatImage, factor: int, target_shape: Tuple[int, int]) -> FloatImage:
    target_height, target_width = target_shape
    if factor <= 1:
        return [row[:target_width] for row in image[:target_height]]
    height, width = image_shape(image)
    result = [[0.0 for _ in range(target_width)] for _ in range(target_height)]
    for y in range(target_height):
        src_y = min(height - 1, y // factor)
        row = image[src_y]
        for x in range(target_width):
            src_x = min(width - 1, x // factor)
            result[y][x] = row[src_x]
    return result


def elementwise_pow(image: FloatImage, exponent: float) -> FloatImage:
    return [[value ** exponent for value in row] for row in image]


def elementwise_abs(image: FloatImage) -> FloatImage:
    return [[abs(value) for value in row] for row in image]


def normalise_image(image: FloatImage) -> FloatImage:
    max_value = max((value for row in image for value in row), default=0.0)
    if max_value <= 0.0:
        return [row[:] for row in image]
    return [[value / max_value for value in row] for row in image]


def variance_of(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    pos = q * (len(sorted_vals) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    weight = pos - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def downsample(points: List[Tuple[int, int]], quota: int) -> List[Tuple[int, int]]:
    if quota <= 0 or len(points) <= quota:
        return points
    stride = max(1, len(points) // quota)
    return points[::stride]


def magnitude(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Module ① – Texture feature extraction
# ---------------------------------------------------------------------------


@dataclass
class TextureFeatures:
    gradient_magnitude: FloatImage
    gradient_orientation: FloatImage
    gabor_response: FloatImage
    gabor_orientation: FloatImage
    texture_saliency: FloatImage


class TextureFeatureExtractor:
    """Extract gradient and multi-scale Gabor descriptors for each pixel."""

    def __init__(
        self,
        orientations: int = 4,
        scales: int = 1,
        base_sigma: float = 1.5,
        gamma: float = 0.5,
        kernel_size: int = 5,
    ) -> None:
        self.orientations = orientations
        self.scales = scales
        self.base_sigma = base_sigma
        self.gamma = gamma
        self.kernel_size = max(3, kernel_size | 1)

    def _gabor_kernel(
        self, sigma: float, theta: float, lambd: float, gamma: float, psi: float
    ) -> FloatImage:
        half = (self.kernel_size - 1) / 2.0
        kernel: FloatImage = []
        for y in range(self.kernel_size):
            row: List[float] = []
            for x in range(self.kernel_size):
                y_coord = y - half
                x_coord = x - half
                x_theta = x_coord * math.cos(theta) + y_coord * math.sin(theta)
                y_theta = -x_coord * math.sin(theta) + y_coord * math.cos(theta)
                gaussian = math.exp(
                    -0.5
                    * (
                        (x_theta ** 2) / (sigma ** 2)
                        + (gamma ** 2) * (y_theta ** 2) / (sigma ** 2)
                    )
                )
                sinusoid = math.cos((2.0 * math.pi * x_theta / lambd) + psi)
                row.append(gaussian * sinusoid)
            kernel.append(row)
        return kernel

    def extract(self, image: Image | FloatImage) -> TextureFeatures:
        if not image or not image[0]:
            raise ValueError("TextureFeatureExtractor expects a non-empty image")

        if isinstance(image[0][0], float):  # type: ignore[index]
            image_float: FloatImage = [row[:] for row in image]  # type: ignore[assignment]
        else:
            image_float = to_float_image(image)  # type: ignore[arg-type]

        height, width = image_shape(image_float)
        factor = 2 if max(height, width) > 256 else 1
        proc_image = downsample_image(image_float, factor)
        proc_height, proc_width = image_shape(proc_image)

        sobel_x = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        sobel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        grad_x = convolve2d(proc_image, sobel_x)
        grad_y = convolve2d(proc_image, sobel_y)

        grad_mag = [[0.0 for _ in range(width)] for _ in range(height)]
        grad_ori = [[0.0 for _ in range(width)] for _ in range(height)]
        for y in range(proc_height):
            for x in range(proc_width):
                gx = grad_x[y][x] / 8.0
                gy = grad_y[y][x] / 8.0
                grad_mag[y][x] = magnitude(gx, gy)
                grad_ori[y][x] = math.atan2(gy, gx)

        grad_mag_norm_proc = normalise_image([row[:proc_width] for row in grad_mag[:proc_height]])

        gabor_max_proc = zeros_like(proc_image)
        gabor_orientation_index_proc = zeros_like(proc_image)

        lambda_base = max(4.0, self.kernel_size / 4.0)
        idx = 0
        for scale in range(self.scales):
            sigma = self.base_sigma * (1.5 ** scale)
            lambd = lambda_base * (1.5 ** scale)
            for orientation_idx in range(self.orientations):
                theta = math.pi * orientation_idx / self.orientations
                kernel = self._gabor_kernel(
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd,
                    gamma=self.gamma,
                    psi=0.0,
                )
                response = convolve2d(proc_image, kernel)
                response_abs = elementwise_abs(response)
                for y in range(proc_height):
                    for x in range(proc_width):
                        value = response_abs[y][x]
                        if value > gabor_max_proc[y][x]:
                            gabor_max_proc[y][x] = value
                            gabor_orientation_index_proc[y][x] = float(idx)
                idx += 1

        gabor_norm_proc = normalise_image(gabor_max_proc)

        if self.orientations * self.scales > 0:
            gabor_orientation_proc = zeros_like(proc_image)
            for y in range(proc_height):
                for x in range(proc_width):
                    orientation_index = int(gabor_orientation_index_proc[y][x]) % self.orientations
                    gabor_orientation_proc[y][x] = orientation_index * (math.pi / self.orientations)
        else:
            gabor_orientation_proc = zeros_like(proc_image)

        blurred = gaussian_blur(proc_image, sigma=1.0)
        squared_blur = gaussian_blur(elementwise_pow(proc_image, 2.0), sigma=1.0)
        texture_variance_proc = zeros_like(proc_image)
        for y in range(proc_height):
            for x in range(proc_width):
                variance = max(squared_blur[y][x] - blurred[y][x] ** 2, 0.0)
                texture_variance_proc[y][x] = variance
        texture_saliency_proc = normalise_image(texture_variance_proc)

        if factor > 1:
            grad_mag_norm = upsample_image(grad_mag_norm_proc, factor, (height, width))
            grad_ori = upsample_image([row[:proc_width] for row in grad_ori[:proc_height]], factor, (height, width))
            gabor_norm = upsample_image(gabor_norm_proc, factor, (height, width))
            gabor_orientation = upsample_image(gabor_orientation_proc, factor, (height, width))
            texture_saliency = upsample_image(texture_saliency_proc, factor, (height, width))
        else:
            grad_mag_norm = grad_mag_norm_proc
            gabor_norm = gabor_norm_proc
            gabor_orientation = gabor_orientation_proc
            texture_saliency = texture_saliency_proc

        return TextureFeatures(
            gradient_magnitude=grad_mag_norm,
            gradient_orientation=grad_ori,
            gabor_response=gabor_norm,
            gabor_orientation=gabor_orientation,
            texture_saliency=texture_saliency,
        )


# ---------------------------------------------------------------------------
# Module ② – Point sampling & feature mapping
# ---------------------------------------------------------------------------


@dataclass
class SampledPointSet:
    coordinates: List[Tuple[float, float]]
    gradient_magnitude: List[float]
    gradient_orientation: List[float]
    gabor_response: List[float]
    gabor_orientation: List[float]
    image_shape: Tuple[int, int]

    def feature_triplet(self, indices: Tuple[int, int, int]) -> Dict[str, List[float]]:
        i, j, k = indices
        return {
            "gradient": [self.gradient_magnitude[i], self.gradient_magnitude[j], self.gradient_magnitude[k]],
            "texture": [self.gabor_response[i], self.gabor_response[j], self.gabor_response[k]],
            "grad_orientation": [self.gradient_orientation[i], self.gradient_orientation[j], self.gradient_orientation[k]],
            "tex_orientation": [self.gabor_orientation[i], self.gabor_orientation[j], self.gabor_orientation[k]],
        }


class PointSampler:
    """Selects representative points guided by edges and texture saliency."""

    def __init__(
        self,
        max_points: int = 1500,
        edge_ratio: float = 0.6,
        texture_ratio: float = 0.4,
        random_state: Optional[int] = None,
    ) -> None:
        if max_points < 3:
            raise ValueError("PointSampler requires at least 3 points")
        self.max_points = max_points
        self.edge_ratio = edge_ratio
        self.texture_ratio = texture_ratio
        self.rng = random.Random(random_state)

    def sample(self, image: Image | FloatImage, features: TextureFeatures) -> SampledPointSet:
        height, width = image_shape(image)

        grad_values = [value for row in features.gradient_magnitude for value in row]
        if not grad_values:
            raise ValueError("Empty gradient magnitude map")
        edge_threshold = max(quantile(grad_values, 0.75), 0.1)
        edge_coords: List[Tuple[int, int]] = [
            (y, x)
            for y in range(height)
            for x in range(width)
            if features.gradient_magnitude[y][x] >= edge_threshold
        ]
        if not edge_coords:
            edge_coords = [
                (y, x)
                for y in range(height)
                for x in range(width)
                if features.gradient_magnitude[y][x] > 0.2
            ]

        edge_quota = max(3, int(self.max_points * self.edge_ratio))
        edge_coords = downsample(edge_coords, edge_quota)

        texture_map = [[max(features.gabor_response[y][x], features.texture_saliency[y][x]) for x in range(width)] for y in range(height)]
        texture_values = [value for row in texture_map for value in row]
        if not texture_values:
            raise ValueError("Empty texture map")

        texture_quota = max(3, int(self.max_points * self.texture_ratio))
        threshold = quantile(texture_values, 1.0 - min(1.0, texture_quota / max(1, len(texture_values))))
        texture_coords = [
            (y, x)
            for y in range(height)
            for x in range(width)
            if texture_map[y][x] >= threshold
        ]
        if not texture_coords:
            texture_coords = [
                (y, x)
                for y in range(height)
                for x in range(width)
                if texture_map[y][x] > 0.0
            ]
        texture_coords = downsample(texture_coords, texture_quota)

        combined_set = {coord for coord in edge_coords}
        combined_set.update(texture_coords)

        if len(combined_set) < 3:
            grid_size = max(3, int(math.sqrt(self.max_points)))
            ys = [int(round(y)) for y in [i * (height - 1) / max(1, grid_size - 1) for i in range(grid_size)]]
            xs = [int(round(x)) for x in [i * (width - 1) / max(1, grid_size - 1) for i in range(grid_size)]]
            for y in ys:
                for x in xs:
                    combined_set.add((y, x))

        coords = list(combined_set)
        if len(coords) > self.max_points:
            coords = self.rng.sample(coords, self.max_points)

        coords.sort()
        points = [(float(x), float(y)) for y, x in coords]

        grad_mag = [features.gradient_magnitude[y][x] for y, x in coords]
        grad_ori = [features.gradient_orientation[y][x] for y, x in coords]
        gabor_resp = [features.gabor_response[y][x] for y, x in coords]
        gabor_ori = [features.gabor_orientation[y][x] for y, x in coords]

        return SampledPointSet(
            coordinates=points,
            gradient_magnitude=grad_mag,
            gradient_orientation=grad_ori,
            gabor_response=gabor_resp,
            gabor_orientation=gabor_ori,
            image_shape=(height, width),
        )


# ---------------------------------------------------------------------------
# Module ③ – Texture-aware α-shape construction
# ---------------------------------------------------------------------------


def circumcircle(triangle: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
    (ax, ay), (bx, by), (cx, cy) = triangle
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return ((float("inf"), float("inf")), float("inf"))
    ax2_ay2 = ax * ax + ay * ay
    bx2_by2 = bx * bx + by * by
    cx2_cy2 = cx * cx + cy * cy
    ux = (
        ax2_ay2 * (by - cy)
        + bx2_by2 * (cy - ay)
        + cx2_cy2 * (ay - by)
    ) / d
    uy = (
        ax2_ay2 * (cx - bx)
        + bx2_by2 * (ax - cx)
        + cx2_cy2 * (bx - ax)
    ) / d
    radius = magnitude(ux - ax, uy - ay)
    return ((ux, uy), radius)


def delaunay_triangulation(points: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    n_points = len(points)
    if n_points < 3:
        raise ValueError("At least three points are required for triangulation")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    if delta_max == 0.0:
        delta_max = 1.0
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0

    super_pts = [
        (mid_x - 20 * delta_max, mid_y - delta_max),
        (mid_x, mid_y + 20 * delta_max),
        (mid_x + 20 * delta_max, mid_y - delta_max),
    ]
    pts_ext = points + super_pts
    super_indices = (n_points, n_points + 1, n_points + 2)

    triangles: List[Tuple[int, int, int]] = [super_indices]
    circumcircles: Dict[Tuple[int, int, int], Tuple[Tuple[float, float], float]] = {
        super_indices: circumcircle([pts_ext[i] for i in super_indices])
    }

    for idx in range(n_points):
        point = pts_ext[idx]
        bad_triangles: List[Tuple[int, int, int]] = []
        for tri in triangles[:]:
            center, radius = circumcircles[tri]
            if not math.isfinite(radius):
                continue
            if magnitude(point[0] - center[0], point[1] - center[1]) <= radius + 1e-9:
                bad_triangles.append(tri)

        polygon: List[Tuple[int, int]] = []
        for tri in bad_triangles:
            triangles.remove(tri)
            del circumcircles[tri]
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                edge_sorted = tuple(sorted(edge))
                if edge_sorted in polygon:
                    polygon.remove(edge_sorted)
                else:
                    polygon.append(edge_sorted)

        for edge in polygon:
            new_tri = (edge[0], edge[1], idx)
            triangle_points = [pts_ext[i] for i in new_tri]
            center, radius = circumcircle(triangle_points)
            if not math.isfinite(radius):
                continue
            triangles.append(new_tri)
            circumcircles[new_tri] = (center, radius)

    result: List[Tuple[int, int, int]] = []
    for tri in triangles:
        if any(v >= n_points for v in tri):
            continue
        result.append(tri)

    if not result:
        raise RuntimeError("Delaunay triangulation failed")
    return result


def triangle_circumradius(points: List[Tuple[float, float]], simplices: List[Tuple[int, int, int]]) -> List[float]:
    radii: List[float] = []
    for a, b, c in simplices:
        ax, ay = points[a]
        bx, by = points[b]
        cx, cy = points[c]
        side_a = magnitude(bx - cx, by - cy)
        side_b = magnitude(cx - ax, cy - ay)
        side_c = magnitude(ax - bx, ay - by)
        s = (side_a + side_b + side_c) / 2.0
        area_sq = max(s * (s - side_a) * (s - side_b) * (s - side_c), 1e-12)
        area = math.sqrt(area_sq)
        radius = (side_a * side_b * side_c) / (4.0 * area)
        radii.append(radius)
    return radii


def triangle_area(points: List[Tuple[float, float]], simplices: List[Tuple[int, int, int]]) -> List[float]:
    areas: List[float] = []
    for a, b, c in simplices:
        ax, ay = points[a]
        bx, by = points[b]
        cx, cy = points[c]
        area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
        areas.append(area)
    return areas


def triangle_variance(values: List[List[float]]) -> List[float]:
    return [variance_of(triple) for triple in values]


def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_polyline(canvas: List[List[int]], points: List[Tuple[float, float]], value: int, thickness: int = 1) -> None:
    if len(points) < 2:
        return
    height = len(canvas)
    width = len(canvas[0]) if height else 0
    radius = max(0, thickness // 2)

    def stamp(px: int, py: int) -> None:
        if radius == 0:
            if 0 <= px < width and 0 <= py < height:
                canvas[py][px] = value
            return
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                sx = px + dx
                sy = py + dy
                if 0 <= sx < width and 0 <= sy < height:
                    canvas[sy][sx] = value

    rounded = [(int(round(x)), int(round(y))) for x, y in points]
    for (x0, y0), (x1, y1) in zip(rounded[:-1], rounded[1:]):
        for px, py in bresenham_line((x0, y0), (x1, y1)):
            stamp(px, py)


@dataclass
class TextureAwareAlphaShapeResult:
    alpha: float
    points: List[Tuple[float, float]]
    simplices: List[Tuple[int, int, int]]
    radii: List[float]
    weights: List[float]
    mask: List[bool]
    areas: List[float]
    image_shape: Tuple[int, int]

    def selected_indices(self) -> List[int]:
        return [idx for idx, flag in enumerate(self.mask) if flag]

    def selected_triangles(self) -> List[Tuple[int, int, int]]:
        indices = self.selected_indices()
        return [self.simplices[idx] for idx in indices]

    def selected_weights(self) -> List[float]:
        return [self.weights[idx] for idx in self.selected_indices()]

    def selected_radii(self) -> List[float]:
        return [self.radii[idx] for idx in self.selected_indices()]

    def boundary_edges(self) -> List[Tuple[int, int]]:
        edges: Dict[Tuple[int, int], int] = {}
        for tri in self.selected_triangles():
            tri_edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for u, v in tri_edges:
                key = (min(u, v), max(u, v))
                edges[key] = edges.get(key, 0) + 1
        return [edge for edge, count in edges.items() if count == 1]

    def connected_components(self) -> int:
        edges = self.boundary_edges()
        if not edges:
            return 0
        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        visited = set()
        components = 0
        for node in adjacency:
            if node in visited:
                continue
            components += 1
            stack = [node]
            visited.add(node)
            while stack:
                current = stack.pop()
                for nxt in adjacency.get(current, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
        return components

    def total_area(self) -> float:
        return sum(self.areas[idx] for idx in self.selected_indices())

    def perimeter(self) -> float:
        edges = self.boundary_edges()
        if not edges:
            return 0.0
        length = 0.0
        for u, v in edges:
            x0, y0 = self.points[u]
            x1, y1 = self.points[v]
            length += magnitude(x0 - x1, y0 - y1)
        return length

    def texture_consistency(self) -> float:
        weights = self.selected_weights()
        if not weights:
            return 0.0
        return sum(weights) / len(weights)

    def contours(self) -> List[List[Tuple[float, float]]]:
        edges = self.boundary_edges()
        if not edges:
            return []
        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        visited_edges = set()
        contours: List[List[Tuple[float, float]]] = []
        for start_u, neighbors in adjacency.items():
            for start_v in neighbors:
                edge_key = (min(start_u, start_v), max(start_u, start_v))
                if edge_key in visited_edges:
                    continue
                contour: List[Tuple[float, float]] = [self.points[start_u], self.points[start_v]]
                visited_edges.add(edge_key)
                u, v = start_u, start_v
                while True:
                    next_candidates = [n for n in adjacency.get(v, []) if n != u]
                    if not next_candidates:
                        break
                    next_node = next_candidates[0]
                    edge_key = (min(v, next_node), max(v, next_node))
                    if edge_key in visited_edges:
                        break
                    contour.append(self.points[next_node])
                    visited_edges.add(edge_key)
                    u, v = v, next_node
                    if v == start_u:
                        break
                contours.append(contour)
        return contours

    def render_mask(self, thickness: int = 2) -> List[List[int]]:
        height, width = self.image_shape
        canvas = [[0 for _ in range(width)] for _ in range(height)]
        for contour in self.contours():
            draw_polyline(canvas, contour, value=1, thickness=thickness)
        return canvas


class TextureAwareAlphaShape:
    """Constructs α-shapes with texture-based weighting."""

    def __init__(self, beta: float = 4.0) -> None:
        self.beta = beta

    def _texture_weight_for_triangle(
        self, samples: SampledPointSet, simplex: Tuple[int, int, int]
    ) -> float:
        features = samples.feature_triplet(simplex)
        grad = features["gradient"]
        tex = features["texture"]
        grad_ori = features["grad_orientation"]
        tex_ori = features["tex_orientation"]

        theta_cos = [math.cos(angle) for angle in grad_ori]
        theta_sin = [math.sin(angle) for angle in grad_ori]
        phi_cos = [math.cos(angle) for angle in tex_ori]
        phi_sin = [math.sin(angle) for angle in tex_ori]

        texture_diff = (
            variance_of(grad)
            + variance_of(tex)
            + variance_of(theta_cos)
            + variance_of(theta_sin)
            + variance_of(phi_cos)
            + variance_of(phi_sin)
        )
        return math.exp(-self.beta * texture_diff)

    def build(self, samples: SampledPointSet, alpha: float) -> TextureAwareAlphaShapeResult:
        points = samples.coordinates
        if len(points) < 3:
            raise ValueError("At least three sample points are required")

        simplices = delaunay_triangulation(points)
        radii = triangle_circumradius(points, simplices)
        areas = triangle_area(points, simplices)
        weights = [self._texture_weight_for_triangle(samples, simplex) for simplex in simplices]
        mask = [radius <= alpha * weight for radius, weight in zip(radii, weights)]

        return TextureAwareAlphaShapeResult(
            alpha=alpha,
            points=points,
            simplices=simplices,
            radii=radii,
            weights=weights,
            mask=mask,
            areas=areas,
            image_shape=samples.image_shape,
        )


# ---------------------------------------------------------------------------
# Module ④ – Stability analysis & α selection
# ---------------------------------------------------------------------------


@dataclass
class StabilityRecord:
    alpha: float
    area: float
    components: int
    perimeter: float
    texture_consistency: float
    csi_geometry: float
    csi_texture: float
    csi: float


@dataclass
class StabilityAnalysis:
    records: List[StabilityRecord]
    best_record: StabilityRecord
    best_result: TextureAwareAlphaShapeResult


def finite_difference(values: List[float], xs: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    derivatives: List[float] = []
    for i in range(n):
        if i == 0:
            dv = values[1] - values[0]
            dx = xs[1] - xs[0]
        elif i == n - 1:
            dv = values[-1] - values[-2]
            dx = xs[-1] - xs[-2]
        else:
            dv = values[i + 1] - values[i - 1]
            dx = xs[i + 1] - xs[i - 1]
        if dx == 0.0:
            derivatives.append(0.0)
        else:
            derivatives.append(dv / dx)
    return derivatives


class AlphaStabilityAnalyzer:
    """Evaluates geometry/texture stability curves and selects α*."""

    def __init__(self, lam: float = 0.7) -> None:
        self.lam = lam

    def evaluate(
        self, results: Sequence[TextureAwareAlphaShapeResult]
    ) -> StabilityAnalysis:
        if not results:
            raise ValueError("No α-shape results provided")

        ordered = sorted(results, key=lambda res: res.alpha)
        alphas = [res.alpha for res in ordered]
        areas = [res.total_area() for res in ordered]
        components = [res.connected_components() for res in ordered]
        perimeters = [res.perimeter() for res in ordered]
        texture_consistency = [res.texture_consistency() for res in ordered]

        area_derivative = finite_difference(areas, alphas)
        texture_derivative = finite_difference(texture_consistency, alphas)

        csi_geometry: List[float] = []
        for area, deriv in zip(areas, area_derivative):
            denom = area if area > 1e-8 else 1e-8
            csi_geometry.append(math.exp(-abs(deriv / denom)))

        csi_texture = [math.exp(-abs(deriv)) for deriv in texture_derivative]
        csi = [self.lam * cg + (1.0 - self.lam) * ct for cg, ct in zip(csi_geometry, csi_texture)]

        records: List[StabilityRecord] = []
        best_idx = max(range(len(csi)), key=lambda idx: csi[idx])
        for idx, res in enumerate(ordered):
            record = StabilityRecord(
                alpha=alphas[idx],
                area=areas[idx],
                components=components[idx],
                perimeter=perimeters[idx],
                texture_consistency=texture_consistency[idx],
                csi_geometry=csi_geometry[idx],
                csi_texture=csi_texture[idx],
                csi=csi[idx],
            )
            records.append(record)

        return StabilityAnalysis(
            records=records,
            best_record=records[best_idx],
            best_result=ordered[best_idx],
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_texture_alpha_shape(
    image: Image,
    alpha_values: Sequence[float],
    beta: float = 4.0,
    lam: float = 0.7,
    max_points: int = 1500,
) -> StabilityAnalysis:
    extractor = TextureFeatureExtractor()
    features = extractor.extract(image)

    sampler = PointSampler(max_points=max_points)
    samples = sampler.sample(image, features)

    tas = TextureAwareAlphaShape(beta=beta)
    results = [tas.build(samples, alpha=float(alpha)) for alpha in alpha_values]

    analyzer = AlphaStabilityAnalyzer(lam=lam)
    return analyzer.evaluate(results)

