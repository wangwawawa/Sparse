"""Texture-aware alpha shape contour extraction.

This module implements the pipeline described in
``docs/texture_aware_alpha_shape.md``.  It contains the following stages:

1. ``TextureFeatureExtractor`` – compute gradient and multi-scale Gabor
   texture descriptors for every pixel in a grayscale image.
2. ``PointSampler`` – select representative pixels for the α-shape based on
   edge evidence and texture saliency while retaining their feature vectors.
3. ``TextureAwareAlphaShape`` – construct a texture-weighted α-shape by
   modifying the triangle acceptance test using the texture consistency term.
4. ``AlphaStabilityAnalyzer`` – scan across α values and pick the one that
   maximises the combined geometry/texture stability criterion.

The public API intentionally mirrors the design document so that each class
corresponds to one section of the specification.  The module is self-contained
and depends only on NumPy, OpenCV, and SciPy.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Module ① – Texture feature extraction
# ---------------------------------------------------------------------------


@dataclass
class TextureFeatures:
    """Container for per-pixel texture descriptors."""

    gradient_magnitude: np.ndarray
    gradient_orientation: np.ndarray
    gabor_response: np.ndarray
    gabor_orientation: np.ndarray
    texture_saliency: np.ndarray


class TextureFeatureExtractor:
    """Extracts gradient and multi-scale Gabor descriptors for each pixel."""

    def __init__(
        self,
        orientations: int = 8,
        scales: int = 3,
        base_sigma: float = 2.0,
        gamma: float = 0.5,
        kernel_size: int = 21,
    ) -> None:
        self.orientations = orientations
        self.scales = scales
        self.base_sigma = base_sigma
        self.gamma = gamma
        self.kernel_size = kernel_size

    def extract(self, image: np.ndarray) -> TextureFeatures:
        if image.ndim != 2:
            raise ValueError("TextureFeatureExtractor expects a grayscale image")

        image_float = image.astype(np.float32)
        if image_float.max() > 1.0:
            image_float /= 255.0

        grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_ori = np.arctan2(grad_y, grad_x)

        max_grad = float(grad_mag.max()) or 1.0
        grad_mag_norm = grad_mag / max_grad

        gabor_max = np.zeros_like(image_float)
        gabor_argmax = np.zeros_like(image_float)

        lambda_base = max(4.0, self.kernel_size / 4.0)
        idx = 0
        for scale in range(self.scales):
            sigma = self.base_sigma * (1.5 ** scale)
            lambd = lambda_base * (1.5 ** scale)
            for orientation_idx in range(self.orientations):
                theta = math.pi * orientation_idx / self.orientations
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    sigma,
                    theta,
                    lambd,
                    self.gamma,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
                response = cv2.filter2D(image_float, cv2.CV_32F, kernel)
                response = np.abs(response)

                update_mask = response > gabor_max
                gabor_max = np.where(update_mask, response, gabor_max)
                gabor_argmax = np.where(update_mask, idx, gabor_argmax)
                idx += 1

        max_response = float(gabor_max.max()) or 1.0
        gabor_norm = gabor_max / max_response

        # Convert the argmax indices back to orientation angles (0..2π)
        if self.orientations * self.scales > 0:
            orientation_indices = gabor_argmax % self.orientations
            gabor_orientation = orientation_indices * (math.pi / self.orientations)
        else:
            gabor_orientation = np.zeros_like(image_float)

        # Texture saliency via local variance (for sampling support)
        blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=1.0, sigmaY=1.0)
        squared_blur = cv2.GaussianBlur(image_float ** 2, (0, 0), sigmaX=1.0, sigmaY=1.0)
        texture_variance = np.maximum(squared_blur - blurred ** 2, 0.0)
        max_var = float(texture_variance.max()) or 1.0
        texture_saliency = texture_variance / max_var

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
    """Represents the sparse set of points used by the α-shape model."""

    coordinates: np.ndarray  # (N, 2) array of (x, y) image coordinates
    gradient_magnitude: np.ndarray
    gradient_orientation: np.ndarray
    gabor_response: np.ndarray
    gabor_orientation: np.ndarray

    image_shape: Tuple[int, int]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "gradient": self.gradient_magnitude,
            "orientation": self.gradient_orientation,
            "texture": self.gabor_response,
            "texture_orientation": self.gabor_orientation,
        }


class PointSampler:
    """Selects representative points guided by edges and texture saliency."""

    def __init__(
        self,
        max_points: int = 4000,
        edge_ratio: float = 0.6,
        texture_ratio: float = 0.4,
        random_state: Optional[int] = None,
    ) -> None:
        if max_points < 3:
            raise ValueError("PointSampler requires at least 3 points")
        self.max_points = max_points
        self.edge_ratio = edge_ratio
        self.texture_ratio = texture_ratio
        self.random_state = np.random.default_rng(random_state)

    def sample(self, image: np.ndarray, features: TextureFeatures) -> SampledPointSet:
        if image.ndim != 2:
            raise ValueError("PointSampler expects a grayscale image")

        h, w = image.shape
        image_u8 = image.astype(np.uint8)

        # Edge-driven sampling (Canny over gradient map)
        low = int(50)
        high = int(150)
        edge_map = cv2.Canny(image_u8, low, high) > 0
        edge_coords = np.column_stack(np.nonzero(edge_map))

        if edge_coords.size == 0:
            edge_coords = np.column_stack(np.nonzero(features.gradient_magnitude > 0.2))

        edge_quota = max(3, int(self.max_points * self.edge_ratio))
        edge_stride = max(1, len(edge_coords) // edge_quota)
        edge_coords = edge_coords[::edge_stride]

        # Texture saliency sampling (top-k variance or Gabor response)
        texture_map = np.maximum(features.gabor_response, features.texture_saliency)
        texture_flat = texture_map.ravel()
        if texture_flat.size == 0:
            raise ValueError("Empty texture map")

        texture_quota = max(3, int(self.max_points * self.texture_ratio))
        threshold = 0.0
        if texture_quota < texture_flat.size:
            threshold = float(np.quantile(texture_flat, 1.0 - texture_quota / texture_flat.size))

        tex_coords = np.column_stack(np.nonzero(texture_map >= threshold))
        if tex_coords.size == 0:
            tex_coords = np.column_stack(np.nonzero(texture_map > 0))

        tex_stride = max(1, len(tex_coords) // texture_quota)
        tex_coords = tex_coords[::tex_stride]

        coords = np.vstack([edge_coords, tex_coords]) if tex_coords.size else edge_coords
        coords = np.unique(coords, axis=0)

        if coords.shape[0] < 3:
            # Uniform grid fallback
            ys, xs = np.mgrid[0:h:complex(0, max(3, int(math.sqrt(self.max_points)))) ,
                              0:w:complex(0, max(3, int(math.sqrt(self.max_points))))]
            coords = np.column_stack([ys.ravel(), xs.ravel()])

        if coords.shape[0] > self.max_points:
            idx = self.random_state.choice(coords.shape[0], size=self.max_points, replace=False)
            coords = coords[idx]

        ys = coords[:, 0]
        xs = coords[:, 1]
        pts = np.stack([xs, ys], axis=1).astype(np.float64)

        grad_mag = features.gradient_magnitude[ys, xs]
        grad_ori = features.gradient_orientation[ys, xs]
        gabor_resp = features.gabor_response[ys, xs]
        gabor_ori = features.gabor_orientation[ys, xs]

        return SampledPointSet(
            coordinates=pts,
            gradient_magnitude=grad_mag,
            gradient_orientation=grad_ori,
            gabor_response=gabor_resp,
            gabor_orientation=gabor_ori,
            image_shape=(h, w),
        )


# ---------------------------------------------------------------------------
# Module ③ – Texture-aware α-shape construction
# ---------------------------------------------------------------------------


def _triangle_circumradius(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    tri_pts = points[simplices]
    a = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], axis=1)
    b = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], axis=1)
    c = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], axis=1)
    s = (a + b + c) / 2.0
    area_sq = np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12)
    area = np.sqrt(area_sq)
    radius = (a * b * c) / (4.0 * area)
    return radius


def _triangle_area(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    tri_pts = points[simplices]
    vec1 = tri_pts[:, 1] - tri_pts[:, 0]
    vec2 = tri_pts[:, 2] - tri_pts[:, 0]
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    return 0.5 * np.abs(cross)


def _variance(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    return ((values - mean) ** 2).mean(axis=1)


@dataclass
class TextureAwareAlphaShapeResult:
    alpha: float
    points: np.ndarray
    simplices: np.ndarray
    radii: np.ndarray
    weights: np.ndarray
    mask: np.ndarray
    areas: np.ndarray

    def selected_triangles(self) -> np.ndarray:
        return self.simplices[self.mask]

    def selected_weights(self) -> np.ndarray:
        return self.weights[self.mask]

    def selected_radii(self) -> np.ndarray:
        return self.radii[self.mask]

    def boundary_edges(self) -> np.ndarray:
        tris = self.selected_triangles()
        if tris.size == 0:
            return np.empty((0, 2), dtype=int)
        edges = np.concatenate(
            [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0
        )
        edges = np.sort(edges, axis=1)
        edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
        return edges_unique[counts == 1]

    def connected_components(self) -> int:
        edges = self.boundary_edges()
        if edges.size == 0:
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
        return float(self.areas[self.mask].sum())

    def perimeter(self) -> float:
        edges = self.boundary_edges()
        if edges.size == 0:
            return 0.0
        pts = self.points
        diffs = pts[edges[:, 0]] - pts[edges[:, 1]]
        lengths = np.linalg.norm(diffs, axis=1)
        return float(lengths.sum())

    def texture_consistency(self) -> float:
        selected = self.selected_weights()
        if selected.size == 0:
            return 0.0
        return float(selected.mean())

    def contours(self) -> List[np.ndarray]:
        edges = self.boundary_edges()
        if edges.size == 0:
            return []
        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        visited_edges = set()
        contours: List[np.ndarray] = []
        for start_u in adjacency:
            for start_v in adjacency[start_u]:
                edge_key = tuple(sorted((start_u, start_v)))
                if edge_key in visited_edges:
                    continue
                contour = [self.points[start_u], self.points[start_v]]
                visited_edges.add(edge_key)
                u, v = start_u, start_v
                while True:
                    neighbors = adjacency[v]
                    next_candidates = [n for n in neighbors if n != u]
                    if not next_candidates:
                        break
                    next_node = next_candidates[0]
                    edge_key = tuple(sorted((v, next_node)))
                    if edge_key in visited_edges:
                        break
                    contour.append(self.points[next_node])
                    visited_edges.add(edge_key)
                    u, v = v, next_node
                    if v == start_u:
                        break
                contours.append(np.array(contour, dtype=np.float32))
        return contours

    def render_mask(self, thickness: int = 2) -> np.ndarray:
        if self.points.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        contours = self.contours()
        if not contours:
            return np.zeros((0, 0), dtype=np.uint8)
        max_x = int(np.ceil(self.points[:, 0].max())) + 1
        max_y = int(np.ceil(self.points[:, 1].max())) + 1
        canvas = np.zeros((max_y, max_x), dtype=np.uint8)
        for contour in contours:
            pts = np.round(contour).astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=thickness)
        return canvas


class TextureAwareAlphaShape:
    """Constructs α-shapes with texture-based weighting."""

    def __init__(self, beta: float = 4.0) -> None:
        self.beta = beta

    def _texture_weights(self, samples: SampledPointSet, simplices: np.ndarray) -> np.ndarray:
        grad = samples.gradient_magnitude[simplices]
        tex = samples.gabor_response[simplices]

        theta = samples.gradient_orientation
        phi = samples.gabor_orientation

        theta_unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        phi_unit = np.stack([np.cos(phi), np.sin(phi)], axis=1)

        theta_cos = theta_unit[simplices, 0]
        theta_sin = theta_unit[simplices, 1]
        phi_cos = phi_unit[simplices, 0]
        phi_sin = phi_unit[simplices, 1]

        texture_diff = (
            _variance(grad)
            + _variance(tex)
            + _variance(theta_cos)
            + _variance(theta_sin)
            + _variance(phi_cos)
            + _variance(phi_sin)
        )

        weights = np.exp(-self.beta * texture_diff)
        return weights

    def build(self, samples: SampledPointSet, alpha: float) -> TextureAwareAlphaShapeResult:
        points = samples.coordinates
        if points.shape[0] < 3:
            raise ValueError("At least three sample points are required")

        triangulation = Delaunay(points)
        simplices = triangulation.simplices
        radii = _triangle_circumradius(points, simplices)
        areas = _triangle_area(points, simplices)
        weights = self._texture_weights(samples, simplices)
        mask = radii <= alpha * weights

        return TextureAwareAlphaShapeResult(
            alpha=alpha,
            points=points,
            simplices=simplices,
            radii=radii,
            weights=weights,
            mask=mask,
            areas=areas,
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


class AlphaStabilityAnalyzer:
    """Evaluates geometry/texture stability curves and selects α*."""

    def __init__(self, lam: float = 0.7) -> None:
        self.lam = lam

    def evaluate(
        self, results: Sequence[TextureAwareAlphaShapeResult]
    ) -> StabilityAnalysis:
        if not results:
            raise ValueError("No α-shape results provided")

        alphas = np.array([res.alpha for res in results], dtype=np.float64)
        order = np.argsort(alphas)
        results = [results[i] for i in order]
        alphas = alphas[order]

        areas = np.array([res.total_area() for res in results])
        components = np.array([res.connected_components() for res in results], dtype=np.float64)
        perimeters = np.array([res.perimeter() for res in results])
        texture_consistency = np.array([res.texture_consistency() for res in results])

        area_derivative = np.gradient(areas, alphas, edge_order=2)
        texture_derivative = np.gradient(texture_consistency, alphas, edge_order=2)

        with np.errstate(divide="ignore", invalid="ignore"):
            csi_geometry = np.exp(-np.abs(area_derivative / np.maximum(areas, 1e-8)))
        csi_texture = np.exp(-np.abs(texture_derivative))
        csi = self.lam * csi_geometry + (1.0 - self.lam) * csi_texture

        records: List[StabilityRecord] = []
        best_idx = int(np.argmax(csi))
        for idx, res in enumerate(results):
            record = StabilityRecord(
                alpha=float(alphas[idx]),
                area=float(areas[idx]),
                components=int(components[idx]),
                perimeter=float(perimeters[idx]),
                texture_consistency=float(texture_consistency[idx]),
                csi_geometry=float(csi_geometry[idx]),
                csi_texture=float(csi_texture[idx]),
                csi=float(csi[idx]),
            )
            records.append(record)

        analysis = StabilityAnalysis(
            records=records,
            best_record=records[best_idx],
            best_result=results[best_idx],
        )
        return analysis


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_texture_alpha_shape(
    image: np.ndarray,
    alpha_values: Sequence[float],
    beta: float = 4.0,
    lam: float = 0.7,
    max_points: int = 4000,
) -> StabilityAnalysis:
    extractor = TextureFeatureExtractor()
    features = extractor.extract(image)

    sampler = PointSampler(max_points=max_points)
    samples = sampler.sample(image, features)

    tas = TextureAwareAlphaShape(beta=beta)
    results = [tas.build(samples, alpha=float(alpha)) for alpha in alpha_values]

    analyzer = AlphaStabilityAnalyzer(lam=lam)
    return analyzer.evaluate(results)

