# Texture-Aware α-Shape Contour Detection

## 1. Research Objective

Design an α-shape model that fuses texture-aware features so contour detection considers both geometry and image texture, approximating human perceptual contours.

* **Traditional α-shape:** Constructs shapes based solely on point locations.
* **Texture-aware α-shape:** Uses texture consistency between points to guide connections for perceptually faithful shapes.

## 2. System Architecture Overview

The system is organized into four primary modules.

| Module | Input | Output | Role |
| --- | --- | --- | --- |
| ① Texture Feature Extraction | Grayscale image | Per-pixel texture feature vectors | Capture local gradient, orientation, and texture consistency |
| ② Point Sampling & Feature Mapping | Texture feature maps | Sampled points with texture descriptors | Select edge/texture-salient pixels and preserve descriptors |
| ③ Texture-Weighted α-Shape | Point set with features | Perceptual contour (edges/polygons) | Adjust α-shape geometry using texture weights |
| ④ Stability Evaluation & α Selection | Contours across α values | Optimal α* and final contour | Choose perceptual contour using geometric and texture stability |

## 3. Module Implementation Details

### 3.1 Texture Feature Extraction

**Goal:** Describe local texture with a multi-dimensional vector encoding edge strength and orientation.

* **Input:** Grayscale image \(I(x, y)\)
* **Output:** Per-pixel texture feature vector \(F(x, y) = [G(x, y), \theta(x, y), R(x, y), \phi(x, y)]\)

| Component | Meaning | Extraction |
| --- | --- | --- |
| \(G(x, y)\) | Gradient magnitude | Sobel operator \(\sqrt{G_x^2 + G_y^2}\) |
| \(\theta(x, y)\) | Edge orientation | \(\tan^{-1}(G_y / G_x)\) |
| \(R(x, y)\) | Dominant texture response | Max response over multi-scale, multi-direction Gabor filter bank |
| \(\phi(x, y)\) | Dominant texture orientation | Orientation of maximal Gabor response |

**Key design choices:**

* 8 orientations, 3 scales for the Gabor bank.
* Normalize magnitudes and responses to \([0, 1]\).
* Optional lightweight descriptors: replace Gabor with Local Binary Patterns (LBP) or Haralick features.

### 3.2 Point Sampling & Feature Mapping

**Goal:** Select representative points for α-shape (edge or texture-salient) with associated texture features.

1. **Sampling strategy**
   * *Edge-prior sampling:* Use Canny/HED/Sobel thresholding to collect edge points.
   * *Texture-salient sampling:* Supplement points in regions with high local variance or Gabor response.
   * *Downsampling:* Target \(N \approx \frac{\text{width} \times \text{height}}{100}\) to reduce computation.

2. **Feature mapping**
   * Map each point \(p = (x, y)\) to \(F_p = [G_p, \theta_p, R_p, \phi_p]\).
   * Produce enhanced point set \(P = \{(x_i, y_i, F_i)\}_{i=1}^N\).

### 3.3 Texture-Weighted α-Shape

1. **Baseline α-shape recap**
   * Compute Delaunay triangulation for \(P\).
   * For each triangle \(t\), compute circumradius \(r_t\).
   * Retain \(t\) if \(r_t \leq \alpha\).
   * Boundary of retained triangles yields contour.

2. **Texture weighting**

   *Texture consistency:* For triangle \(t\) with vertices \(i, j, k\), define
   \[
   W_{\text{tex}}(t) = \exp(-\beta \, \mathcal{D}_{\text{tex}}(t))
   \]
   where
   \[
   \mathcal{D}_{\text{tex}}(t) = \operatorname{Var}(G_i, G_j, G_k) + \operatorname{Var}(R_i, R_j, R_k) + \operatorname{Var}(\cos \theta_i, \cos \theta_j, \cos \theta_k) + \operatorname{Var}(\cos \phi_i, \cos \phi_j, \cos \phi_k)
   \]
   and \(\beta \in [3, 6]\) controls texture influence.

   *Retention criterion:* Adjust α-shape condition to
   \[
   r_t \leq \alpha \cdot W_{\text{tex}}(t).
   \]

   This favors triangles with coherent texture while suppressing cross-texture connections.

3. **Output:** Boundary of retained triangles forms texture-aware contour \(\Gamma_{\text{tex}}(\alpha)\).

### 3.4 Stability Analysis & α Selection

1. **α sweep:** Evaluate \(\alpha \in [\alpha_{\min}, \alpha_{\max}]\) with \(M\) scales to build \(\Gamma_{\text{tex}}(\alpha_m)\).
2. **Metrics:**

   | Metric | Meaning | Computation |
   | --- | --- | --- |
   | \(A(\alpha)\) | Contour area | Sum polygon areas |
   | \(C(\alpha)\) | Connected components | Count contours |
   | \(T(\alpha)\) | Avg. texture consistency | Mean \(W_{\text{tex}}\) |
   | \(P(\alpha)\) | Total perimeter | Sum perimeters |

   *Geometric stability:* \(CSI_g(\alpha) = \exp(-|A'(\alpha) / A(\alpha)|)\)

   *Texture stability:* \(CSI_t(\alpha) = \exp(-|T'(\alpha)|)\)

   *Combined:* \(CSI(\alpha) = \lambda CSI_g(\alpha) + (1 - \lambda) CSI_t(\alpha)\), \(\lambda \in [0.5, 0.8]\).

3. **Optimal α:** \(\alpha^* = \arg\max_{\alpha} CSI(\alpha)\).
4. **Deliverables:** Final contour \(\Gamma_{\text{tex}}(\alpha^*)\), stability curves, \(\alpha^*\) visualization.

## 4. Experimental Protocol

### 4.1 Dataset

* **SIOS** dataset of sparse-structured objects.

### 4.2 Baselines

| Category | Algorithm | Notes |
| --- | --- | --- |
| Traditional | Canny, Sobel | Baselines |
| Learning-based | HED, RCF, U²Net | Deep contour detectors |
| Geometric | Vanilla α-shape | Geometric-only reference |
| Proposed | Texture-aware α-shape | This work |

### 4.3 Evaluation Metrics

| Metric | Type | Description |
| --- | --- | --- |
| ODS / OIS F-measure | Standard boundary accuracy | Compare with ground truth |
| Boundary Recall / Precision | Boundary metrics | — |
| Closure Rate | Contour closure rate | — |
| Topological Consistency | Connectivity stability | Changes in components |
| Texture Coherence Score | Novel metric | Texture uniformity inside contour |

## 5. Workflow

| Stage | Task | Output |
| --- | --- | --- |
| ① | Data preparation (grayscale, normalization) | Preprocessed images |
| ② | Sobel + Gabor feature extraction | Texture feature maps |
| ③ | Point sampling + feature mapping | Point set \(P\) |
| ④ | Multi-scale α-shape with texture weights | Contour sets for each α |
| ⑤ | Stability computation & α* selection | \(\alpha^*\), CSI curves |
| ⑥ | Quantitative evaluation & visualization | Metrics table, contour overlays |
| ⑦ | Ablation study | Results without Sobel/Gabor |
| ⑧ | Parameter analysis | Influence of \(\alpha, \beta, \lambda\) |

## 6. Parameter Recommendations

| Parameter | Meaning | Suggested range |
| --- | --- | --- |
| \(\alpha\) | Contour tightness | [5, 25] |
| \(\beta\) | Texture weight | [2, 6] |
| \(\lambda\) | Geometry vs. texture balance | [0.5, 0.8] |
| Sampling stride | Point density | 1–3 |
| Gabor orientations × scales | Filter diversity | 8 × 3 |

## 7. Expected Observations

| Observation | Explanation |
| --- | --- |
| Smooth α-shape transitions | Penalizing cross-texture edges |
| Robust contours on weak edges | Texture similarity supplements gradients |
| Fewer breaks in textured backgrounds | Multi-scale responses reinforce coherence |
| Unimodal stability curve | Perceptually optimal \(\alpha^*\) aligns with human observation |

## 8. Key Takeaway

> Do not rebuild α-shape from scratch; teach it to **perceive texture**, evolving the algorithm from pure geometry into a **perceptual model**.
