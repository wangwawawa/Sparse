# -*- coding: utf-8 -*-
"""
Batch sweep version:
Visualizes multiple (q1, q2, p) settings for KDE ribbon + geodesic ridge extraction.
"""

import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, color, exposure, measure, morphology, img_as_float
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.graph import route_through_array
from scipy.interpolate import splprep, splev
import sparse  # 需与脚本同目录

# ----------- 工具函数 -----------
def ensure_dir(p):
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def to_gray01(img):
    if img.ndim == 3: img = color.rgb2gray(img)
    return img_as_float(img)

def image_to_uint8(gray01):
    g = exposure.rescale_intensity(gray01, in_range="image", out_range=(0,1))
    return (g*255 + 0.5).astype(np.uint8)

def largest_closed_contour_from_binary(img01):
    cs = measure.find_contours(img01.astype(float), 0.5)
    if not cs: return None
    best, area = None, -1
    for c in cs:
        x, y = c[:,1], c[:,0]
        a = 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
        if a > area: best, area = np.c_[x,y], a
    return best

def outer_contour_from_sparse(gray_u8, lbd=0.965):
    edge_img = sparse.edge(gray_u8, lbd)
    eg = color.rgb2gray(edge_img) if edge_img.ndim==3 else edge_img.astype(np.float32)/255.0
    eg_bin = (eg > 0.5).astype(np.uint8)
    return largest_closed_contour_from_binary(eg_bin)

def kde_from_dark(gray01, sigma, mask=None):
    H, W = gray01.shape
    sel = gray01 < np.median(gray01)
    if mask is not None: sel &= mask
    ys, xs = np.nonzero(sel)
    img = np.zeros((H,W), dtype=np.float32)
    if xs.size: img[ys, xs] = 1.0
    dens = gaussian_filter(img, sigma=sigma, mode="nearest")
    dens = exposure.rescale_intensity(dens)
    return dens

def ribbon_mask_from_kde(dens, mask, q1, q2, open_rad=2):
    vals = dens[mask]
    v1, v2 = np.percentile(vals, [q1, q2])
    band = (dens >= v1) & (dens <= v2) & mask
    band = morphology.binary_opening(band, morphology.disk(open_rad))
    band = morphology.remove_small_holes(band, 256)
    band = morphology.remove_small_objects(band, 256)
    return band

def pick_terminals_in_ribbon(gmag, ribbon, k_top=600):
    ys, xs = np.nonzero(ribbon)
    if len(xs) < 2:
        H, W = gmag.shape
        return (H//2, W//3), (H//2, 2*W//3)
    vals = gmag[ys, xs]
    idx = np.argsort(vals)[::-1][:k_top]
    cand = np.c_[xs[idx], ys[idx]]
    dists = ((cand[None,:,:]-cand[:,None,:])**2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    s, t = cand[i], cand[j]
    return (int(t[1]), int(t[0])), (int(s[1]), int(s[0]))  # (row,col)

def geodesic_on_ribbon(dens, ribbon, p=2.2, grad_sigma=1.6):
    gmag = gaussian_gradient_magnitude(dens, sigma=grad_sigma)
    gmag = exposure.rescale_intensity(gmag)
    cost = 1.0 / (gmag**p + 1e-6)
    big = cost.max() * 50.0
    cost = np.where(ribbon, cost, big)
    start_rc, end_rc = pick_terminals_in_ribbon(gmag, ribbon, k_top=600)
    idxs, _ = route_through_array(cost, start_rc, end_rc, fully_connected=True)
    path = np.asarray(idxs, dtype=np.float32)
    xy = np.c_[path[:,1], path[:,0]]
    if len(xy) > 20:
        try:
            tck, _ = splprep([xy[:,0], xy[:,1]], s=8.0)
            u = np.linspace(0, 1, len(xy)*2)
            xs, ys = splev(u, tck)
            xy = np.c_[xs, ys]
        except Exception:
            pass
    return xy

# ----------- 主函数 -----------
def run_sweep(
    input_path="data/input/6.png",
    output_path="data/output/6_ridge_sweep.png",
    lbd=0.965,
    kde_sigma=6,
    grad_sigma=1.6
):
    img = io.imread(input_path)
    gray = to_gray01(img)
    H, W = gray.shape
    gray_u8 = image_to_uint8(gray)

    outer = outer_contour_from_sparse(gray_u8, lbd)
    mask = np.zeros((H,W), np.uint8)
    rr, cc = polygon(outer[:,1], outer[:,0], shape=(H,W))
    mask[rr,cc] = 1
    mask_bool = mask.astype(bool)

    dens = kde_from_dark(gray, sigma=kde_sigma, mask=mask_bool)
    dens *= mask_bool.astype(np.float32)

    # 批量参数组合
    combos = [
        (60,82,2.0), (62,85,2.2), (64,87,2.4),
        (66,88,2.2), (68,90,2.0), (70,92,2.4)
    ]

    fig, axs = plt.subplots(3,2, figsize=(10,12), dpi=150)
    axs = axs.ravel()

    for i,(q1,q2,p) in enumerate(combos):
        ribbon = ribbon_mask_from_kde(dens, mask_bool, q1,q2,open_rad=2)
        ridge = geodesic_on_ribbon(dens, ribbon, p=p, grad_sigma=grad_sigma)
        ax = axs[i]
        ax.imshow(gray, cmap="gray", interpolation="nearest")
        ax.plot(outer[:,0], outer[:,1], color="red", lw=1.8)
        ax.plot(ridge[:,0], ridge[:,1], color="cyan", lw=1.5)
        ax.set_title(f"q1={q1}, q2={q2}, p={p}")
        ax.axis("off")

    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("✅ Saved:", output_path)

if __name__ == "__main__":
    run_sweep()
