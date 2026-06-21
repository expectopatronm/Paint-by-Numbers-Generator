from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .color import Lstar_from_rgb, relative_luminance, rgb_to_hsv

# Clean-stencil helpers (adaptive threshold + light thinning + optional tone tweaks)
# ---------------------------
def _map_stencil_brightness_slider(value: float) -> float:
    """0..1 → 0.5..2.0 (like your demo)."""
    return 0.5 + float(np.clip(value, 0.0, 1.0)) * 1.5

def _map_stencil_sharpness_slider(value: float) -> float:
    """0..1 → 0.5..3.0 (like your demo)."""
    return 0.5 + float(np.clip(value, 0.0, 1.0)) * 2.5

def _adjust_brightness_rgb(image_rgb_u8: np.ndarray, factor: float) -> np.ndarray:
    pil_img = Image.fromarray(image_rgb_u8)
    out = ImageEnhance.Brightness(pil_img).enhance(float(factor))
    return np.array(out)

def _adjust_sharpness_rgb(image_rgb_u8: np.ndarray, factor: float) -> np.ndarray:
    pil_img = Image.fromarray(image_rgb_u8)
    out = ImageEnhance.Sharpness(pil_img).enhance(float(factor))
    return np.array(out)

def _apply_clean_stencil_rgb(image_rgb_u8: np.ndarray, *, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Adaptive threshold + light erosion to get a crisp, printable stencil.
    Input/Output: RGB uint8.
    """
    gray = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2GRAY)
    block_size = max(3, block_size | 1)   # must be odd, >=3
    C = int(C)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )

    kernel = np.ones((2, 2), np.uint8)
    thin = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)

    result = cv2.bitwise_not(thin)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

# ---------------------------
# Grouping strategies (MIXED palette) — exclusive
# ---------------------------
def group_classic_exclusive(palette: np.ndarray) -> Dict[str, List[int]]:
    """
    Classic buckets by luminance/saturation with **exclusive assignment**:
      1) highs (top ~20% luminance)
      2) darks (bottom ~25% luminance)
      3) neutrals (low saturation ≤ 0.20, excluding highs)
      4) mids (everything else)
    """
    n = len(palette)
    lums = np.array([relative_luminance(c) for c in palette])
    sats = np.array([rgb_to_hsv(c)[1] for c in palette])
    q25 = np.quantile(lums, 0.25); q80 = np.quantile(lums, 0.80)

    assigned = np.full(n, False, dtype=bool)
    highs = [i for i in range(n) if (lums[i] >= q80)]
    for i in highs: assigned[i] = True
    darks = [i for i in range(n) if (not assigned[i]) and (lums[i] <= q25)]
    for i in darks: assigned[i] = True
    neutrals = [i for i in range(n) if (not assigned[i]) and (sats[i] <= 0.20)]
    for i in neutrals: assigned[i] = True
    mids = [i for i in range(n) if not assigned[i]]
    return {"darks": darks, "mids": mids, "neutrals": neutrals, "highs": highs}


def group_value5_exclusive(palette: np.ndarray) -> Dict[str, List[int]]:
    """Five value bands by luminance percentiles (exclusive by construction)."""
    L = np.array([relative_luminance(c) for c in palette])
    q10, q25, q70, q85 = np.quantile(L, [0.10, 0.25, 0.70, 0.85])
    deep = [i for i in range(len(palette)) if L[i] <= q10]
    core = [i for i in range(len(palette)) if (q10 < L[i] <= q25)]
    mids = [i for i in range(len(palette)) if (q25 < L[i] <= q70)]
    half = [i for i in range(len(palette)) if (q70 < L[i] <= q85)]
    highs = [i for i in range(len(palette)) if L[i] > q85]
    return {"deep": deep, "core": core, "mids": mids, "half": half, "highs": highs}


def adjacent_rings_by_value(labels_full: np.ndarray,
                            approx_palette: np.ndarray) -> List[List[int]]:
    """
    Group colors into adjacency-based 'rings' from outer → inner, and
    sort each ring dark → light.

    - Adjacency priority: rings are BFS layers from the image border in the
      label adjacency graph.
    - Dark→light: within each ring we sort by relative luminance ascending.
    """
    H, W = labels_full.shape
    C = approx_palette.shape[0]

    # --- Build adjacency graph between labels (4-neighborhood) ---
    adj: List[set[int]] = [set() for _ in range(C)]

    # vertical neighbors
    for y in range(H - 1):
        row = labels_full[y]
        next_row = labels_full[y + 1]
        diff = (row != next_row)
        if not diff.any():
            continue
        xs = np.where(diff)[0]
        for x in xs:
            a = int(row[x])
            b = int(next_row[x])
            if a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)

    # horizontal neighbors
    for y in range(H):
        row = labels_full[y]
        diff = (row[:-1] != row[1:])
        if not diff.any():
            continue
        xs = np.where(diff)[0]
        for x in xs:
            a = int(row[x])
            b = int(row[x + 1])
            if a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)

    # --- Seeds: labels that touch the border are 'outer' (distance 0) ---
    border_labels = (
        set(labels_full[0, :]) |
        set(labels_full[-1, :]) |
        set(labels_full[:, 0]) |
        set(labels_full[:, -1])
    )

    dist: List[int | None] = [None] * C
    q = deque()
    for lab in border_labels:
        i = int(lab)
        dist[i] = 0
        q.append(i)

    # BFS over adjacency graph
    while q:
        i = q.popleft()
        for j in adj[i]:
            if dist[j] is None:
                dist[j] = dist[i] + 1
                q.append(j)

    # Unreached labels (e.g. unused clusters) → shove them into a final inner ring
    if any(d is None for d in dist):
        maxd = max(d for d in dist if d is not None) if any(d is not None for d in dist) else 0
        for i in range(C):
            if dist[i] is None:
                dist[i] = maxd + 1

    # --- Build rings by distance ---
    rings: Dict[int, List[int]] = {}
    for i, d in enumerate(dist):
        rings.setdefault(int(d), []).append(i)

    # --- Sort rings by distance (outer→inner) and each ring dark→light ---
    ordered_rings: List[List[int]] = []
    for d in sorted(rings.keys()):
        idxs = rings[d]

        # adjacency priority: we keep rings split by d
        # dark→light inside the ring
        idxs.sort(key=lambda i: relative_luminance(approx_palette[i]))
        ordered_rings.append(idxs)

    return ordered_rings


def build_value_tweaks(palette: np.ndarray, recipes_text: List[str], *, threshold=0.25) -> Dict[int, str]:
    """Suggest tiny +/- value tweaks for colors sharing the same recipe."""
    groups = {}
    for i, r in enumerate(recipes_text): groups.setdefault(r, []).append(i)
    tweaks = {i: "" for i in range(len(palette))}
    for _, idxs in groups.items():
        if len(idxs) <= 1: continue
        Ls = np.array([Lstar_from_rgb(palette[i]) for i in idxs], float)
        L_mean = float(Ls.mean())
        for ci, L in zip(idxs, Ls):
            delta = L - L_mean
            tweaks[ci] = ("Value tweak: + tiny White" if delta > threshold
                          else "Value tweak: + tiny Black" if delta < -threshold
                          else "Value tweak: none (base)")
    return tweaks

# ---------------------------
# Image / label utilities
# ---------------------------
def ensure_gray(bgr: np.ndarray) -> np.ndarray:
    """Ensure single-channel uint8 grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def im2float01(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def float01_to_u8(imgf: np.ndarray) -> np.ndarray:
    return (np.clip(imgf, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def clahe_gray(gray_u8: np.ndarray, clip=2.0, tiles=8) -> np.ndarray:
    """CLAHE for local contrast normalization."""
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(gray_u8)


def canny_from_gradients(gray_u8: np.ndarray, low_high_ratio=0.35, high_pct=90) -> Tuple[int, int]:
    """Set Canny thresholds from gradient percentiles for robustness."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).ravel()
    high = float(np.percentile(mag, high_pct))
    high = np.clip(high, 10, 255)
    low = max(5.0, high * low_high_ratio)
    return int(low), int(high)


def remove_small_components_bool(bin_u8: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny blobs from a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def illumination_flatten(gray_u8: np.ndarray, smin: int, strength01: float) -> np.ndarray:
    """Divide-by-blur illumination correction."""
    if strength01 <= 0: return gray_u8
    sigma = smin * lerp(0.03, 0.08, strength01)
    base = cv2.GaussianBlur(gray_u8, (0, 0), sigma)
    g = im2float01(gray_u8); b = im2float01(base)
    flat = np.clip(g / (b + 1e-4), 0, 2.5)
    flat = flat / flat.max() if flat.max() > 0 else flat
    return float01_to_u8(flat)


def bilateral_edge_aware(gray_u8: np.ndarray, strength01: float) -> np.ndarray:
    """Bilateral filter that preserves edges while taming textures."""
    if strength01 <= 0: return gray_u8
    sigma_color = lerp(10, 80, strength01)
    sigma_space = lerp(3, 12, strength01)
    return cv2.bilateralFilter(gray_u8, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def _auto_edge_mask(edge_strength_u8: np.ndarray, target_fg=0.04, min_fg=0.01, max_fg=0.08, iters=8) -> np.ndarray:
    """Find a binary edge mask that lands in a desired foreground fraction."""
    es = edge_strength_u8.astype(np.uint8); H, W = es.shape[:2]
    N = H * W; nz = es[es > 0]
    if nz.size == 0: return np.zeros_like(es, dtype=np.uint8)
    lo, hi = 50.0, 98.0; best = None
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        T = np.percentile(nz, p)
        _, binm = cv2.threshold(es, int(T), 255, cv2.THRESH_BINARY)
        fg = np.count_nonzero(binm) / float(N)
        best = binm
        if fg < min_fg: lo = 45.0; hi = p
        elif fg > max_fg: lo = p; hi = 99.0
        else: break
    return best

# ---------------------------
# Pencil sketch (OLD image-edge method)
# ---------------------------
def pencil_readable_norm(
    bgr: np.ndarray,
    sketchiness01=0.99,
    softness01=0.1,
    highlight_clip01=0.99,
    edge_boost01=0.99,
    texture_suppression01=0.1,
    illumination01=0.1,
    despeckle01=0.25,
    stroke01=0.1,
    line_floor01=0.99,
    use_clahe=True,
    gamma_midtones=0.99,
    canny_high_pct=90,
) -> np.ndarray:
    """
    OLD method: produce a clean pencil-like grayscale (uint8) from image edges.
    Suitable for printing and as an underlay via multiply blending.
    """
    gray = ensure_gray(bgr); h, w = gray.shape[:2]; smin = min(h, w)

    gray = illumination_flatten(gray, smin, illumination01)
    if use_clahe:
        gray = clahe_gray(gray, clip=lerp(1.3, 2.0, softness01), tiles=8)

    blur_sharp = cv2.GaussianBlur(gray, (0, 0), smin * 0.003)
    sharp = cv2.addWeighted(gray, 1.4, blur_sharp, -0.4, 0)

    g_s = bilateral_edge_aware(sharp, texture_suppression01)
    gf = im2float01(g_s)
    gf = np.power(np.clip(gf, 0, 1), gamma_midtones)

    inv = 1.0 - gf
    sigma = smin * lerp(0.006, 0.016, softness01)
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma, sigmaY=sigma)
    denom = np.maximum(1e-4, 1.0 - blur)
    dodge = np.clip(gf / denom, 0, 1)
    dodge = np.minimum(dodge, lerp(0.90, 0.975, highlight_clip01))

    low, high = canny_from_gradients(
        g_s,
        low_high_ratio=lerp(0.32, 0.50, 1.0 - sketchiness01),
        high_pct=int(np.clip(canny_high_pct, 50, 99)),
    )
    can = cv2.Canny(g_s, low, high)

    sigma1 = smin * lerp(0.003, 0.010, sketchiness01)
    sigma2 = sigma1 * 1.6
    g1 = cv2.GaussianBlur(gf, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gf, (0, 0), sigma2)
    dog = g1 - g2
    tau = lerp(0.9, 1.18, sketchiness01)
    phi = lerp(8.0, 22.0, sketchiness01 + edge_boost01 * 0.3)
    xdog = 1.0 - (0.5 * (1 + np.tanh(phi * (dog - tau))))
    xdog_u8 = float01_to_u8(xdog)

    edge_mix = cv2.max(can, xdog_u8)
    target_fg = lerp(0.025, 0.065, sketchiness01)
    edge_bin = _auto_edge_mask(edge_mix, target_fg=target_fg, min_fg=0.015, max_fg=0.09)

    if despeckle01 > 0:
        min_area = int(lerp(0, 0.0020, despeckle01) * (h * w))
        edge_bin = remove_small_components_bool(edge_bin, min_area)

    k = max(2, int(round(min(h, w) * lerp(0.0015, 0.0040, stroke01))))
    edge_bin = cv2.dilate(edge_bin, np.ones((k, k), np.uint8), 1)

    edge_mask = edge_bin.astype(np.float32) / 255.0
    ink_floor = 1.0 - (line_floor01 * edge_mask)
    tone_edge_mul = 1.0 - 0.40 * edge_mask

    pencil = np.minimum(dodge * tone_edge_mul, ink_floor)
    return float01_to_u8(pencil)


def render_image_sketch_gray(bgr: np.ndarray, args=None, **kwargs) -> np.ndarray:
    edge_pct = float(getattr(args, "edge_percentile", kwargs.pop("canny_high_pct", 90.0)))
    return pencil_readable_norm(bgr, canny_high_pct=edge_pct, **kwargs)


def original_edge_sketch_with_grid(img_pil: Image.Image, grid_step=80, grid_color=200, **pencil_kwargs) -> Image.Image:
    """
    Legacy helper: produce the OLD pencil sketch and draw a crisp grid on top.
    Returns a PIL L-mode image for the “Original Edge Sketch + Grid” page.
    """
    rgb = np.array(img_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sketch_u8 = render_image_sketch_gray(bgr, **pencil_kwargs)  # grayscale uint8
    rgb_sketch = cv2.cvtColor(sketch_u8, cv2.COLOR_GRAY2RGB)
    rgb_with_grid = add_grid_to_rgb(rgb_sketch, grid_step=grid_step, grid_color=grid_color)
    gray_with_grid = cv2.cvtColor(rgb_with_grid, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray_with_grid, mode="L")


def load_external_sketch_gray(sketch_path: str, size: tuple[int, int]) -> np.ndarray:
    """
    Load a user-supplied sketch/stencil image as grayscale uint8, resized to
    match the working image size. Black lines on white paper are expected.
    """
    if not sketch_path:
        raise ValueError("External sketch path is empty.")
    if not os.path.exists(sketch_path):
        raise FileNotFoundError(f"External sketch file not found: {sketch_path}")

    with Image.open(sketch_path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im.convert("RGBA"))
            im = bg.convert("L")
        else:
            im = im.convert("L")
        if im.size != size:
            im = im.resize(size, resample=Image.BILINEAR)
        return np.array(im, dtype=np.uint8)

def add_grid_to_rgb(arr: np.ndarray, grid_step=80, grid_color=200) -> np.ndarray:
    """Overlay a grid onto an RGB uint8 image array, non-destructively."""
    out = arr.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("add_grid_to_rgb expects an HxWx3 RGB array.")
    if grid_step and grid_step > 0:
        out[:, ::grid_step, :] = grid_color
        out[::grid_step, :, :] = grid_color
    return out

# ---------------------------
# Region cleanup
# ---------------------------
def potts_smooth_labels(
    rgb_u8: np.ndarray,
    labels_u8: np.ndarray,
    *,
    centroids_rgb_u8: np.ndarray | None = None,
    beta: float = 7.0,
    iterations: int = 4,
) -> np.ndarray:
    """
    Smooth a clustered label map with a Potts/MRF-style prior.

    Each pixel keeps a color-fidelity term against the cluster color, while
    the Potts term rewards agreeing with its 4-neighbors. This removes many
    isolated speckles without switching to superpixels.
    """
    if iterations <= 0 or beta <= 0:
        return labels_u8

    labels = labels_u8.astype(np.int32, copy=True)
    label_ids = np.unique(labels)
    if label_ids.size <= 1:
        return labels_u8

    rgb = rgb_u8.astype(np.uint8, copy=False)
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    if centroids_rgb_u8 is not None:
        centroids_rgb = np.asarray(centroids_rgb_u8, dtype=np.uint8)
        centroids_lab = cv2.cvtColor(centroids_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    else:
        centroids_lab = np.zeros((int(label_ids.max()) + 1, 3), dtype=np.float32)
        for label_id in label_ids:
            mask = labels == int(label_id)
            if np.any(mask):
                centroids_lab[int(label_id)] = np.mean(lab_img[mask], axis=0)

    h, w = labels.shape
    beta = float(beta)

    for _ in range(max(1, int(iterations))):
        old = labels.copy()
        best_labels = old.copy()
        best_energy = np.full((h, w), np.inf, dtype=np.float32)

        up = np.empty_like(old)
        down = np.empty_like(old)
        left = np.empty_like(old)
        right = np.empty_like(old)
        up[0, :] = old[0, :]
        up[1:, :] = old[:-1, :]
        down[-1, :] = old[-1, :]
        down[:-1, :] = old[1:, :]
        left[:, 0] = old[:, 0]
        left[:, 1:] = old[:, :-1]
        right[:, -1] = old[:, -1]
        right[:, :-1] = old[:, 1:]

        for label_id in label_ids:
            lid = int(label_id)
            if lid >= len(centroids_lab):
                continue
            diff = lab_img - centroids_lab[lid]
            fidelity = np.sqrt(np.sum(diff * diff, axis=2))
            smooth = (
                (up != lid).astype(np.float32)
                + (down != lid).astype(np.float32)
                + (left != lid).astype(np.float32)
                + (right != lid).astype(np.float32)
            )
            energy = fidelity + beta * smooth
            take = energy < best_energy
            best_energy[take] = energy[take]
            best_labels[take] = lid

        labels = best_labels
        if np.array_equal(labels, old):
            break

    return labels.astype(labels_u8.dtype, copy=False)


def cleanup_label_regions(labels_u8: np.ndarray, *, min_region_px: int = 0, min_region_pct: float = 0.0) -> np.ndarray:
    """
    Remove/merge impractically small components in the label map:
      - Reassign tiny component pixels to the most frequent neighboring label.
    """
    H, W = labels_u8.shape
    total_px = H * W
    thr = max(int(round(min_region_px)), int(round(min_region_pct * 0.01 * total_px)))
    if thr <= 0:
        return labels_u8

    labels = labels_u8.copy()
    unique_labels = np.unique(labels)
    kernel = np.ones((3, 3), np.uint8)

    for lab in unique_labels:
        mask = (labels == lab).astype(np.uint8) * 255
        num, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= thr:
                continue
            comp_mask = (cc_labels == i).astype(np.uint8)
            ring = cv2.dilate(comp_mask, kernel, 1) - comp_mask
            neighbor_labels = labels[ring.astype(bool)]
            if neighbor_labels.size == 0:
                continue
            vals, counts = np.unique(neighbor_labels[neighbor_labels != lab], return_counts=True)
            if vals.size == 0:
                continue
            new_lab = int(vals[np.argmax(counts)])
            labels[comp_mask.astype(bool)] = new_lab

    return labels
