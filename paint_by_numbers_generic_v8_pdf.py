#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paint-by-Numbers PDF generator — PROFESSIONAL edition (mixed-palette everywhere).

This version focuses on **color fidelity** and **painter usability** for
professional art reproduction. It implements perceptual clustering in CIELAB,
region cleanup, exclusive grouping, vector (label-based) outlines, optional
region numbering, and a more robust paint-mix recipe search — while preserving
your original flow and guarantees.

------------------------------------------------------------------------------
What this version guarantees
------------------------------------------------------------------------------
• The printed PBN imagery (Page 1 preview, all step/per-color frames, and the
  final completed page) uses ONLY the colors from your **MIXED** palette
  (derived from the limited tube/base colors).
• The Color Key’s left tile swatch shows the **MIXED** color (not the cluster).
• Value-tweak suggestions are computed from MIXED colors (L* deviations within
  identical-recipe groups).
• Grouping into steps (classic/value5/combined) is based on MIXED colors.
• Grouping buckets are **mutually exclusive** (no double-painting).
• Keys have right-justified component swatches; per-row text wraps to swatch.
• Tight margins; frame pages devote ~75% width to the frame and ~25% to the key.
• Clustering supports **Lab** (default) or RGB; Lab improves perceptual fidelity.
• Region post-processing merges/removes impractically small color regions.
• Outline can be derived from **image edges** (OLD method), **label boundaries**
  (NEW method), or **both** — fully switchable; **OLD method is default**.
• Optional **region numbering** (printed indices) at component centroids.
• Integer-mix search allows variable total parts ≤ max_parts and gently prefers
  simpler recipes when ΔE is similar.
• Exhaustive docstrings and comments throughout.

------------------------------------------------------------------------------
CLI (key args)
------------------------------------------------------------------------------
python make_pbn.py input.jpg \
  --colors 25 \
  --cluster-space lab \
  --frame-mode combined \
  --per-color-frames \
  --min-region-pct 0.02 \
  --sketch-style old          # old=image-edge sketch (DEFAULT)
  # or: --sketch-style new    # new=label-boundary outlines
  # or: --sketch-style both   # combine old+new

Arguments of note:
  --cluster-space {lab,rgb}          Clustering color space (Lab recommended).
  --outline-mode {image,labels,both} Low-level switch (image==old, labels==new).
  --sketch-style {old,new,both}      High-level alias; overrides --outline-mode.
  --label-regions                    Print region numbers at centroids.
  --min-region-px / --min-region-pct Region cleanup thresholds.
  --edge-percentile                  Canny high percentile (used).
  ... (see argparse --help for all flags)

"""

from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans

import textwrap as _tw


# ---------------------------
# Base “tube” paint palette
# ---------------------------

BASE_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Titanium White": (245, 245, 245),
    "Lemon Yellow": (250, 239, 80),
    "Vermillion Red": (214, 66, 50),
    "Carmine": (170, 25, 60),
    "Ultramarine": (25, 50, 140),
    "Pthalo Green": (20, 100, 70),
    "Yellow Ochre": (196, 158, 84),
    "Lamp Black": (20, 20, 20),
}


# ---------------------------
# Color science helpers
# ---------------------------

def srgb_to_linear_arr(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert sRGB 0..1 to linear-light 0..1 (vectorized)."""
    rgb_arr = np.clip(rgb_arr, 0.0, 1.0)
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_arr(lin: np.ndarray) -> np.ndarray:
    """Convert linear-light 0..1 to sRGB 0..1 (vectorized)."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def srgb8_to_xyz(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB (0..255) to XYZ (D65)."""
    lin = srgb_to_linear_arr(rgb_u8.astype(np.float32) / 255.0)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    return M @ lin


def xyz_to_srgb8(xyz: np.ndarray) -> np.ndarray:
    """XYZ (D65) to sRGB (0..255)."""
    M = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    lin = M @ xyz
    srgb = np.clip(linear_to_srgb_arr(lin), 0.0, 1.0)
    return srgb * 255.0


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """XYZ (D65) to CIELAB (L*, a*, b*)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t):
        return np.where(t > (6 / 29) ** 3, np.cbrt(t), (1 / 3) * (29 / 6) ** 2 * t + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b], dtype=np.float32)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """CIELAB to XYZ (D65)."""
    L, a, b = lab
    Yn = 1.0
    Xn = 0.95047
    Zn = 1.08883
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t):
        return np.where(t > 6 / 29, t ** 3, (3 * (6 / 29) ** 2) * (t - 4 / 29))

    x = Xn * finv(fx)
    y = Yn * finv(fy)
    z = Zn * finv(fz)
    return np.array([x, y, z], dtype=np.float32)


def rgb8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB (0..255) to CIELAB."""
    return xyz_to_lab(srgb8_to_xyz(rgb_u8))


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    """CIELAB to sRGB (0..255)."""
    return xyz_to_srgb8(lab_to_xyz(lab))


def relative_luminance(rgb_u8: Sequence[int]) -> float:
    """Perceptual luminance Y from sRGB 0..255 (relative; not L*)."""
    lin = srgb_to_linear_arr(np.array(rgb_u8, dtype=np.float32) / 255.0)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def Lstar_from_rgb(rgb_u8: Sequence[int]) -> float:
    """Compute CIELAB L* from sRGB 0..255."""
    return float(np.clip(rgb8_to_lab(np.array(rgb_u8, dtype=np.float32))[0], 0, 100))


def deltaE_lab(rgb1_u8: Sequence[int], rgb2_u8: Sequence[int]) -> float:
    """ΔE*ab between two sRGB colors (ΔE*ab; ΔE2000 would be stricter)."""
    return float(np.linalg.norm(rgb8_to_lab(np.array(rgb1_u8, dtype=np.float32)) -
                                rgb8_to_lab(np.array(rgb2_u8, dtype=np.float32))))


# ---------------------------
# Mixing models
# ---------------------------

def mix_linear(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """Linear-light additive mix (approximate for light, less for paint)."""
    w = parts / np.sum(parts)
    lin = np.sum(srgb_to_linear_arr((base_rgbs / 255.0).T) * w, axis=1)
    return np.clip(255 * linear_to_srgb_arr(lin), 0, 255)


def mix_lab(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """Average in Lab space (often naive for real paint, but perceptual)."""
    w = parts / np.sum(parts)
    labs = np.array([rgb8_to_lab(c) for c in base_rgbs], dtype=np.float32)
    lab = np.sum(labs.T * w, axis=1)
    return np.clip(lab_to_rgb8(lab), 0, 255)


def mix_subtractive(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """Simple subtractive-like (1 - Π(1 - c)^w) heuristic in RGB."""
    w = parts / np.sum(parts)
    c = (base_rgbs / 255.0)
    res = 1.0 - np.prod((1.0 - c) ** w[:, None], axis=0)
    return np.clip(res * 255.0, 0, 255)


def mix_km_generic(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Heuristic “KM-like” mixing via Beer–Lambert per-channel.

    True Kubelka–Munk requires pigment optical data. This expedient:
      R_mix = exp( -Σ w_i * A_i ),  A_i = -ln(R_i)
    with R_i as sRGB channel reflectances (0..1).
    """
    w = parts / np.sum(parts)
    R = np.clip(base_rgbs / 255.0, 1e-4, 1.0)
    A = -np.log(R)
    A_mix = np.sum(A.T * w, axis=1)
    R_mix = np.exp(-A_mix)
    return np.clip(R_mix * 255.0, 0, 255)


def mix_color(parts: np.ndarray, base_rgbs: np.ndarray, model: str) -> np.ndarray:
    """Dispatch to the requested mixing model."""
    if model == "linear":
        return mix_linear(parts, base_rgbs)
    elif model == "lab":
        return mix_lab(parts, base_rgbs)
    elif model == "subtractive":
        return mix_subtractive(parts, base_rgbs)
    elif model == "km":
        return mix_km_generic(parts, base_rgbs)
    else:
        return mix_linear(parts, base_rgbs)


# ---------------------------
# Search helpers (recipes) — variable sum(parts) and gentle regularization
# ---------------------------

def enumerate_partitions_upto(total: int, k: int):
    """Yield k-tuples of nonnegative integers with sum <= total and not all zeros."""
    if k == 1:
        for t in range(total + 1):
            if t > 0:
                yield (t,)
        return
    for i in range(total + 1):
        for rest in enumerate_partitions_upto(total - i, k - 1):
            s = (i,) + rest
            if any(p > 0 for p in s):
                yield s


def integer_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 10,
    max_components: int = 3,
    model: str = "km",
    prefer_simple_lambda_components: float = 0.03,
    prefer_simple_lambda_parts: float = 0.01,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Brute-force an integer-part recipe approximating target_rgb using up to
    `max_components` base colors and **sum(parts) <= max_parts**.

    score = ΔE*ab(mix, target) +
            λc * (num_components - 1) +
            λp * (sum(parts) / max_parts)

    Returns:
        (entries, best_rgb, best_deltaE)
    """
    base_rgbs_full = np.array([BASE_PALETTE[n] for n in base_names], dtype=float)
    target = np.array(target_rgb, dtype=float)

    best_score = float("inf")
    best_err = float("inf")
    best_entries: List[Tuple[str, int]] = []
    best_rgb = target

    N = len(base_names)
    max_components = max(1, min(max_components, N, (max_parts if max_parts > 0 else 1)))

    for m in range(1, max_components + 1):
        for combo in itertools.combinations(range(N), m):
            base_rgbs = base_rgbs_full[list(combo)]
            for parts in enumerate_partitions_upto(max_parts, m):
                s = sum(parts)
                if s == 0:
                    continue
                parts_arr = np.array(parts, dtype=float)
                mix_rgb = mix_color(parts_arr, base_rgbs, model)
                err = deltaE_lab(mix_rgb, target)
                score = (err
                         + prefer_simple_lambda_components * (m - 1)
                         + prefer_simple_lambda_parts * (s / float(max_parts)))
                if score < best_score:
                    best_score = score
                    best_err = err
                    best_rgb = mix_rgb
                    best_entries = [(base_names[i], int(p)) for i, p in zip(combo, parts) if p > 0]

    if len(best_entries) == 1:
        n, p = best_entries[0]
        best_entries = [(n, max(1, p))]

    return best_entries, best_rgb, best_err


def recipe_text(entries: List[Tuple[str, int]]) -> str:
    """Human-readable e.g. '2 parts Yellow + 1 part Black'."""
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries]) if entries else "—"


def rgb_to_hsv(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """Return (h, s, v) with s,v ∈ [0..1]."""
    rgb = np.array(rgb, dtype=float) / 255.0
    mx = float(rgb.max()); mn = float(rgb.min()); diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == rgb[0]:
        h = (60 * ((rgb[1] - rgb[2]) / diff) + 360) % 360
    elif mx == rgb[1]:
        h = (60 * ((rgb[2] - rgb[0]) / diff) + 120) % 360
    else:
        h = (60 * ((rgb[0] - rgb[1]) / diff) + 240) % 360
    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return float(h), float(s), float(v)


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


def original_edge_sketch_with_grid(img_pil: Image.Image, grid_step=80, grid_color=200, **pencil_kwargs) -> Image.Image:
    """
    Legacy helper: produce the OLD pencil sketch and draw a crisp grid on top.
    Returns a PIL L-mode image for the “Original Edge Sketch + Grid” page.
    """
    rgb = np.array(img_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sketch_u8 = pencil_readable_norm(bgr, **pencil_kwargs)  # grayscale uint8
    rgb_sketch = cv2.cvtColor(sketch_u8, cv2.COLOR_GRAY2RGB)
    rgb_with_grid = add_grid_to_rgb(rgb_sketch, grid_step=grid_step, grid_color=grid_color)
    gray_with_grid = cv2.cvtColor(rgb_with_grid, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray_with_grid, mode="L")


# ---------------------------
# Label-based OUTLINE + numbering (NEW)
# ---------------------------

def label_boundaries_u8(labels_u8: np.ndarray, thick_px: int = 1) -> np.ndarray:
    """Compute region boundaries from a label map; returns binary uint8 mask."""
    up = np.zeros_like(labels_u8); up[1:] = (labels_u8[1:] != labels_u8[:-1])
    left = np.zeros_like(labels_u8); left[:, 1:] = (labels_u8[:, 1:] != labels_u8[:, :-1])
    edges = np.logical_or(up, left).astype(np.uint8) * 255
    if thick_px > 1:
        edges = cv2.dilate(edges, np.ones((thick_px, thick_px), np.uint8), 1)
    return edges


def label_component_centers(labels_u8: np.ndarray) -> Dict[int, List[Tuple[float, float]]]:
    """Connected components per label → list of centers (cx, cy) per label."""
    H, W = labels_u8.shape
    centers_by_label: Dict[int, List[Tuple[float, float]]] = {}
    for lab in np.unique(labels_u8):
        mask = (labels_u8 == lab).astype(np.uint8) * 255
        num, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        pts = []
        for i in range(1, num):
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))
        centers_by_label[int(lab)] = pts
    return centers_by_label


def draw_region_numbers_rgb(
    rgb_img_u8: np.ndarray,
    centers_by_label: Dict[int, List[Tuple[float, float]]],
    *,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    """Overlay numeric labels at component centroids with readable outline."""
    out = rgb_img_u8.copy()
    for lab, pts in centers_by_label.items():
        label_str = str(int(lab) + 1)  # user-facing indices 1..K
        for (cx, cy) in pts:
            pos = (int(round(cx)), int(round(cy)))
            cv2.putText(out, label_str, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(out, label_str, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


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


# ---------------------------
# Color key drawer
# ---------------------------

def draw_color_key(
    ax,
    target_palette: np.ndarray,
    recipes: List[str],
    entries_per_color: List[List[Tuple[str, int]]],
    base_palette: Dict[str, Tuple[int, int, int]],
    *,
    used_indices: List[int] | None = None,
    title="Color Key • Ratios + Component Paints",
    tweaks=None,
    wrap_width=55,
    show_components=True,
    deltaEs=None,
    left_pad=None,
    right_margin=None,
    swatch_step=None,
    swatch_w=None,
    no_band_bg=True,
    text_gap=0.05,
    approx_palette=None,
):
    if used_indices is None:
        used_indices = list(range(len(target_palette)))
    if tweaks is None:
        tweaks = {i: "" for i in range(len(target_palette))}
    base_order = list(base_palette.keys())

    XMAX = 16.5
    LEFT_PAD = 1.25 if left_pad is None else max(1.05, float(left_pad))
    RIGHT_MARGIN = 0.20 if right_margin is None else float(right_margin)
    swatch_w = 0.70 if swatch_w is None else float(swatch_w)
    swatch_step = 0.80 if swatch_step is None else float(swatch_step)

    def comp_names(entries):
        return [n for (n, _) in entries]

    max_n_comp = 0
    for ci in used_indices:
        max_n_comp = max(max_n_comp, len(comp_names(entries_per_color[ci])))

    gutter_right = 16.5 - RIGHT_MARGIN
    band_left = gutter_right - (max_n_comp * swatch_step)
    if not no_band_bg:
        ax.add_patch(Rectangle((band_left - 0.0001, 0), gutter_right - (band_left - 0.0001),
                               len(used_indices), facecolor="white", edgecolor="none", zorder=0.5))

    for row_idx, ci in enumerate(used_indices):
        show_rgb = (approx_palette[ci] if approx_palette is not None else target_palette[ci])
        ax.add_patch(Rectangle((0, row_idx), 1, 1, color=(show_rgb / 255), ec="k", lw=0.2))
        ax.text(0.5, row_idx + 0.5, f"{ci + 1}", ha="center", va="center", fontsize=8, color="black",
                bbox=dict(facecolor=(1, 1, 1, 0.45), edgecolor="none", boxstyle="round,pad=0.1"))

        Lstar = Lstar_from_rgb(show_rgb)
        tweak_str = f" • L*={Lstar:.1f}"
        if deltaEs is not None:
            tweak_str += f" • ΔE≈{deltaEs[ci]:.2f}"
        if tweaks.get(ci, ""):
            tweak_str += f" • {tweaks[ci]}"
        text_str = f"{ci + 1}: {recipes[ci]}{tweak_str}"

        row_comp_names = [n for n in base_order if n in comp_names(entries_per_color[ci])]
        n_comp = len(row_comp_names)
        row_start_x = gutter_right - (n_comp * swatch_step)
        avail_units = max(1.0, (row_start_x - text_gap) - LEFT_PAD)
        full_text_band = 16.5 - RIGHT_MARGIN - LEFT_PAD
        frac = np.clip(avail_units / max(1.0, full_text_band), 0.2, 1.2)
        local_wrap = max(20, int(round(wrap_width * float(frac))))
        ax.text(LEFT_PAD, row_idx + 0.5, _tw.fill(text_str, width=local_wrap), va="center", fontsize=8, wrap=True, zorder=1.0)

        if show_components and n_comp > 0:
            for j, name in enumerate(row_comp_names):
                comp_rgb = np.array(base_palette[name]) / 255.0
                x = row_start_x + j * swatch_step
                ax.add_patch(Rectangle((x, row_idx), swatch_w, 1, color=comp_rgb, ec="k", lw=0.2, zorder=1.5))

    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, len(used_indices))
    ax.invert_yaxis()
    ax.axis("off")
    t = ax.set_title(title + " (swatch = mixed color)", pad=3)
    t.set_wrap(True)


# ---------------------------
# Figure helper
# ---------------------------

def new_fig(size):
    fig = plt.figure(figsize=size)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.965, wspace=0.02, hspace=0.02)
    return fig


# ---------------------------
# Main CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="A4 PDF paint-by-numbers generator (professional fidelity).")
    p.add_argument("input", help="Input image file path")
    p.add_argument("--pdf", default="paint_by_numbers_guide.pdf", help="Output PDF path")
    p.add_argument("--colors", type=int, default=30, help="Number of colors (KMeans clusters)")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                   help="Optional resize WxH for KMeans only (keeps output at full res). Omit to keep original.")
    p.add_argument("--cluster-space", choices=["lab", "rgb"], default="lab",
                   help="Clustering color space (Lab is recommended for perceptual fidelity).")
    p.add_argument("--palette", nargs="*", default=list(BASE_PALETTE.keys()))
    p.add_argument("--components", type=int, default=5, help="Max components per mixed color")
    p.add_argument("--max-parts", type=int, default=10, help="Max TOTAL parts per mixed color (sum(parts) ≤ this)")
    p.add_argument("--mix-model", choices=["linear", "lab", "subtractive", "km"], default="km",
                   help="Mixing model for recipes")
    p.add_argument("--frame-mode", choices=["classic", "value5", "both", "combined"], default="combined",
                   help="Frame set: classic, value5, both (separate), or combined (interleaved 9-step)")
    p.add_argument("--wrap", type=int, default=55, help="Wrap width for color key text")
    p.add_argument("--grid-step", type=int, default=400, help="Grid spacing in pixels (0 = no grid)")
    p.add_argument("--edge-percentile", type=float, default=90.0, help="(Used) Edge detection high percentile for Canny")
    p.add_argument("--hide-components", action="store_true", help="Do not show component swatches in color key")
    p.add_argument("--per-color-frames", action="store_true",
                   help="If set, add a separate frame for each color (inserted before the completed page).")
    p.add_argument("--sketch-alpha", type=float, default=0.55,
                   help="Opacity of the outline underlay (0=no effect, 1=full outline) on step/per-color pages.")
    p.add_argument("--per-color-cumulative", action="store_true",
                   help="Per-color frames build cumulatively: prior colors appear at --prev-alpha; current color is 100%")
    p.add_argument("--prev-alpha", type=float, default=0.75,
                   help="Opacity for all previous colors on cumulative per-color frames (0..1)")
    p.add_argument("--min-region-px", type=int, default=0,
                   help="Remove/merge label components smaller than this many pixels")
    p.add_argument("--min-region-pct", type=float, default=0.0,
                   help="Remove/merge label components smaller than this percentage of total pixels (e.g. 0.5)")
    # Low-level switch (kept for backward compatibility)
    p.add_argument("--outline-mode", choices=["image", "labels", "both"], default="image",
                   help="Outline source: image edges (OLD), label boundaries (NEW), or both. (Default: image/OLD)")
    # High-level alias for convenience (overrides outline-mode if provided)
    p.add_argument("--sketch-style", choices=["old", "new", "both"],
                   help="Alias for --outline-mode: old=image, new=labels, both=both. If set, it overrides --outline-mode.")
    p.add_argument("--label-regions", action="store_true",
                   help="Overlay region numbers at component centroids in frames (user-visible indices 1..K)")
    args = p.parse_args()

    # Map the alias onto outline-mode (alias wins if provided)
    if args.sketch_style:
        args.outline_mode = {"old": "image", "new": "labels", "both": "both"}[args.sketch_style]

    # -------------------------
    # Load + optional resize for clustering
    # -------------------------
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size
    rgb_full = np.array(img)
    bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)

    if args.resize:
        Wc, Hc = map(int, args.resize)
        img_small = img.resize((Wc, Hc), resample=Image.BILINEAR)
    else:
        img_small = img.copy()
    data_small = np.array(img_small)
    Hs, Ws, _ = data_small.shape
    pixels_small = data_small.reshape((-1, 3)).astype(np.float32)

    # -------------------------
    # Perceptual KMeans (Lab default)
    # -------------------------
    if args.cluster_space == "lab":
        def rgbrow_to_labrows(arr_uint8):
            labs = []
            for r, g, b in arr_uint8:
                lab = rgb8_to_lab(np.array([r, g, b], dtype=np.float32))
                labs.append(lab)
            return np.array(labs, dtype=np.float32)
        feats = rgbrow_to_labrows(pixels_small.astype(np.uint8))
    else:
        feats = pixels_small

    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=8)
    kmeans.fit(feats)
    labels_small = kmeans.labels_.reshape(Hs, Ws).astype(np.uint8)

    # Centroids in RGB
    if args.cluster_space == "lab":
        centroids_lab = kmeans.cluster_centers_.astype(np.float32)
        centroids_rgb = [np.clip(np.rint(lab_to_rgb8(lab)), 0, 255) for lab in centroids_lab]
        centroids = np.array(centroids_rgb, dtype=np.uint8)
    else:
        centroids = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)

    # Upsample labels to full res
    labels_full = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_full = np.array(labels_full, dtype=np.uint8)

    # Region cleanup
    labels_full = cleanup_label_regions(
        labels_full,
        min_region_px=max(0, int(args.min_region_px)),
        min_region_pct=max(0.0, float(args.min_region_pct)),
    )

    # Build recipes (MIXED palette)
    names = args.palette
    all_entries, all_recipes, approx_rgbs, deltaEs = [], [], [], []
    for col in centroids.astype(np.float32):
        entries, approx_rgb, err = integer_mix_best(
            col, names, max_parts=args.max_parts, max_components=args.components, model=args.mix_model
        )
        all_entries.append(entries)
        all_recipes.append(recipe_text(entries))
        approx_rgbs.append(np.array(approx_rgb, dtype=float))
        deltaEs.append(err)
    approx_uint8 = np.clip(np.rint(np.array(approx_rgbs)), 0, 255).astype(np.uint8)

    # PBN image from MIXED palette
    seg_mixed_small = approx_uint8[labels_small]
    pbn_image = Image.fromarray(seg_mixed_small).resize((orig_w, orig_h), resample=Image.NEAREST)
    pbn_image = np.array(pbn_image, dtype=np.uint8)

    # Groupings (exclusive)
    classic = group_classic_exclusive(approx_uint8)
    value5 = group_value5_exclusive(approx_uint8)

    classic_order = [
        ("Frame 1 – Highlights", classic["highs"]),
        ("Frame 2 – Shadows / Dark Blocks", classic["darks"]),
        ("Frame 3 – Neutrals / Background", classic["neutrals"]),
        ("Frame 4 – Mid-tone Masses", classic["mids"]),
        ("Frame 5 – Completed", list(range(args.colors))),
    ]
    value5_order = [
        ("Value A – Deep Shadows (lowest ~10%)", value5["deep"]),
        ("Value B – Core Shadows (to ~25%)", value5["core"]),
        ("Value C – Midtones (to ~70%)", value5["mids"]),
        ("Value D – Half-Lights (to ~85%)", value5["half"]),
        ("Value E – Highlights (top ~15%)", value5["highs"]),
    ]

    def frames_from_order(order):
        frames = []
        for title, idxs in order:
            if len(idxs) == 0: continue
            mask = np.isin(labels_full, np.array(idxs, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames.append((title, idxs, frame_img))
        return frames

    if args.frame_mode == "combined":
        painted = set()
        def remaining(idx_list): return [i for i in idx_list if i not in painted]
        sequence = [
            ("Step 1 – Deep Shadows", value5["deep"]),
            ("Step 2 – Core Shadows", value5["core"]),
            ("Step 3 – Shadows / Dark Blocks", classic["darks"]),
            ("Step 4 – Value Midtones", value5["mids"]),
            ("Step 5 – Mid-tone Masses", classic["mids"]),
            ("Step 6 – Neutrals / Background", classic["neutrals"]),
            ("Step 7 – Half-Lights", value5["half"]),
            ("Step 8 – Highlights", value5["highs"]),
            ("Step 9 – Highlight Accents", classic["highs"]),
        ]
        frames_combined = []
        for title, idxs in sequence:
            rem = remaining(idxs); painted.update(rem)
            if not rem: continue
            mask = np.isin(labels_full, np.array(rem, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames_combined.append((title, rem, frame_img))
        frames_to_emit = frames_combined
    elif args.frame_mode == "classic":
        frames_to_emit = frames_from_order(classic_order)
    elif args.frame_mode == "value5":
        frames_to_emit = frames_from_order(value5_order)
    else:
        frames_to_emit = frames_from_order(classic_order) + frames_from_order(value5_order)

    tweaks = build_value_tweaks(approx_uint8, all_recipes, threshold=0.25)

    # -------------------------
    # Outline prep (OLD: image edges) and/or (NEW: label boundaries)
    # -------------------------
    if args.outline_mode in ("image", "both"):
        sketch_gray = pencil_readable_norm(
            bgr_full, canny_high_pct=float(args.edge_percentile)
        )
    else:
        sketch_gray = None

    if args.outline_mode in ("labels", "both"):
        boundaries = label_boundaries_u8(labels_full, thick_px=1)
        bound_norm = (boundaries.astype(np.float32) / 255.0)
        label_outline_gray = float01_to_u8(1.0 - bound_norm * 0.85)
    else:
        label_outline_gray = None

    if args.outline_mode == "image":
        outline_gray = sketch_gray
    elif args.outline_mode == "labels":
        outline_gray = label_outline_gray
    else:
        if sketch_gray is None: outline_gray = label_outline_gray
        elif label_outline_gray is None: outline_gray = sketch_gray
        else:
            a = im2float01(sketch_gray); b = im2float01(label_outline_gray)
            outline_gray = float01_to_u8(np.clip(a * b, 0, 1))

    if outline_gray is not None:
        a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
        sketch_norm = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
        sketch_factor_rgb = ((1.0 - a) + a * sketch_norm)[..., None]
    else:
        sketch_factor_rgb = None

    centers_by_label = label_component_centers(labels_full) if args.label_regions else None

    # -------------------------
    # PDF assembly
    # -------------------------
    A4_LANDSCAPE = (11.69, 8.27)
    with (PdfPages(args.pdf) as pdf):
        # Page 1
        fig = new_fig(A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1.0, 1.55], figure=fig, wspace=0.01, hspace=0.03)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[:, 1])
        ax1.imshow(rgb_full); ax1.set_title("Original", pad=2); ax1.axis("off")
        ax2.imshow(pbn_image)
        ax2.set_title(f"Paint by Numbers ({args.colors} colors) • cluster={args.cluster_space} • mixmodel={args.mix_model} • max parts≤{args.max_parts}", pad=2)
        ax2.axis("off")
        draw_color_key(ax3, centroids, all_recipes, all_entries, BASE_PALETTE,
                       used_indices=list(range(args.colors)),
                       title="Color Key • All Clusters",
                       tweaks=tweaks,
                       wrap_width=int(args.wrap * 1.5),
                       show_components=not args.hide_components,
                       deltaEs=deltaEs,
                       swatch_step=0.55, swatch_w=0.55, right_margin=0.10, left_pad=1.10, no_band_bg=True, text_gap=0.03,
                       approx_palette=approx_uint8)
        pdf.savefig(fig, dpi=300); plt.close(fig)

        # Page 2 (title reflects OLD vs NEW)
        if outline_gray is not None:
            if args.outline_mode == "image":
                # Precisely the legacy look/title
                legacy_page = original_edge_sketch_with_grid(img, grid_step=args.grid_step,
                                                             grid_color=200,
                                                             canny_high_pct=float(args.edge_percentile))
                fig = new_fig(A4_LANDSCAPE); ax = fig.add_subplot(111)
                ax.imshow(legacy_page, cmap='gray'); ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px)", pad=2); ax.axis("off")
                pdf.savefig(fig, dpi=300); plt.close(fig)
            else:
                # New or combined outline page
                fig = new_fig(A4_LANDSCAPE); ax = fig.add_subplot(111)
                ref_rgb = cv2.cvtColor(outline_gray, cv2.COLOR_GRAY2RGB)
                ref_with_grid = add_grid_to_rgb(ref_rgb, grid_step=args.grid_step, grid_color=200)
                ax.imshow(ref_with_grid); ax.set_title(
                    f"Outline + Grid ({'labels' if args.outline_mode=='labels' else 'combined'}) (step={args.grid_step}px)", pad=2)
                ax.axis("off"); pdf.savefig(fig, dpi=300); plt.close(fig)

        # Step pages
        for title, idxs, frame in frames_to_emit:
            if sketch_factor_rgb is not None:
                frame_f = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
                composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
            else:
                composite_u8 = frame

            if centers_by_label is not None:
                composite_u8 = draw_region_numbers_rgb(composite_u8, centers_by_label, font_scale=0.45, thickness=1)

            frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

            fig = new_fig(A4_LANDSCAPE)
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
            axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])

            suffix = " + Grid (outline multiply underlay)" if sketch_factor_rgb is not None else " + Grid"
            axL.imshow(frame_with_grid); axL.set_title(title + suffix, pad=2); axL.axis("off")

            draw_color_key(axR, centroids, all_recipes, all_entries, BASE_PALETTE,
                           used_indices=idxs,
                           title=f"Color Key • {title}",
                           tweaks=tweaks,
                           wrap_width=max(30, int(args.wrap * 0.7)),
                           show_components=not args.hide_components,
                           deltaEs=deltaEs, left_pad=1.25, right_margin=0.18, text_gap=0.05,
                           approx_palette=approx_uint8)
            pdf.savefig(fig, dpi=300); plt.close(fig)

        # Per-color pages (optional)
        if args.per_color_frames:
            a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
            if sketch_factor_rgb is None and outline_gray is not None:
                s = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
                sketch_factor_rgb = ((1.0 - a) + a * s)[..., None]

            H, W = labels_full.shape; prev_mask = np.zeros((H, W), dtype=bool)
            for i in range(args.colors):
                curr_mask = (labels_full == i)
                frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                if args.per_color_cumulative and args.prev_alpha > 0 and prev_mask.any():
                    sel_prev = prev_mask; white_f = 255.0
                    prev_blend = ((1.0 - args.prev_alpha) * white_f +
                                  args.prev_alpha * pbn_image[sel_prev].astype(np.float32)).round().astype(np.uint8)
                    frame_rgb[sel_prev] = prev_blend

                frame_rgb[curr_mask] = pbn_image[curr_mask]

                if sketch_factor_rgb is not None:
                    frame_f = np.clip(frame_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                    composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                    composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
                else:
                    composite_u8 = frame_rgb

                if centers_by_label is not None:
                    composite_u8 = draw_region_numbers_rgb(composite_u8, centers_by_label, font_scale=0.45, thickness=1)

                frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

                fig = new_fig(A4_LANDSCAPE)
                gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
                axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])

                axL.imshow(frame_with_grid)
                axL.set_title((f"Per-Color • #{i + 1} + Grid "
                               f"{'(cumulative, prevα=' + str(args.prev_alpha) + ')' if args.per_color_cumulative else ''} "
                               f"{'(outline multiply underlay)' if sketch_factor_rgb is not None else ''}"), pad=2)
                axL.axis("off")

                draw_color_key(axR, centroids, all_recipes, all_entries, BASE_PALETTE,
                               used_indices=[i],
                               title=f"Color Key • Color #{i + 1}",
                               tweaks=tweaks,
                               wrap_width=max(30, int(args.wrap * 0.7)),
                               show_components=not args.hide_components,
                               deltaEs=deltaEs, left_pad=1.25, right_margin=0.18, text_gap=0.05,
                               approx_palette=approx_uint8)
                pdf.savefig(fig, dpi=300); plt.close(fig)

                if args.per_color_cumulative: prev_mask |= curr_mask
                else: prev_mask[:] = False

        # Final page
        completed_with_grid = add_grid_to_rgb(pbn_image, grid_step=args.grid_step, grid_color=200)
        if centers_by_label is not None:
            completed_with_grid = draw_region_numbers_rgb(completed_with_grid, centers_by_label, font_scale=0.5, thickness=1)
        fig = new_fig(A4_LANDSCAPE); ax = fig.add_subplot(111)
        ax.imshow(completed_with_grid); ax.set_title("Completed — All Colors Applied + Grid", pad=2); ax.axis("off")
        pdf.savefig(fig, dpi=300); plt.close(fig)

    print(f"✅ Saved A4 landscape PDF to {args.pdf} (frame-mode={args.frame_mode}, outline={args.outline_mode})")


if __name__ == "__main__":
    main()
