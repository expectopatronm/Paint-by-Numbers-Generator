#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paint-by-Numbers PDF generator with tinting-strength aware mixing for oil paints.

This version attempts to mitigate two common real-life mismatches:

1. Strong pigments (black, phthalo green, etc.) dominating mixes unrealistically under
   a naive “parts count” model (i.e. “5 parts black” is overkill).
2. Warm browns or flesh tones drifting too pink or oversaturated in predicted mixes.

It does this by:
- Assigning a **strength multiplier** to each base pigment.
- Scaling raw mixing “parts” by strength, then applying a diminishing-returns transform.
- Mixing in a hybrid empirical KM-style reflectance model (absorption + scattering proxies).
- Applying a mild hue-limited bias correction only in warm/brown hue zones.
- Preserving all your clustering, region cleanup, palette enforcement, key drawing,
  PDF output, etc., exactly as before.

You’ll want to tune the `STRENGTH` values and bias thresholds to match your actual pigments.

References:
- The Kubelka–Munk model is foundational to pigment mixing theory :contentReference[oaicite:0]{index=0}
- Modern oil paint mixing experiments use K and S spectra for better predictive accuracy :contentReference[oaicite:1]{index=1}
- Tinting strength (how strongly a pigment “tints” a mix) is well known in coatings and paint science :contentReference[oaicite:2]{index=2}
- Pigment particle size strongly influences tinting strength :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Sequence, Tuple
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import textwrap as _tw
import colorsys

import svgwrite
from skimage.morphology import skeletonize
from skimage import measure


# ---------------------------
# Color space & conversion helpers
# ---------------------------

def srgb_to_linear_arr(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert sRGB (0..1) to linear light (0..1), vectorized."""
    rgb_arr = np.clip(rgb_arr, 0.0, 1.0)
    return np.where(rgb_arr <= 0.04045,
                    rgb_arr / 12.92,
                    ((rgb_arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_arr(lin: np.ndarray) -> np.ndarray:
    """Convert linear light (0..1) to sRGB (0..1), vectorized."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308,
                    12.92 * lin,
                    1.055 * np.power(lin, 1 / 2.4) - 0.055)


def srgb8_to_xyz(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 (0..255) to XYZ (D65)."""
    lin = srgb_to_linear_arr(rgb_u8.astype(np.float32) / 255.0)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151520, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    return M @ lin


def xyz_to_srgb8(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ (D65) to sRGB uint8 (0..255)."""
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32)
    lin = M @ xyz
    srgb = np.clip(linear_to_srgb_arr(lin), 0.0, 1.0)
    return srgb * 255.0


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to CIELAB (L*, a*, b*)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t):
        return np.where(t > (6 / 29) ** 3,
                        np.cbrt(t),
                        (1 / 3) * (29 / 6) ** 2 * t + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b], dtype=np.float32)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to XYZ."""
    L, a, b = lab
    Yn = 1.0;
    Xn = 0.95047;
    Zn = 1.08883
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t):
        return np.where(t > 6 / 29,
                        t ** 3,
                        3 * (6 / 29) ** 2 * (t - 4 / 29))

    x = Xn * finv(fx)
    y = Yn * finv(fy)
    z = Zn * finv(fz)
    return np.array([x, y, z], dtype=np.float32)


def rgb8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 to CIELAB."""
    return xyz_to_lab(srgb8_to_xyz(rgb_u8))


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to sRGB uint8."""
    return xyz_to_srgb8(lab_to_xyz(lab))


def relative_luminance(rgb_u8: Sequence[int]) -> float:
    """Compute perceptual relative luminance (Y) from sRGB uint8."""
    lin = srgb_to_linear_arr(np.array(rgb_u8, dtype=np.float32) / 255.0)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def Lstar_from_rgb(rgb_u8: Sequence[int]) -> float:
    """Extract L* from a color in sRGB uint8."""
    return float(np.clip(rgb8_to_lab(np.array(rgb_u8, dtype=np.float32))[0], 0, 100))


def deltaE_lab(rgb1_u8: Sequence[int], rgb2_u8: Sequence[int]) -> float:
    """Compute ΔE*ab between two sRGB uint8 colors."""
    return float(np.linalg.norm(rgb8_to_lab(np.array(rgb1_u8, dtype=np.float32)) -
                                rgb8_to_lab(np.array(rgb2_u8, dtype=np.float32))))


def rgb_to_hsv(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """Convert sRGB uint8 to (h, s, v) in [deg, 0..1, 0..1]."""
    rf, gf, bf = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    return h * 360.0, s, v

# ---------------------------
# Hue-limited correction (warm/brown bias)
# ---------------------------

def bias_warm_browns(rgb: np.ndarray) -> np.ndarray:
    """
    Apply a mild correction only for hues in the brown/orange range,
    if they appear to drift too pink (i.e., red dominance).
    Other hues are left nearly unaffected.

    Args:
      rgb: array-like length 3 (floats) in 0..255.

    Returns:
      corrected rgb as float array (0..255).
    """
    # Convert to HSV
    h, s, v = rgb_to_hsv(rgb.astype(int))
    if 10.0 <= h <= 45.0 and s > 0.25:
        r, g, b = rgb
        red_dom = (r - max(g, b)) / max(1, r)
        if red_dom > 0.10:
            # Slightly reduce brightness and saturation
            v = v * 0.93
            s = s * 0.97
    rf, gf, bf = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return np.clip(np.array([rf, gf, bf]) * 255.0, 0, 255)

# ---------------------------
# Mixing models
# ---------------------------

def mix_linear(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Simple linear-light weighted average mixing.
    This is a purely additive model (not physically accurate for pigments).
    """
    w = parts / np.sum(parts)
    lin = np.sum(srgb_to_linear_arr((base_rgbs / 255.0).T) * w, axis=1)
    return np.clip(255.0 * linear_to_srgb_arr(lin), 0, 255)


def mix_lab(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Perceptual mixing via averaging in Lab space.
    Useful fallback but often too light in shadows.
    """
    w = parts / np.sum(parts)
    labs = np.array([rgb8_to_lab(c) for c in base_rgbs], dtype=np.float32)
    lab = np.sum(labs.T * w, axis=1)
    return np.clip(lab_to_rgb8(lab), 0, 255)


def mix_subtractive(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Simple subtractive heuristic: 1 - Π(1 - c)^w per channel.
    Works okay in some midtones, but lacks realism near extremes.
    """
    w = parts / np.sum(parts)
    c = (base_rgbs / 255.0)
    res = 1.0 - np.prod((1.0 - c) ** w[:, None], axis=0)
    return np.clip(res * 255.0, 0, 255)


def mix_km_generic(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Basic KM-like mixing (Beer-Lambert per-channel) without strength correction.
    Useful for comparison with enhanced model.
    """
    w = parts / np.sum(parts)
    R = np.clip(base_rgbs / 255.0, 1e-4, 1.0)
    A = -np.log(R)
    A_mix = np.sum(A.T * w, axis=1)
    R_mix = np.exp(-A_mix)
    return np.clip(R_mix * 255.0, 0, 255)


def mix_km_strength(parts: np.ndarray,
                    base_rgbs: np.ndarray,
                    base_names: Sequence[str]) -> np.ndarray:
    """
    Tinting-strength aware empirical KM hybrid mixing.

    This model:
     - Scales raw parts by strength multipliers → “scaled parts”
     - Applies a diminishing-returns transform to avoid over-dominance
     - Normalizes to effective weights w
     - Combines absorption (K) and scattering proxies (S)
     - Adds cross-term interactions (beta)
     - Applies hue-limited warm-brown bias correction

    Args:
      parts: 1D array of raw parts (e.g. [2, 1, 0])
      base_rgbs: corresponding base pigment RGBs
      base_names: list of pigment names corresponding to base_rgbs

    Returns:
      rgb (floats) in 0..255 of the mixed color
    """
    raw = parts.astype(float)
    # Map to strength multipliers
    strength_arr = np.array([STRENGTH.get(n, 1.0) for n in base_names], dtype=float)
    scaled = raw * strength_arr

    # Diminishing-returns: prevents runaway dominance
    k = 0.3
    w_eff = scaled / (1.0 + k * scaled)

    total = np.sum(w_eff)
    if total > 0:
        w = w_eff / total
    else:
        w = w_eff

    # Absorption proxy (K) and scattering proxy (S)
    R = np.clip(base_rgbs / 255.0, 1e-4, 1.0)
    K = -np.log(R)
    S = 1.0 - R

    # Weighted sum K
    Kmix = np.sum(w[:, None] * K, axis=0)
    # Add empirical cross-terms
    beta = 0.12
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            diff = np.abs(K[i] - K[j])
            Kmix += beta * w[i] * w[j] * diff

    Smix = np.sum(w[:, None] * S, axis=0)
    gamma = 0.06
    Smix = Smix * (1.0 - gamma * np.abs(Smix - np.mean(Smix)))

    Rmix = np.exp(-Kmix)
    Rmix = Rmix * (1.0 - 0.08 * Smix)

    rgb = np.clip(Rmix * 255.0, 0, 255)
    return bias_warm_browns(rgb)


def mix_color(parts: np.ndarray,
              base_rgbs: np.ndarray,
              model: str,
              base_names: Sequence[str] = ()) -> np.ndarray:
    """
    Dispatch mixing by chosen model.
    For the “km” model, you must supply base_names so strength is aligned.
    """
    if model == "linear":
        return mix_linear(parts, base_rgbs)
    elif model == "lab":
        return mix_lab(parts, base_rgbs)
    elif model == "subtractive":
        return mix_subtractive(parts, base_rgbs)
    elif model == "km":
        return mix_km_strength(parts, base_rgbs, base_names)
    else:
        return mix_linear(parts, base_rgbs)

# ---------------------------
# Recipe enumeration / search
# ---------------------------

def enumerate_partitions_upto(total: int, k: int):
    """
    Yield all k-tuples of nonnegative ints summing ≤ total, not all zero.
    (Standard integer partition for search.)
    """
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
    Brute-force search for integer “parts” recipes approximating target_rgb, using ≤ max_parts.
    Score = ΔE + penalties for complexity.

    Args:
      target_rgb: desired color (float or int RGB).
      base_names: full list of base pigment names.
      max_parts, max_components: search limits.
      model: mixing model name (“km”, “linear”, “lab”, “subtractive”).
      prefer_simple_lambda_*: regularization strength.

    Returns:
      entries: list of (pigment_name, part_count)
      best_rgb: predicted mixed color
      best_err: ΔE error
    """
    N = len(base_names)
    base_rgbs_full = np.array([BASE_PALETTE[n] for n in base_names], dtype=float)
    target = np.array(target_rgb, dtype=float)

    best_score = float("inf")
    best_err = float("inf")
    best_entries: List[Tuple[str, int]] = []
    best_rgb = target.copy()

    max_components = max(1, min(max_components, N, (max_parts if max_parts > 0 else 1)))

    for m in range(1, max_components + 1):
        for combo in itertools.combinations(range(N), m):
            combo_names = [base_names[i] for i in combo]
            combo_rgbs = base_rgbs_full[list(combo)]
            for parts in enumerate_partitions_upto(max_parts, m):
                s = sum(parts)
                if s == 0:
                    continue
                parts_arr = np.array(parts, dtype=float)
                mix_rgb = mix_color(parts_arr, combo_rgbs, model, combo_names)
                err = deltaE_lab(mix_rgb, target)
                score = (err
                         + prefer_simple_lambda_components * (m - 1)
                         + prefer_simple_lambda_parts * (s / float(max_parts)))
                if score < best_score:
                    best_score = score
                    best_err = err
                    best_rgb = mix_rgb
                    best_entries = [(combo_names[i], int(parts[i])) for i in range(m) if parts[i] > 0]

    if len(best_entries) == 1:
        n, p = best_entries[0]
        best_entries = [(n, max(1, p))]

    return best_entries, best_rgb, best_err


def recipe_text(entries: List[Tuple[str, int]]) -> str:
    """Human-readable recipe string, e.g. “2 parts Yellow + 1 part Black”."""
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries]) if entries else "—"

# ---------------------------
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

# ==============================================================
# Centerline tracing (convert stencil to single-line SVG)
# ==============================================================

def _gray_int_to_hex(c: int) -> str:
    """200 -> '#C8C8C8' etc."""
    c = int(np.clip(c, 0, 255))
    h = f"{c:02X}"
    return f"#{h}{h}{h}"

def run_centerline_trace(args):
    """
    Generate a single-stroke centerline SVG from the final clean stencil outline,
    and overlay a grid in the SVG.

    Expects on `args`:
      - export_centerline_svg: bool
      - centerline_output: str
      - centerline_blur, centerline_threshold, centerline_otsu,
        centerline_dilate, centerline_simplify
      - outline_gray: np.ndarray (grayscale stencil to trace)
      - grid_step: int           # <-- used for SVG grid
      - (optional) grid_color: int (0..255)  # if not present, use 200
    """
    if not hasattr(args, "outline_gray") or args.outline_gray is None:
        print("⚠️  No stencil available for centerline tracing — skipped.")
        return

    gray = args.outline_gray
    img = cv2.GaussianBlur(gray, (args.centerline_blur, args.centerline_blur), 0) if args.centerline_blur > 0 else gray

    # Threshold
    if args.centerline_otsu:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        t = args.centerline_threshold if args.centerline_threshold is not None else 128
        _, bw = cv2.threshold(img, int(t), 255, cv2.THRESH_BINARY_INV)

    # Dilation (optional) to connect gaps
    if args.centerline_dilate > 0:
        kernel = np.ones((2, 2), np.uint8)
        bw = cv2.dilate(bw, kernel, iterations=args.centerline_dilate)

    # Skeletonize to centerlines
    skel = skeletonize((bw > 0).astype(np.uint8)).astype(np.uint8)

    # Extract contours as polylines
    contours = measure.find_contours(skel, 0.5)
    h, w = skel.shape

    # Prepare SVG
    dwg = svgwrite.Drawing(args.centerline_output, size=(w, h))
    # Optional: white background rect, if you prefer explicit white:
    # dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="#FFFFFF"))

    # ---------------------------
    # Add GRID (draw first so it's behind the strokes)
    # ---------------------------
    grid_step = int(getattr(args, "grid_step", 250))
    grid_gray = int(getattr(args, "grid_color", 200))
    grid_hex = _gray_int_to_hex(grid_gray)

    if grid_step > 0:
        # Vertical lines
        x = 0
        while x <= w:
            dwg.add(dwg.line(start=(x, 0), end=(x, h), stroke=grid_hex, stroke_width=0.5, opacity=0.7))
            x += grid_step
        # Horizontal lines
        y = 0
        while y <= h:
            dwg.add(dwg.line(start=(0, y), end=(w, y), stroke=grid_hex, stroke_width=0.5, opacity=0.7))
            y += grid_step

    # ---------------------------
    # Add CENTERLINES
    # ---------------------------
    for cnt in contours:
        pts = [(float(c[1]), float(c[0])) for c in cnt]
        if args.centerline_simplify > 0:
            epsilon = float(args.centerline_simplify)
            approx = cv2.approxPolyDP(np.array(pts, dtype=np.float32), epsilon, False)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        dwg.add(dwg.polyline(points=pts, fill="none", stroke="black", stroke_width=0.1))

    # Save SVG
    dwg.save()
    print(f"✅ Centerline SVG with grid saved: {args.centerline_output} (blur={args.centerline_blur}, simplify={args.centerline_simplify}, grid_step={grid_step})")

    # Optional vpype post-processing if installed
    try:
        import subprocess
        subprocess.run(
            ["vpype", "read", args.centerline_output, "linemerge", "linesimplify", "-t", "0.3", "reloop", "write", args.centerline_output],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ vpype optimization applied.")
    except Exception:
        print("ℹ️  vpype not available — saved raw SVG instead.")


def _auto_grid_step(img_width: int, min_cols: int) -> int:
    """
    Choose a pixel step so that floor(img_width / step) >= min_cols.
    Using step = floor(img_width / min_cols) guarantees >= min_cols columns.
    """
    step = max(1, img_width // max(1, int(min_cols)))
    return int(step)


# ---------------------------
# Main (DICT config, no argparse)
# ---------------------------
def main(config: dict | None = None):
    """
    Run the generator using a dict-based config.
    Pass only the keys you want to override; unspecified keys use DEFAULT_CONFIG.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    args = SimpleNamespace(**cfg)

    # Map the alias onto outline-mode (alias wins if provided)
    if args.sketch_style:
        args.outline_mode = {"old": "image", "new": "labels", "both": "both"}[args.sketch_style]

    # -------------------------
    # Load + optional resize for clustering
    # -------------------------
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # If grid_step is "auto" (or <=0 / None), compute from width
    if (getattr(args, "grid_step", None) in (None, "auto")) or (
            isinstance(args.grid_step, (int, float)) and args.grid_step <= 0):
        args.grid_step = _auto_grid_step(orig_w, getattr(args, "grid_min_cols", 5))

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

    # Optional clean-stencil post-processing of the outline
    if outline_gray is not None and args.apply_clean_stencil:
        outline_rgb = cv2.cvtColor(outline_gray, cv2.COLOR_GRAY2RGB)
        outline_rgb = _apply_clean_stencil_rgb(
            outline_rgb,
            block_size=int(args.stencil_block_size),
            C=int(args.stencil_C),
        )
        b_factor = _map_stencil_brightness_slider(float(args.stencil_brightness))
        s_factor = _map_stencil_sharpness_slider(float(args.stencil_sharpness))
        outline_rgb = _adjust_brightness_rgb(outline_rgb, b_factor)
        outline_rgb = _adjust_sharpness_rgb(outline_rgb, s_factor)
        outline_gray = cv2.cvtColor(outline_rgb, cv2.COLOR_RGB2GRAY)

    if outline_gray is not None:
        a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
        sketch_norm = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
        sketch_factor_rgb = ((1.0 - a) + a * sketch_norm)[..., None]
    else:
        sketch_factor_rgb = None

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

        # Page 2 (Outline page uses the same image as the underlay)
        if outline_gray is not None:
            fig = new_fig(A4_LANDSCAPE);
            ax = fig.add_subplot(111)

            # If the clean-stencil pipeline was applied, outline_gray already reflects it.
            if args.apply_clean_stencil:
                # Show the post-processed (clean) stencil with the grid
                ref_rgb = cv2.cvtColor(outline_gray, cv2.COLOR_GRAY2RGB)
                ref_with_grid = add_grid_to_rgb(ref_rgb, grid_step=args.grid_step, grid_color=200)
                mode_tag = (
                    "labels" if args.outline_mode == "labels"
                    else "combined" if args.outline_mode == "both"
                    else "image"
                )
                ax.imshow(ref_with_grid)
                ax.set_title(f"Clean Stencil Outline + Grid ({mode_tag}) (step={args.grid_step}px)", pad=2)
                ax.axis("off")
                pdf.savefig(fig, dpi=300);
                plt.close(fig)

            else:
                # Legacy behavior (no clean stencil): keep the old special page for image-edges;
                # otherwise show the raw outline/combined outline.
                if args.outline_mode == "image":
                    legacy_page = original_edge_sketch_with_grid(
                        img, grid_step=args.grid_step, grid_color=200,
                        canny_high_pct=float(args.edge_percentile)
                    )
                    ax.imshow(legacy_page, cmap='gray')
                    ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px)", pad=2)
                    ax.axis("off")
                    pdf.savefig(fig, dpi=300);
                    plt.close(fig)
                else:
                    ref_rgb = cv2.cvtColor(outline_gray, cv2.COLOR_GRAY2RGB)
                    ref_with_grid = add_grid_to_rgb(ref_rgb, grid_step=args.grid_step, grid_color=200)
                    ax.imshow(ref_with_grid)
                    ax.set_title(
                        f"Outline + Grid ({'labels' if args.outline_mode == 'labels' else 'combined'}) "
                        f"(step={args.grid_step}px)", pad=2
                    )
                    ax.axis("off")
                    pdf.savefig(fig, dpi=300);
                    plt.close(fig)

        # Step pages
        for title, idxs, frame in frames_to_emit:
            if sketch_factor_rgb is not None:
                frame_f = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
                composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
            else:
                composite_u8 = frame

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

        # -------------------------
        # Per-color pages (ordered to match Step sequence, big areas first within each step)
        # -------------------------
        if args.per_color_frames:
            # Ensure we have the multiply-underlay factor if an outline exists
            a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
            if sketch_factor_rgb is None and outline_gray is not None:
                s = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
                sketch_factor_rgb = ((1.0 - a) + a * s)[..., None]

            # Compute per-color pixel areas so we can sort large shapes first
            H, W = labels_full.shape
            area = np.array([(labels_full == i).sum() for i in range(args.colors)], dtype=np.int64)

            # Build the per-color order from frames_to_emit (Step pages) and sort within each step by area ↓
            per_color_order = []
            for _title, idxs, _frame in frames_to_emit:
                for idx in sorted(idxs, key=lambda i: -int(area[i])):
                    if idx not in per_color_order:
                        per_color_order.append(idx)

            # Fallback: append any colors that didn't appear in the step frames (should be rare)
            for i in range(args.colors):
                if i not in per_color_order:
                    per_color_order.append(i)

            # Render the per-color pages in the computed order
            prev_mask = np.zeros((H, W), dtype=bool)
            for i in per_color_order:
                curr_mask = (labels_full == i)
                frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                # Optional cumulative display of previously painted regions
                if args.per_color_cumulative and args.prev_alpha > 0 and prev_mask.any():
                    sel_prev = prev_mask
                    white_f = 255.0
                    prev_blend = ((1.0 - args.prev_alpha) * white_f +
                                  args.prev_alpha * pbn_image[sel_prev].astype(np.float32)).round().astype(np.uint8)
                    frame_rgb[sel_prev] = prev_blend

                # Current color regions
                frame_rgb[curr_mask] = pbn_image[curr_mask]

                # Multiply blend with outline (if present)
                if sketch_factor_rgb is not None:
                    frame_f = np.clip(frame_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                    composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                    composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
                else:
                    composite_u8 = frame_rgb

                # Grid overlay
                frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

                # Page layout
                fig = new_fig(A4_LANDSCAPE)
                gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
                axL = fig.add_subplot(gs[0, 0]);
                axR = fig.add_subplot(gs[0, 1])

                axL.imshow(frame_with_grid)
                axL.set_title((f"Per-Color • #{i + 1} + Grid "
                               f"{'(cumulative, prevα=' + str(args.prev_alpha) + ')' if args.per_color_cumulative else ''} "
                               f"{'(outline multiply underlay)' if sketch_factor_rgb is not None else ''}"), pad=2)
                axL.axis("off")

                draw_color_key(
                    axR, centroids, all_recipes, all_entries, BASE_PALETTE,
                    used_indices=[i],
                    title=f"Color Key • Color #{i + 1}",
                    tweaks=tweaks,
                    wrap_width=max(30, int(args.wrap * 0.7)),
                    show_components=not args.hide_components,
                    deltaEs=deltaEs,
                    left_pad=1.25, right_margin=0.18, text_gap=0.05,
                    approx_palette=approx_uint8
                )

                pdf.savefig(fig, dpi=300)
                plt.close(fig)

                # Update cumulative mask
                if args.per_color_cumulative:
                    prev_mask |= curr_mask
                else:
                    prev_mask[:] = False

        # Final page
        completed_with_grid = add_grid_to_rgb(pbn_image, grid_step=args.grid_step, grid_color=200)
        fig = new_fig(A4_LANDSCAPE); ax = fig.add_subplot(111)
        ax.imshow(completed_with_grid); ax.set_title("Completed — All Colors Applied + Grid", pad=2); ax.axis("off")
        pdf.savefig(fig, dpi=300); plt.close(fig)

    print(f"✅ Saved A4 landscape PDF to {args.pdf} (frame-mode={args.frame_mode}, outline={args.outline_mode})")

    # --------------------------------------------------
    # Optional: export centerline SVG for Inkscape plotting
    # --------------------------------------------------
    if args.export_centerline_svg:
        # Attach stencil to args so the helper can access it
        args.outline_gray = locals().get("outline_gray", None)
        run_centerline_trace(args)


if __name__ == "__main__":
    # ---------------------------
    # Base tube pigment palette
    # ---------------------------

    BASE_PALETTE = {
        # Existing colors
        "Titanium White": (218, 220, 224),
        "Lemon Yellow": (232, 206, 6),
        "Vermillion Red": (231, 44, 75),
        "Carmine": (213, 14, 33),
        "Ultramarine": (39, 51, 115),
        "Pthalo Green": (4, 95, 94),
        "Yellow Ochre": (200, 143, 16),
        "Lamp Black": (24, 14, 19),

        # New Schmincke Norma colors
        # "Cobalt Blue Hue": (45, 80, 170),  # mid-value, cooler than Ultramarine
        # "Payne's Grey": (45, 60, 80),  # deep bluish-grey
        # "Ivory Black": (26, 23, 24),  # slightly warmer than Lamp Black
        # "Indian Yellow": (230, 150, 20),  # transparent orange-yellow
        "Alizarin Crimson Hue": (120, 20, 30),  # deep cool red
        "Vandyke Brown": (45, 30, 20),  # dark warm brown
        "Indigo": (25, 40, 70),  # deep blue with grey undertone
        "Olive Green": (90, 100, 40),  # muted earthy green
        "Burnt Umber": (42, 17, 12),
        "Burnt Sienna": (80, 36, 25),
    }

    # Tinting strength multipliers: how strongly each pigment “tints” per unit part.
    # Values above 1.0 mean “stronger than average”; less than 1.0 means “weaker”.
    # You will need to calibrate these by observing real mixtures.
    STRENGTH = {
        "Titanium White": 1.0,
        "Lemon Yellow": 0.9,
        "Vermillion Red": 1.1,
        "Carmine": 1.2,
        "Ultramarine": 1.0,
        "Pthalo Green": 2.0,
        "Yellow Ochre": 0.7,
        "Lamp Black": 2.5,

        # New Schmincke Norma colors
        # "Cobalt Blue Hue": 0.9,  # moderate tinting, weaker than Phthalo
        # "Payne's Grey": 1.5,  # quite strong because of black + blue mix
        # "Ivory Black": 2.3,  # slightly less strong than Lamp Black
        # "Indian Yellow": 1.2,  # transparent and strong tint
        "Alizarin Crimson Hue": 1.3,  # deep tint, transparent
        "Vandyke Brown": 1.0,  # moderate tinting, earthy
        "Indigo": 1.4,  # strong tint due to dark synthetic pigments
        "Olive Green": 1.1,  # moderate tint, earthy but fairly strong
        "Burnt Umber": 1.0,
        "Burnt Sienna": 0.8,
    }

    # ---------------------------
    # Main CLI
    # ---------------------------

    # --- NEW: central config with previous CLI defaults ---
    DEFAULT_CONFIG = {
        "input": "pics/4.jpg",  # was a required CLI arg; override as needed
        "pdf": "paint_by_numbers_guide.pdf",
        "colors": 20,
        "resize": None,  # e.g. (W, H)
        "cluster_space": "lab",  # {"lab","rgb"}
        "palette": list(BASE_PALETTE.keys()),
        "components": 5,
        "max_parts": 10,
        "mix_model": "km",  # {"linear","lab","subtractive","km"}
        "frame_mode": "combined",  # {"classic","value5","both","combined"}
        "wrap": 55,
        # in DEFAULT_CONFIG
        "grid_step": "auto",  # was 250
        "grid_min_cols": 5,  # new: minimum boxes horizontally
        "edge_percentile": 90.0,
        "hide_components": False,
        "per_color_frames": True,
        "sketch_alpha": 0.55,
        "per_color_cumulative": True,
        "prev_alpha": 0.50,
        "min_region_px": 0,
        "min_region_pct": 0.0,
        "outline_mode": "image",  # {"image","labels","both"}
        "sketch_style": None,  # {"old","new","both"}; if set, overrides outline_mode
        "apply_clean_stencil": True,
        "stencil_brightness": 1.0,  # sliders 0..1 (mapped internally)
        "stencil_sharpness": 1.0,
        "stencil_block_size": 11,
        "stencil_C": 2,
        # --- Centerline trace (Inkscape plotting) options ---
        "export_centerline_svg": True,  # Run centerline trace after PDF generation
        "centerline_output": "centerline_output.svg",  # Output SVG filename (saved locally)
        "centerline_blur": 1,  # Gaussian blur amount (lower = more detail)
        "centerline_threshold": None,  # Manual threshold (0–255), None = use Otsu
        "centerline_otsu": True,  # Use Otsu automatic threshold
        "centerline_dilate": 0,  # Dilation iterations (connect broken lines)
        "centerline_simplify": 0,  # Polyline simplification epsilon (higher = smoother)
    }

    main()
