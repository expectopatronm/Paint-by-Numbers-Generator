#!/usr/bin/env python3
"""
Paint-by-Numbers PDF generator (mixed-palette everywhere).

What this version guarantees
----------------------------
• The printed PBN imagery (Page 1 preview, all step/per-color frames, and the final completed page)
  uses ONLY the colors from your MIXED palette (derived from the limited tube colors).
• The Color Key’s left tile swatch shows the MIXED color (not the cluster target).
• Value-tweak suggestions are computed from MIXED colors (L* deviations within identical-recipe groups).
• Grouping into steps (classic/value5/combined) is based on MIXED colors.
• Keys have right-justified component swatches; per-row text wraps up to that row’s swatch block.
• Tight margins; frame pages devote ~75% width to the frame and ~25% to the key.

CLI
---
python make_pbn.py input.jpg --colors 25 --frame-mode combined --per-color-frames
"""

import argparse
import itertools
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
BASE_PALETTE = {
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
def srgb_to_linear_arr(rgb_arr):
    """Convert sRGB 0..255 to linear-light 0..1 (vectorized)."""
    rgb_arr = np.clip(rgb_arr / 255.0, 0, 1)
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055)/1.055) ** 2.4)

def linear_to_srgb_arr(lin):
    """Convert linear-light 0..1 to sRGB 0..1 (vectorized)."""
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055)

def srgb_to_xyz(rgb):
    """sRGB (0..255) to XYZ (D65)."""
    lin = srgb_to_linear_arr(rgb/255.0)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return M @ lin

def xyz_to_srgb(xyz):
    """XYZ (D65) to sRGB (0..255)."""
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])
    lin = M @ xyz
    srgb = np.clip(linear_to_srgb_arr(lin), 0, 1)
    return srgb * 255.0

def xyz_to_lab(xyz):
    """XYZ (D65) to CIELAB (L*, a*, b*)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0]/Xn, xyz[1]/Yn, xyz[2]/Zn
    def f(t):
        return np.where(t > (6/29)**3, np.cbrt(t), (1/3)*(29/6)**2 * t + 4/29)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.array([L, a, b])

def lab_to_xyz(lab):
    """CIELAB to XYZ (D65)."""
    L, a, b = lab
    Yn = 1.0
    Xn = 0.95047
    Zn = 1.08883
    fy = (L + 16)/116
    fx = fy + a/500
    fz = fy - b/200
    def finv(t):
        return np.where(t > 6/29, t**3, (3*(6/29)**2)*(t - 4/29))
    x = Xn * finv(fx)
    y = Yn * finv(fy)
    z = Zn * finv(fz)
    return np.array([x, y, z])

def rgb_to_lab(rgb):
    """sRGB (0..255) to CIELAB."""
    return xyz_to_lab(srgb_to_xyz(rgb))

def lab_to_rgb(lab):
    """CIELAB to sRGB (0..255)."""
    return xyz_to_srgb(lab_to_xyz(lab))

def relative_luminance(rgb):
    """Perceptual luminance Y from sRGB (0..255)."""
    lin = srgb_to_linear_arr(np.array(rgb, dtype=float))
    return 0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]

def Lstar_from_rgb(rgb):
    """Convenience: compute CIELAB L* from sRGB."""
    return float(np.clip(rgb_to_lab(np.array(rgb, float))[0], 0, 100))

# ---------------------------
# Mixing models
# ---------------------------
def mix_linear(parts, base_rgbs):
    """Linear-light additive mix of base colors."""
    w = parts / np.sum(parts)
    lin = np.sum(srgb_to_linear_arr(base_rgbs.T) * w, axis=1)
    return np.clip(255*linear_to_srgb_arr(lin), 0, 255)

def mix_lab(parts, base_rgbs):
    """Average colors in Lab space."""
    w = parts / np.sum(parts)
    labs = np.array([rgb_to_lab(c) for c in base_rgbs])
    lab = np.sum(labs.T * w, axis=1)
    return np.clip(lab_to_rgb(lab), 0, 255)

def mix_subtractive(parts, base_rgbs):
    """Simple subtractive mix (1 - Π(1 - c)^w)."""
    w = parts / np.sum(parts)
    c = (base_rgbs/255.0)
    res = 1.0 - np.prod((1.0 - c) ** w[:, None], axis=0)
    return np.clip(res*255.0, 0, 255)

def mix_km_generic(parts, base_rgbs):
    """Approximate Kubelka–Munk via Beer–Lambert in RGB (quick heuristic)."""
    w = parts / np.sum(parts)
    R = np.clip(base_rgbs/255.0, 1e-4, 1.0)
    A = -np.log(R)
    A_mix = np.sum(A.T * w, axis=1)
    R_mix = np.exp(-A_mix)
    return np.clip(R_mix*255.0, 0, 255)

def mix_color(parts, base_rgbs, model):
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
# Search helpers
# ---------------------------
def enumerate_partitions(total, k):
    """Yield k-tuples of nonnegative integers that sum to `total`."""
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in enumerate_partitions(total - i, k - 1):
            yield (i,) + rest

def deltaE_lab(rgb1, rgb2):
    """ΔE*ab between two sRGB colors."""
    return float(np.linalg.norm(rgb_to_lab(rgb1) - rgb_to_lab(rgb2)))

def integer_mix_best(target_rgb, base_names, max_parts=5, max_components=3, model="km"):
    """
    Brute-force an integer-part recipe that approximates target_rgb
    using up to `max_components` base colors and exactly `max_parts` parts.
    """
    base_rgbs_full = np.array([BASE_PALETTE[n] for n in base_names], dtype=float)
    target = np.array(target_rgb, dtype=float)

    best_err = float('inf')
    best_entries = []
    best_rgb = target

    N = len(base_names)
    max_components = min(max_components, N, max_parts if max_parts > 0 else 1)

    for m in range(1, max_components + 1):
        for combo in itertools.combinations(range(N), m):
            for parts in enumerate_partitions(max_parts, m):
                if sum(parts) != max_parts or all(p == 0 for p in parts):
                    continue
                parts_arr = np.array(parts, dtype=float)
                base_rgbs = base_rgbs_full[list(combo)]
                mix_rgb = mix_color(parts_arr, base_rgbs, model)
                err = deltaE_lab(mix_rgb, target)
                if err < best_err:
                    best_err = err
                    best_rgb = mix_rgb
                    best_entries = [(base_names[i], int(p)) for i, p in zip(combo, parts) if p > 0]

    if len(best_entries) == 1:
        n, _ = best_entries[0]
        best_entries = [(n, 1)]
    return best_entries, best_rgb, best_err

def recipe_text(entries):
    """Human-readable e.g. '2 parts Yellow + 1 part Black'."""
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries]) if entries else "—"

def rgb_to_hsv(rgb):
    """Return (h, s, v) with s,v ∈ [0..1]."""
    rgb = np.array(rgb, dtype=float) / 255.0
    mx = rgb.max()
    mn = rgb.min()
    diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == rgb[0]:
        h = (60 * ((rgb[1]-rgb[2]) / diff) + 360) % 360
    elif mx == rgb[1]:
        h = (60 * ((rgb[2]-rgb[0]) / diff) + 120) % 360
    else:
        h = (60 * ((rgb[0]-rgb[1]) / diff) + 240) % 360
    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v

# ---------------------------
# Grouping strategies (we'll call these with the MIXED palette)
# ---------------------------
def group_classic(palette):
    """Classic buckets by luminance/saturation: darks, mids, neutrals, highs."""
    n = len(palette)
    lums = np.array([relative_luminance(c) for c in palette])
    sats = np.array([rgb_to_hsv(c)[1] for c in palette])
    q25 = np.quantile(lums, 0.25)
    q80 = np.quantile(lums, 0.80)
    darks = [i for i in range(n) if lums[i] <= q25]
    highs = [i for i in range(n) if lums[i] >= q80]
    neutrals = [i for i in range(n) if (sats[i] <= 0.20) and (i not in highs)]
    mids = [i for i in range(n) if i not in darks and i not in highs and i not in neutrals]
    return {"darks": darks, "mids": mids, "neutrals": neutrals, "highs": highs}

def group_value5(palette):
    """Five value bands by luminance percentiles."""
    L = np.array([relative_luminance(c) for c in palette])
    q10, q25, q70, q85 = np.quantile(L, [0.10, 0.25, 0.70, 0.85])
    deep = [i for i in range(len(palette)) if L[i] <= q10]
    core = [i for i in range(len(palette)) if (q10 < L[i] <= q25)]
    mids = [i for i in range(len(palette)) if (q25 < L[i] <= q70)]
    half = [i for i in range(len(palette)) if (q70 < L[i] <= q85)]
    highs = [i for i in range(len(palette)) if L[i] > q85]
    return {"deep": deep, "core": core, "mids": mids, "half": half, "highs": highs}

def build_value_tweaks(palette, recipes_text, *, threshold=0.25):
    """
    Suggest tiny +/- value tweaks for colors that share the same recipe.
    """
    groups = {}
    for i, r in enumerate(recipes_text):
        groups.setdefault(r, []).append(i)

    tweaks = {i: "" for i in range(len(palette))}
    for _, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        Ls = np.array([Lstar_from_rgb(palette[i]) for i in idxs], float)
        L_mean = float(Ls.mean())
        for ci, L in zip(idxs, Ls):
            delta = L - L_mean
            if delta > threshold:
                tweaks[ci] = "Value tweak: + tiny White"
            elif delta < -threshold:
                tweaks[ci] = "Value tweak: + tiny Black"
            else:
                tweaks[ci] = "Value tweak: none (base)"
    return tweaks

# ---------------------------
# Image utilities
# ---------------------------
def ensure_gray(bgr):
    """Ensure single-channel uint8 grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def im2float01(img_u8): return img_u8.astype(np.float32) / 255.0
def float01_to_u8(imgf): return (np.clip(imgf, 0, 1) * 255.0 + 0.5).astype(np.uint8)
def lerp(a, b, t): return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def clahe_gray(gray_u8, clip=2.0, tiles=8):
    """CLAHE for local contrast normalization."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(gray_u8)

def canny_from_gradients(gray_u8, low_high_ratio=0.35, high_pct=90):
    """Set Canny thresholds from gradient percentiles for robustness."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy).ravel()
    high = float(np.percentile(mag, high_pct))
    high = np.clip(high, 10, 255)
    low = max(5.0, high * low_high_ratio)
    return int(low), int(high)

def remove_small_components(bin_u8, min_area):
    """Remove tiny blobs from a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def size_norm(short_side, frac, odd=True, minv=3):
    """Kernel/blur size proportional to the short image side."""
    k = max(minv, int(round(short_side * frac)))
    if odd: k |= 1
    return k

def illumination_flatten(gray_u8, smin, strength01):
    """Divide-by-blur illumination correction."""
    if strength01 <= 0:
        return gray_u8
    sigma = smin * lerp(0.03, 0.08, strength01)
    base = cv2.GaussianBlur(gray_u8, (0,0), sigma)
    g = im2float01(gray_u8)
    b = im2float01(base)
    flat = np.clip(g / (b + 1e-4), 0, 2.5)
    flat = flat / flat.max() if flat.max() > 0 else flat
    return float01_to_u8(flat)

def bilateral_edge_aware(gray_u8, strength01):
    """Bilateral filter that preserves edges while taming textures."""
    if strength01 <= 0:
        return gray_u8
    sigma_color = lerp(10, 80, strength01)
    sigma_space = lerp(3, 12, strength01)
    return cv2.bilateralFilter(gray_u8, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def _auto_edge_mask(edge_strength_u8, target_fg=0.04, min_fg=0.01, max_fg=0.08, iters=8):
    """Find a binary edge mask that lands in a desired foreground fraction."""
    es = edge_strength_u8.astype(np.uint8)
    H, W = es.shape[:2]
    N = H * W
    nz = es[es > 0]
    if nz.size == 0:
        return np.zeros_like(es, dtype=np.uint8)
    lo, hi = 50.0, 98.0
    best = None
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        T = np.percentile(nz, p)
        _, binm = cv2.threshold(es, int(T), 255, cv2.THRESH_BINARY)
        fg = np.count_nonzero(binm) / float(N)
        best = binm
        if fg < min_fg:
            lo = 45.0
            hi = p
        elif fg > max_fg:
            lo = p
            hi = 99.0
        else:
            break
    return best

# ---------------------------
# Pencil sketch pipeline
# ---------------------------
def pencil_readable_norm(
    bgr,
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
    gamma_midtones=0.99
):
    """Produce a clean pencil-like grayscale, suitable for grid overlay and printing."""
    gray = ensure_gray(bgr)
    h, w = gray.shape[:2]
    smin = min(h, w)

    gray = illumination_flatten(gray, smin, illumination01)
    if use_clahe:
        gray = clahe_gray(gray, clip=lerp(1.3, 2.0, softness01), tiles=8)

    blur_sharp = cv2.GaussianBlur(gray, (0,0), smin*0.003)
    sharp = cv2.addWeighted(gray, 1.4, blur_sharp, -0.4, 0)

    g_s = bilateral_edge_aware(sharp, texture_suppression01)
    gf  = im2float01(g_s)
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
        high_pct=int(lerp(90, 97, sketchiness01))
    )
    can = cv2.Canny(g_s, low, high)

    sigma1 = smin * lerp(0.003, 0.010, sketchiness01)
    sigma2 = sigma1 * 1.6
    g1 = cv2.GaussianBlur(gf, (0,0), sigma1)
    g2 = cv2.GaussianBlur(gf, (0,0), sigma2)
    dog = g1 - g2
    tau = lerp(0.9, 1.18, sketchiness01)
    phi = lerp(8.0, 22.0, sketchiness01 + edge_boost01*0.3)
    xdog = 1.0 - (0.5 * (1 + np.tanh(phi * (dog - tau))))
    xdog_u8 = float01_to_u8(xdog)

    edge_mix = cv2.max(can, xdog_u8)
    target_fg = lerp(0.025, 0.065, sketchiness01)
    edge_bin = _auto_edge_mask(edge_mix, target_fg=target_fg,
                               min_fg=0.015, max_fg=0.09)

    if despeckle01 > 0:
        min_area = int(lerp(0, 0.0020, despeckle01) * (h*w))
        edge_bin = remove_small_components(edge_bin, min_area)
    k = size_norm(smin, lerp(0.0015, 0.0040, stroke01), odd=False, minv=2)
    edge_bin = cv2.dilate(edge_bin, np.ones((k, k), np.uint8), 1)

    edge_mask = edge_bin.astype(np.float32) / 255.0
    ink_floor = 1.0 - (line_floor01 * edge_mask)
    tone_edge_mul = 1.0 - 0.40 * edge_mask
    pencil = np.minimum(dodge * tone_edge_mul, ink_floor)

    return float01_to_u8(pencil)

def original_edge_sketch_with_grid(img, grid_step=80, grid_color=200, **pencil_kwargs):
    """Pencil sketch + grid overlay from a PIL image."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sketch_u8 = pencil_readable_norm(bgr, **pencil_kwargs)  # grayscale uint8

    out = sketch_u8.copy()
    if grid_step and grid_step > 0:
        out[:, ::grid_step] = grid_color   # vertical lines
        out[::grid_step, :] = grid_color   # horizontal lines
    return Image.fromarray(out, mode="L")

def add_grid_to_rgb(arr, grid_step=80, grid_color=200):
    """Overlay a grid onto an RGB uint8 image array, non-destructively."""
    out = arr.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("add_grid_to_rgb expects an HxWx3 RGB array.")
    for x in range(0, arr.shape[1], grid_step):  # vertical
        out[:, x:x+1, :] = grid_color
    for y in range(0, arr.shape[0], grid_step):  # horizontal
        out[y:y+1, :, :] = grid_color
    return out

# ---------------------------
# Color key drawer
# ---------------------------
def draw_color_key(
    ax,
    target_palette,
    recipes,
    entries_per_color,
    base_palette,
    used_indices=None,
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

    def comp_names(entries): return [n for (n, _) in entries]
    max_n_comp = 0
    for ci in used_indices:
        max_n_comp = max(max_n_comp, len(comp_names(entries_per_color[ci])))

    gutter_right = XMAX - RIGHT_MARGIN
    band_left = gutter_right - (max_n_comp * swatch_step)

    if not no_band_bg:
        ax.add_patch(Rectangle((band_left - 0.0001, 0),
                               gutter_right - (band_left - 0.0001),
                               len(used_indices),
                               facecolor="white", edgecolor="none", zorder=0.5))

    for row_idx, ci in enumerate(used_indices):
        show_rgb = (approx_palette[ci] if approx_palette is not None else target_palette[ci])

        ax.add_patch(Rectangle((0, row_idx), 1, 1, color=(show_rgb/255), ec="k", lw=0.2))
        ax.text(0.5, row_idx + 0.5, f"{ci+1}",
                ha="center", va="center", fontsize=8, color="black",
                bbox=dict(facecolor=(1,1,1,0.45), edgecolor='none', boxstyle='round,pad=0.1'))

        Lstar = Lstar_from_rgb(show_rgb)
        tweak_str = f"  • L*={Lstar:.1f}"
        if deltaEs is not None:
            tweak_str += f"  • ΔE≈{deltaEs[ci]:.2f}"
        if tweaks.get(ci, ""):
            tweak_str += f"  • {tweaks[ci]}"
        text_str = f"{ci+1}: {recipes[ci]}{tweak_str}"

        row_comp_names = [n for n in base_order if n in comp_names(entries_per_color[ci])]
        n_comp = len(row_comp_names)
        row_start_x = gutter_right - (n_comp * swatch_step)

        avail_units = max(1.0, (row_start_x - text_gap) - LEFT_PAD)
        full_text_band = XMAX - RIGHT_MARGIN - LEFT_PAD
        frac = np.clip(avail_units / max(1.0, full_text_band), 0.2, 1.2)
        local_wrap = max(20, int(round(wrap_width * float(frac))))
        ax.text(LEFT_PAD, row_idx + 0.5, _tw.fill(text_str, width=local_wrap),
                va="center", fontsize=8, wrap=True, zorder=1.0)

        if show_components and n_comp > 0:
            for j, name in enumerate(row_comp_names):
                comp_rgb = np.array(base_palette[name]) / 255.0
                x = row_start_x + j * swatch_step
                ax.add_patch(Rectangle((x, row_idx), swatch_w, 1,
                                       color=comp_rgb, ec="k", lw=0.2, zorder=1.5))

    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, len(used_indices))
    ax.invert_yaxis()
    ax.axis("off")
    t = ax.set_title(title + "  (swatch = mixed color)", pad=3)
    t.set_wrap(True)

# ---------------------------
# Figure helper (tight page margins)
# ---------------------------
def new_fig(size):
    fig = plt.figure(figsize=size)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.965, wspace=0.02, hspace=0.02)
    return fig

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="A4 PDF with classic, value5, or combined 9-step frames.")
    parser.add_argument("input",
                        help="Input image file path")
    parser.add_argument("--pdf", default="paint_by_numbers_guide.pdf",
                        help="Output PDF path")
    parser.add_argument("--colors", type=int, default=30,
                        help="Number of colors (KMeans clusters)")
    parser.add_argument("--resize", type=int, nargs=2, default=[480, 480], metavar=("W", "H"),
                        help="Resize input to WxH for KMeans")
    parser.add_argument("--palette", nargs="*", default=list(BASE_PALETTE.keys()))
    parser.add_argument("--components", type=int, default=5,
                        help="Max components per mixed color")
    parser.add_argument("--max-parts", type=int, default=10,
                        help="Max parts per mixed color")
    parser.add_argument("--mix-model", choices=["linear","lab","subtractive","km"], default="km",
                        help="Mixing model for recipes")
    parser.add_argument("--frame-mode", choices=["classic","value5","both","combined"], default="combined",
                        help="Frame set: classic (4+complete), value5 (5), both (separate), or combined (interleaved 9-step)")
    parser.add_argument("--wrap", type=int, default=55,
                        help="Wrap width for color key text")
    parser.add_argument("--grid-step", type=int, default=80,
                        help="Grid spacing in pixels (0 = no grid)")
    parser.add_argument("--edge-percentile", type=float, default=85.0,
                        help="Edge detection percentile for sketch grid")
    parser.add_argument("--hide-components", action="store_true",
                        help="Do not show component swatches in color key")
    parser.add_argument("--per-color-frames", action="store_true",
                        help="If set, add a separate frame for each color (inserted before the completed page).")
    parser.add_argument(
        "--sketch-alpha", type=float, default=0.55,
        help="Opacity/strength of the pencil sketch underlay (0=no effect, 1=full sketch) on step/per-color pages."
    )
    parser.add_argument("--per-color-cumulative", action="store_true",
                        help="Per-color frames build cumulatively: prior colors appear at --prev-alpha; current color is 100%")
    parser.add_argument("--prev-alpha", type=float, default=0.75,
                        help="Opacity for all previous colors on cumulative per-color frames (0..1)")

    args = parser.parse_args()

    # Load + preprocess
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # Build BOTH: a plain pencil sketch (no grid) for underlay, and a sketch+grid for its own page
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sketch_gray = pencil_readable_norm(bgr)               # uint8 grayscale, NO GRID (for underlay)
    sketch_img_with_grid = original_edge_sketch_with_grid(img, grid_step=args.grid_step)  # for display page

    # KMeans on a smaller proxy for speed
    img_small = img.resize(tuple(args.resize), resample=Image.BILINEAR)
    data_small = np.array(img_small)
    Hs, Ws, _ = data_small.shape
    pixels_small = data_small.reshape((-1, 3))

    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=5).fit(pixels_small)
    labels_small = kmeans.labels_.reshape(Hs, Ws).astype(np.uint8)
    centroids = kmeans.cluster_centers_.astype(float)
    target_palette = centroids.astype(np.uint8)  # informational only (targets)

    # Build recipes against base palette (MIXED colors are what we use everywhere)
    names = args.palette
    all_entries, all_recipes, approx_rgbs, deltaEs = [], [], [], []
    for col in centroids:
        entries, approx_rgb, err = integer_mix_best(col, names,
                                                    max_parts=args.max_parts,
                                                    max_components=args.components,
                                                    model=args.mix_model)
        all_entries.append(entries)
        all_recipes.append(recipe_text(entries))
        approx_rgbs.append(np.array(approx_rgb, dtype=float))
        deltaEs.append(err)

    # MIXED palette (uint8)
    approx_uint8 = np.clip(np.rint(np.array(approx_rgbs)), 0, 255).astype(np.uint8)

    # Build the PBN image from the MIXED palette
    seg_mixed_small = approx_uint8[labels_small]                            # mixed colors at small res
    labels_orig = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_orig = np.array(labels_orig, dtype=np.uint8)
    pbn_image = Image.fromarray(seg_mixed_small).resize((orig_w, orig_h), resample=Image.NEAREST)
    pbn_image = np.array(pbn_image, dtype=np.uint8)                         # MIXED color image

    # Grouping (based on MIXED palette)
    classic = group_classic(approx_uint8)
    value5  = group_value5(approx_uint8)

    # Orders
    classic_order = [
        ("Frame 1 – Shadows / Dark Blocks", classic["darks"]),
        ("Frame 2 – Mid-tone Masses",       classic["mids"]),
        ("Frame 3 – Neutrals / Background", classic["neutrals"]),
        ("Frame 4 – Highlights",            classic["highs"]),
        ("Frame 5 – Completed",             list(range(args.colors))),
    ]
    value5_order = [
        ("Value A – Deep Shadows (lowest ~10%)", value5["deep"]),
        ("Value B – Core Shadows (to ~25%)",     value5["core"]),
        ("Value C – Midtones (to ~70%)",         value5["mids"]),
        ("Value D – Half-Lights (to ~85%)",      value5["half"]),
        ("Value E – Highlights (top ~15%)",      value5["highs"]),
    ]

    def frames_from_order(order):
        """Materialize (title, indices, frame_image) from an order list."""
        frames = []
        for title, idxs in order:
            if len(idxs) == 0:
                continue
            mask = np.isin(labels_orig, np.array(idxs, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames.append((title, idxs, frame_img))
        return frames

    # Build frame sequence(s)
    if args.frame_mode == "combined":
        painted = set()
        def remaining(idx_list): return [i for i in idx_list if i not in painted]
        sequence = [
            ("Step 1 – Deep Shadows",          value5["deep"]),
            ("Step 2 – Core Shadows",          value5["core"]),
            ("Step 3 – Shadows / Dark Blocks", classic["darks"]),
            ("Step 4 – Value Midtones",        value5["mids"]),
            ("Step 5 – Mid-tone Masses",       classic["mids"]),
            ("Step 6 – Neutrals / Background", classic["neutrals"]),
            ("Step 7 – Half-Lights",           value5["half"]),
            ("Step 8 – Highlights",            value5["highs"]),
            ("Step 9 – Highlight Accents",     classic["highs"]),
        ]
        frames_combined = []
        for title, idxs in sequence:
            rem = remaining(idxs)
            painted.update(rem)
            if not rem:
                continue
            mask = np.isin(labels_orig, np.array(rem, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames_combined.append((title, rem, frame_img))
        frames_to_emit = frames_combined
    elif args.frame_mode == "classic":
        frames_to_emit = frames_from_order(classic_order)
    elif args.frame_mode == "value5":
        frames_to_emit = frames_from_order(value5_order)
    else:  # "both"
        frames_to_emit = frames_from_order(classic_order) + frames_from_order(value5_order)

    # Value tweaks from the MIXED palette
    tweaks = build_value_tweaks(approx_uint8, all_recipes, threshold=0.25)

    # ---------------------------
    # PDF assembly (tight margins)
    # ---------------------------
    A4_LANDSCAPE = (11.69, 8.27)

    with (PdfPages(args.pdf) as pdf):
        # Page 1: Overview (MIXED preview + Key)
        fig = new_fig(A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1.0, 1.55], figure=fig, wspace=0.01, hspace=0.03)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])
        ax1.imshow(img)
        t = ax1.set_title("Original", pad=2)
        t.set_wrap(True)
        ax1.axis("off")

        # MIXED PBN preview (no grid here for cleanliness)
        ax2.imshow(pbn_image)
        t = ax2.set_title(
            f"Paint by Numbers ({args.colors} colors) • model={args.mix_model} • max parts={args.max_parts}",
            pad=2
        )
        t.set_wrap(True)
        ax2.axis("off")

        # Key (swatch = MIXED color)
        draw_color_key(
            ax3, target_palette, all_recipes, all_entries, BASE_PALETTE,
            used_indices=list(range(args.colors)),
            title="Color Key • All Clusters",
            tweaks=tweaks,
            wrap_width=int(args.wrap * 1.5),
            show_components=not args.hide_components,
            deltaEs=deltaEs,
            swatch_step=0.55, swatch_w=0.55, right_margin=0.10, left_pad=1.10,
            no_band_bg=True, text_gap=0.03,
            approx_palette=approx_uint8,
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 2: Edge sketch (+grid) for reference only
        fig = new_fig(A4_LANDSCAPE)
        ax = fig.add_subplot(111)
        ax.imshow(sketch_img_with_grid, cmap='gray')
        t = ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px)", pad=2)
        t.set_wrap(True)
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # --- Prepare multiply-underlay once ---
        # Normalize sketch and turn into a multiplicative factor controlled by alpha:
        # factor = 1 - a*(1 - sketch_norm) ∈ [1-a, 1]; whites do nothing, dark lines darken.
        a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
        sketch_norm = np.clip(sketch_gray.astype(np.float32) / 255.0, 0.0, 1.0)
        sketch_factor = (1.0 - a) + a * sketch_norm  # shape HxW
        sketch_factor_rgb = sketch_factor[..., None]  # broadcast to channels

        # Frame pages (¾ image / ¼ key) — apply sketch under ALL areas (paint + unpainted)
        for title, idxs, frame in frames_to_emit:
            frame_f = np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0)
            # Multiply blend (no grid in the underlay)
            composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
            composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)

            # Grid on top (crisp)
            frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

            # Layout
            fig = new_fig(A4_LANDSCAPE)
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
            axL = fig.add_subplot(gs[0, 0])
            axR = fig.add_subplot(gs[0, 1])

            axL.imshow(frame_with_grid)
            t = axL.set_title(title + " + Grid (sketch multiply underlay)", pad=2)
            t.set_wrap(True)
            axL.axis("off")

            draw_color_key(axR, target_palette, all_recipes, all_entries, BASE_PALETTE,
                           used_indices=idxs,
                           title=f"Color Key • {title}",
                           tweaks=tweaks,
                           wrap_width=max(30, int(args.wrap * 0.7)),
                           show_components=not args.hide_components,
                           deltaEs=deltaEs,
                           left_pad=1.25, right_margin=0.18, text_gap=0.05,
                           approx_palette=approx_uint8)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        # Optional: per-color pages (¾ / ¼) — cumulative build option supported
        if args.per_color_frames:
            # Make sure we have the multiply underlay factor ready (from earlier section)
            a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
            # If you already computed sketch_gray and sketch_factor_rgb above, this is a no-op; otherwise compute here:
            try:
                sketch_factor_rgb
            except NameError:
                # Build a clean sketch (no grid) and turn into multiply factor
                rgb = np.array(img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                sketch_gray = pencil_readable_norm(bgr)  # grayscale uint8, no grid
                sketch_norm = np.clip(sketch_gray.astype(np.float32) / 255.0, 0.0, 1.0)
                sketch_factor = (1.0 - a) + a * sketch_norm
                sketch_factor_rgb = sketch_factor[..., None]

            prev_alpha = float(np.clip(args.prev_alpha, 0.0, 1.0))

            H, W = labels_orig.shape
            prev_mask = np.zeros((H, W), dtype=bool)

            for i in range(args.colors):
                curr_mask = (labels_orig == i)

                # Start from white canvas
                frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                # 1) Lay down all PREVIOUS colors at prev_alpha (if any)
                if prev_alpha > 0 and prev_mask.any():
                    # Linear blend toward the actual paint color, from white:
                    # prev_mix = (1 - prev_alpha)*white + prev_alpha*paint
                    sel_prev = prev_mask
                    white_f = 255.0
                    prev_blend = ((1.0 - prev_alpha) * white_f +
                                  prev_alpha * pbn_image[sel_prev].astype(np.float32)).round().astype(np.uint8)
                    frame_rgb[sel_prev] = prev_blend

                # 2) Put CURRENT color opaquely (100%)
                frame_rgb[curr_mask] = pbn_image[curr_mask]

                # 3) Sketch underlay EVERYWHERE via multiply (keeps grid crisp later)
                frame_f = np.clip(frame_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)

                # 4) Grid on top
                frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

                # 5) Page layout
                fig = new_fig(A4_LANDSCAPE)
                gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
                axL = fig.add_subplot(gs[0, 0])
                axR = fig.add_subplot(gs[0, 1])

                axL.imshow(frame_with_grid)
                t = axL.set_title(
                    (f"Per-Color • #{i + 1} + Grid "
                     f"{'(cumulative, prevα=' + str(prev_alpha) + ')' if args.per_color_cumulative else ''} "
                     "(sketch multiply underlay)"),
                    pad=2
                )
                t.set_wrap(True)
                axL.axis("off")

                draw_color_key(axR, target_palette, all_recipes, all_entries, BASE_PALETTE,
                               used_indices=[i],
                               title=f"Color Key • Color #{i + 1}",
                               tweaks=tweaks,
                               wrap_width=max(30, int(args.wrap * 0.7)),
                               show_components=not args.hide_components,
                               deltaEs=deltaEs,
                               left_pad=1.25, right_margin=0.18, text_gap=0.05,
                               approx_palette=approx_uint8)
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

                # 6) Update cumulative mask if the mode is on
                if args.per_color_cumulative:
                    prev_mask |= curr_mask
                else:
                    # If not cumulative, keep prev_mask empty so only curr color shows each page
                    prev_mask[:] = False

        # Final page: completed with grid (no sketch underlay here, to show true colors)
        completed_with_grid = add_grid_to_rgb(pbn_image, grid_step=args.grid_step, grid_color=200)
        fig = new_fig(A4_LANDSCAPE)
        ax = fig.add_subplot(111)
        ax.imshow(completed_with_grid)
        t = ax.set_title("Completed — All Colors Applied + Grid", pad=2)
        t.set_wrap(True)
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"✅ Saved A4 landscape PDF to {args.pdf} (frame-mode={args.frame_mode})")

if __name__ == "__main__":
    main()
