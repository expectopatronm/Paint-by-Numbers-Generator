# pencil_sketch.py
# Convert a photo to a paint-ready pencil sketch (OpenCV / NumPy)

import cv2
import numpy as np
from pathlib import Path
import argparse

# ==============================
# Utilities
# ==============================
def ensure_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def pad_white(img, pad=24):
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
    )

def im2float01(img_u8): return img_u8.astype(np.float32) / 255.0
def float01_to_u8(imgf): return (np.clip(imgf, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def size_norm(short_side, frac, odd=True, minv=3):
    k = max(minv, int(round(short_side * frac)))
    if odd: k |= 1
    return k

def lerp(a, b, t): return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def pad_from01(short_side, pad01):
    if pad01 <= 0: return 0
    return max(8, int(round(short_side * lerp(0.0, 0.05, pad01))))

def clahe_gray(gray_u8, clip=2.0, tiles=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(gray_u8)

def canny_from_gradients(gray_u8, low_high_ratio=0.35, high_pct=90):
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy).ravel()
    high = float(np.percentile(mag, high_pct))
    high = np.clip(high, 10, 255)
    low = max(5.0, high * low_high_ratio)
    return int(low), int(high)

def remove_small_components(bin_u8, min_area):
    # Drop tiny specks (prevents the "granite" look)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_u8, connectivity=8
    )
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def illumination_flatten(gray_u8, smin, strength01):
    # Estimate slow-varying illumination and flatten it
    if strength01 <= 0:
        return gray_u8
    sigma = smin * lerp(0.03, 0.08, strength01)   # big blur
    base = cv2.GaussianBlur(gray_u8, (0,0), sigma)
    g = im2float01(gray_u8); b = im2float01(base)
    flat = np.clip(g / (b + 1e-4), 0, 2.5)
    flat = flat / flat.max() if flat.max() > 0 else flat
    return float01_to_u8(flat)

def bilateral_edge_aware(gray_u8, strength01):
    # Edge-aware denoise that preserves contours, kills micro-texture
    if strength01 <= 0:
        return gray_u8
    # Map 0..1 -> modest but safe bilateral
    sigma_color = lerp(10, 80, strength01)
    sigma_space = lerp(3, 12, strength01)
    # d=0 lets OpenCV decide based on sigmas
    return cv2.bilateralFilter(gray_u8, d=0,
                               sigmaColor=sigma_color,
                               sigmaSpace=sigma_space)

def _auto_edge_mask(edge_strength_u8, target_fg=0.04, min_fg=0.01, max_fg=0.08, iters=8):
    """
    edge_strength_u8: 0..255 (from max(Canny, XDoG))
    Returns a 0/255 binary mask with foreground ratio near target_fg.
    """
    es = edge_strength_u8.astype(np.uint8)
    H, W = es.shape[:2]; N = H * W
    nz = es[es > 0]
    if nz.size == 0:
        return np.zeros_like(es, dtype=np.uint8)

    # binary search over percentile to hit desired fg coverage
    lo, hi = 50.0, 98.0
    best = None
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        T = np.percentile(nz, p)
        _, binm = cv2.threshold(es, int(T), 255, cv2.THRESH_BINARY)
        fg = np.count_nonzero(binm) / float(N)
        best = binm
        if fg < min_fg:  # too few edges -> lower threshold (lower percentile)
            lo = 45.0; hi = p
        elif fg > max_fg:  # too many edges -> raise threshold (higher percentile)
            lo = p; hi = 99.0
        else:
            break
    return best

# ==============================
# Pencil sketch (improved)
# ==============================
def pencil_readable_norm(
    bgr,
    # Balanced, line-forward defaults:
    sketchiness01=0.99,        # tones <- -> lines (more lines)
    softness01=0.1,           # crisp <- -> soft
    highlight_clip01=0.99,     # avoid highlight blowout
    edge_boost01=0.99,
    texture_suppression01=0.1,# kill micro-texture
    illumination01=0.1,       # mild light flatten
    despeckle01=0.25,
    stroke01=0.1,             # stroke width
    line_floor01=0.99,         # how dark lines are forced
    use_clahe=True,
    gamma_midtones=0.99        # NEW: darken midtones slightly before edges
):
    gray = ensure_gray(bgr)
    h, w = gray.shape[:2]; smin = min(h, w)

    # Illumination + mild CLAHE
    gray = illumination_flatten(gray, smin, illumination01)
    if use_clahe:
        gray = clahe_gray(gray, clip=lerp(1.3, 2.0, softness01), tiles=8)

    # Gentle pre-sharpen to bring features forward
    blur_sharp = cv2.GaussianBlur(gray, (0,0), smin*0.003)
    sharp = cv2.addWeighted(gray, 1.4, blur_sharp, -0.4, 0)

    # Edge-aware denoise (preserve contours)
    g_s = bilateral_edge_aware(sharp, texture_suppression01)
    gf  = im2float01(g_s)

    # Optional midtone gamma to avoid washed look
    gf = np.power(np.clip(gf, 0, 1), gamma_midtones)

    # Tones via dodge
    inv = 1.0 - gf
    sigma = smin * lerp(0.006, 0.016, softness01)
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma, sigmaY=sigma)
    denom = np.maximum(1e-4, 1.0 - blur)
    dodge = np.clip(gf / denom, 0, 1)
    dodge = np.minimum(dodge, lerp(0.90, 0.975, highlight_clip01))

    # Hybrid edges: Canny + XDoG
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
    xdog = 1.0 - (0.5 * (1 + np.tanh(phi * (dog - tau))))  # 0..1, high=edge
    xdog_u8 = float01_to_u8(xdog)

    # Auto-tuned threshold to target edge density
    edge_mix = cv2.max(can, xdog_u8)                 # 0..255 edge strength
    target_fg = lerp(0.025, 0.065, sketchiness01)    # more sketchy -> more edges kept
    edge_bin = _auto_edge_mask(edge_mix, target_fg=target_fg,
                               min_fg=0.015, max_fg=0.09)

    # Despeckle and stroke width
    if despeckle01 > 0:
        min_area = int(lerp(0, 0.0020, despeckle01) * (h*w))
        edge_bin = remove_small_components(edge_bin, min_area)
    k = size_norm(smin, lerp(0.0015, 0.0040, stroke01), odd=False, minv=2)
    edge_bin = cv2.dilate(edge_bin, np.ones((k, k), np.uint8), 1)

    # Hard line floor and multiplicative darkening near edges
    edge_mask = edge_bin.astype(np.float32) / 255.0
    ink_floor = 1.0 - (line_floor01 * edge_mask)     # ensures dark lines
    tone_edge_mul = 1.0 - 0.40 * edge_mask           # nudge tones darker at edges
    pencil = np.minimum(dodge * tone_edge_mul, ink_floor)

    return float01_to_u8(pencil)

# ==============================
# CLI
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", dest="out", type=str, default="pencil_readable.png")
    ap.add_argument("--pad", dest="pad01", type=float, default=0.9,
                    help="0..1 pad fraction relative to short side (visual)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    bgr = cv2.imread(str(in_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    h, w = bgr.shape[:2]; smin = min(h, w)

    # Run
    pencil_img = pencil_readable_norm(bgr)

    # Save (with white border for print)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad_px = pad_from01(smin, max(0.0, min(1.0, args.pad01)))
    cv2.imwrite(str(out_path), pad_white(pencil_img, pad_px))
    print("Saved:", out_path.resolve())

if __name__ == "__main__":
    main()
