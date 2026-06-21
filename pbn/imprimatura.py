from __future__ import annotations

import colorsys

import numpy as np
from sklearn.cluster import KMeans

from .color import rgb8_to_lab

def _map_pre_brighten_pct_to_factor(pct: float) -> float:
    """Clamp 0..100 → factor 1.00..2.00."""
    p = max(0.0, min(100.0, float(pct)))
    return 1.0 + (p / 100.0)


def _prev_highlight_rgb(args) -> np.ndarray | None:
    """
    Returns an RGB uint8 color for highlighting previous regions,
    or None to keep the existing 'use original colors' behavior.
    """
    mode = str(getattr(args, "prev_highlight_mode", "none")).lower()
    if mode == "neon_orange":
        return np.array([255, 90, 0], dtype=np.uint8)      # bright neon orange
    if mode == "neon_green":
        return np.array([57, 255, 20], dtype=np.uint8)     # bright neon green
    if mode == "custom":
        rgb = getattr(args, "prev_highlight_rgb", (255, 90, 0))
        return np.clip(np.array(rgb, dtype=np.int32), 0, 255).astype(np.uint8)
    return None


# ---------------------------
# Image-aware Imprimatura selection
# ---------------------------
def _rgb_to_lab_arr_u8(rows_u8: np.ndarray) -> np.ndarray:
    # rows_u8: N x 3 uint8
    labs = [rgb8_to_lab(r.astype(np.float32)) for r in rows_u8]
    return np.vstack(labs).astype(np.float32)

def _rgb_to_hsv_arr_u8(rows_u8: np.ndarray) -> np.ndarray:
    out = []
    for r,g,b in rows_u8:
        h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        out.append((h*360.0, s, v))
    return np.array(out, dtype=np.float32)

def _dominant_hue_deg(rows_u8: np.ndarray, *, min_sat=0.15, k=3) -> float | None:
    """Find a dominant hue (in degrees) from moderately saturated pixels."""
    hsv = _rgb_to_hsv_arr_u8(rows_u8)
    sel = hsv[hsv[:,1] >= float(min_sat)]
    if sel.size == 0:
        return None
    # k-means on hue on the unit circle
    H = sel[:,0]  # 0..360
    # map to 2D circle for clustering
    pts = np.stack([np.cos(np.deg2rad(H)), np.sin(np.deg2rad(H))], axis=1)
    km = KMeans(n_clusters=min(k, len(pts)), n_init=8, random_state=42)
    km.fit(pts)
    # pick the cluster with most members
    labels, counts = np.unique(km.labels_, return_counts=True)
    idx = int(labels[np.argmax(counts)])
    center = km.cluster_centers_[idx]
    ang = (np.rad2deg(np.arctan2(center[1], center[0])) + 360.0) % 360.0
    return float(ang)


def choose_imprimatura_target_from_image(rgb_image_u8: np.ndarray,
                                         *,
                                         mode: str = "match_light",  # "match_light" | "complement_dominant" | "neutral_warm"
                                         target_L: float = 50.0,
                                         chroma_s: float = 0.25) -> np.ndarray:
    """
    Returns target sRGB uint8 for imprimatura, computed from the image:
      - match_light: hue from highlight pixels (top ~20% L*)
      - complement_dominant: complement of dominant midtone hue
      - neutral_warm: fixed warm-neutral fallback if scene is ambiguous
    Value (L*) is clamped near a mid-tone, chroma modest.
    """
    H, W, _ = rgb_image_u8.shape
    flat = rgb_image_u8.reshape(-1,3).astype(np.uint8)

    # LAB for luminance segmentation
    labs = _rgb_to_lab_arr_u8(flat)
    Ls = labs[:,0]
    p80 = np.percentile(Ls, 80)
    p20, p80_mid = np.percentile(Ls, 20), np.percentile(Ls, 80)

    if mode == "match_light":
        hl = flat[Ls >= p80]
        hue = _dominant_hue_deg(hl, min_sat=0.10, k=3)
        if hue is None:
            mode = "neutral_warm"
    if mode == "complement_dominant":
        midmask = (Ls >= p20) & (Ls <= p80_mid)
        mids = flat[midmask] if np.count_nonzero(midmask) > 0 else flat
        hue_dom = _dominant_hue_deg(mids, min_sat=0.12, k=4)
        hue = ((hue_dom + 180.0) % 360.0) if hue_dom is not None else None
        if hue is None:
            mode = "neutral_warm"

    if mode == "neutral_warm":
        # ~brown-paper warm: ~35°–45° is yellow-orange
        hue = 38.0

    # Build an HSV with modest chroma; set V from target L* roughly
    # Approx map L*~V for mid ranges:
    v = np.clip((target_L/100.0)*0.92 + 0.06, 0.0, 1.0)
    s = float(np.clip(chroma_s, 0.05, 0.45))
    r,g,b = colorsys.hsv_to_rgb(hue/360.0, s, v)
    rgb = np.array([r,g,b])*255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

