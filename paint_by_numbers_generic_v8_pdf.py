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
import os
import subprocess
import itertools
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.random import default_rng
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import textwrap as _tw
import colorsys
import mixbox as _mixbox

import svgwrite
from skimage.morphology import skeletonize
from skimage import measure


# Cache latents for speed
_MIXBOX_LATENTS: dict[tuple[int,int,int], list[float]] = {}

def _latent_for_rgb_u8(rgb_u8) -> list[float]:
    key = (int(rgb_u8[0]), int(rgb_u8[1]), int(rgb_u8[2]))
    z = _MIXBOX_LATENTS.get(key)
    if z is None:
        z = _mixbox.rgb_to_latent(key)  # returns list/tuple of length mixbox.LATENT_SIZE
        _MIXBOX_LATENTS[key] = z
    return z

def mix_learned(parts: np.ndarray, base_rgbs: np.ndarray) -> np.ndarray:
    """
    Mix N colors exactly as documented by Mixbox:
    - convert RGB -> latent
    - weighted average latents
    - latent -> RGB
    Returns sRGB uint8 floats (0..255).
    """
    parts = np.asarray(parts, dtype=float)
    if parts.sum() <= 0:
        return base_rgbs[0].astype(float)

    w = parts / parts.sum()

    z_mix = None
    for wi, rgb in zip(w, base_rgbs):
        if wi <= 0:
            continue
        zi = _latent_for_rgb_u8(rgb)
        if z_mix is None:
            z_mix = [wi * v for v in zi]
        else:
            for i in range(len(z_mix)):
                z_mix[i] += wi * zi[i]

    r, g, b = _mixbox.latent_to_rgb(z_mix)  # uint8s
    return np.array([float(r), float(g), float(b)], dtype=float)


# =========================
# PAM (CLARA-only, auto-sized)
# =========================

# --- perceptual distance helpers ---
def _deltaE76_pairwise(X_lab, Y_lab):
    X = np.asarray(X_lab, np.float32); Y = np.asarray(Y_lab, np.float32)
    return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2))


def _ciede2000_to_one(X_lab, y_lab):
    X = np.asarray(X_lab, np.float32); y = np.asarray(y_lab, np.float32)
    L1,a1,b1 = X[:,0], X[:,1], X[:,2]
    L2,a2,b2 = float(y[0]), float(y[1]), float(y[2])
    C1 = np.sqrt(a1*a1 + b1*b1); C2 = np.sqrt(a2*a2 + b2*b2)
    Cbar = 0.5*(C1 + C2); Cbar7 = Cbar**7
    G = 0.5*(1 - np.sqrt(Cbar7 / (Cbar7 + 25**7 + 1e-12)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1); C2p = np.sqrt(a2p*a2p + b2*b2)
    def _hp(ap, b):
        ang = (np.degrees(np.arctan2(b, ap)) % 360.0)
        return np.where((ap==0)&(b==0), 0.0, ang)
    h1p = _hp(a1p, b1); h2p = _hp(a2p, b2)
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp-360, dhp)
    dhp = np.where(dhp < -180, dhp+360, dhp)
    dhp = np.where((C1p*C2p)==0, 0.0, dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
    Lbarp = 0.5*(L1+L2); Cbarp = 0.5*(C1p+C2p)
    sumh = h1p + h2p; diffh = np.abs(h1p - h2p)
    hbarp = np.where((C1p*C2p)==0, h1p+h2p,
                     np.where(diffh <= 180, 0.5*sumh,
                              np.where(sumh < 360, 0.5*(sumh+360), 0.5*(sumh-360))))
    T = 1 - 0.17*np.cos(np.radians(hbarp-30)) + 0.24*np.cos(np.radians(2*hbarp)) \
          + 0.32*np.cos(np.radians(3*hbarp+6)) - 0.20*np.cos(np.radians(4*hbarp-63))
    dtheta = 30*np.exp(-((hbarp-275)/25)**2)
    Rc = 2*np.sqrt((Cbarp**7) / (Cbarp**7 + 25**7 + 1e-12))
    Sl = 1 + (0.015*(Lbarp-50)**2)/np.sqrt(20 + (Lbarp-50)**2)
    Sc = 1 + 0.045*Cbarp
    Sh = 1 + 0.015*Cbarp*T
    Rt = -np.sin(np.radians(2*dtheta))*Rc
    kl=kc=kh=1.0
    return np.sqrt((dLp/(kl*Sl))**2 + (dCp/(kc*Sc))**2 + (dHp/(kh*Sh))**2 + Rt*(dCp/(kc*Sc))*(dHp/(kh*Sh)))


def _deltaE00_pairwise(X_lab, Y_lab):
    X = np.asarray(X_lab, np.float32); Y = np.asarray(Y_lab, np.float32)
    out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
    for j in range(Y.shape[0]): out[:, j] = _ciede2000_to_one(X, Y[j])
    return out


def _pairwise_sqdist(X, Y=None):
    X = np.asarray(X, np.float32); Y = X if Y is None else np.asarray(Y, np.float32)
    x2 = np.sum(X*X, axis=1)[:, None]; y2 = np.sum(Y*Y, axis=1)[None, :]
    return x2 + y2 - 2.0*(X @ Y.T)


# --- core PAM on a (small) sample (for CLARA) ---
def pam_cluster_metric(feats, k, *, metric="deltaE00", max_iter=100, random_state=42):
    rng = default_rng(random_state)
    n = feats.shape[0]

    # metric lambdas
    if metric == "deltaE00":
        pairwise = _deltaE00_pairwise; to_one = _ciede2000_to_one
    elif metric == "deltaE76":
        pairwise = _deltaE76_pairwise; to_one = lambda X,y: np.sqrt(np.sum((X-y)**2, axis=1, dtype=np.float32))
    else:  # L2
        pairwise = lambda X,Y: np.sqrt(np.maximum(_pairwise_sqdist(X,Y), 0.0))
        to_one   = lambda X,y: np.sqrt(np.maximum(np.sum((X-y)**2, axis=1, dtype=np.float32), 0.0))

    # k-medoids++ init
    idx0 = int(rng.integers(0, n)); med_idx = [idx0]
    d = to_one(feats, feats[idx0]); d = np.maximum(d, 0.0)
    for _ in range(1, k):
        w = d**2; probs = w / (w.sum() + 1e-12)
        nxt = int(rng.choice(n, p=probs)); med_idx.append(nxt)
        d = np.minimum(d, to_one(feats, feats[nxt]))
    med_idx = np.array(med_idx, int)

    # precompute distances to medoids
    D = pairwise(feats, feats[med_idx])             # (n,k)
    nearest_j = np.argmin(D, axis=1); nearest_d = D[np.arange(n), nearest_j]
    D_mask = D.copy(); D_mask[np.arange(n), nearest_j] = np.inf
    second_d = D_mask.min(axis=1); second_j = D_mask.argmin(axis=1)

    prev_cost = float(nearest_d.sum())
    for _ in range(int(max_iter)):
        improved = False
        non_meds = np.setdiff1d(np.arange(n), med_idx, assume_unique=True)
        for h in non_meds:
            dh = to_one(feats, feats[h])  # (n,)
            best_delta_h = 0.0
            for mpos, m in enumerate(med_idx):
                in_m  = (nearest_j == mpos)
                gain  = np.sum(np.minimum(dh[in_m], second_d[in_m]) - nearest_d[in_m])
                not_m = ~in_m
                drop  = np.sum(np.minimum(nearest_d[not_m], dh[not_m]) - nearest_d[not_m])
                delta = gain + drop
                if delta < best_delta_h:
                    best_delta_h = delta
                    best_swap = (mpos, h, dh)
            if best_delta_h < -1e-8:
                mpos, hidx, dh = best_swap
                improved = True
                med_idx[mpos] = hidx
                D[:, mpos] = dh
                new_nearest_j = np.argmin(D, axis=1); new_nearest_d = D[np.arange(n), new_nearest_j]
                D_mask = D.copy(); D_mask[np.arange(n), new_nearest_j] = np.inf
                second_d = D_mask.min(axis=1); second_j = D_mask.argmin(axis=1)
                nearest_j, nearest_d = new_nearest_j, new_nearest_d
        cost = float(nearest_d.sum())
        if (not improved) or cost >= prev_cost - 1e-8: break
        prev_cost = cost

    return nearest_j.astype(np.uint8), med_idx


# --- blockwise assignment (never allocate n×k for the full image) ---
def _assign_blockwise(feats, medoids, metric="deltaE00", block_size=120000):
    if metric == "deltaE00":
        pair = _deltaE00_pairwise
    elif metric == "deltaE76":
        pair = _deltaE76_pairwise
    else:
        def pair(X,Y): return np.sqrt(np.maximum(_pairwise_sqdist(X,Y), 0.0))
    n = feats.shape[0]; labels = np.empty(n, np.uint8)
    for s in range(0, n, int(block_size)):
        e = min(n, s + int(block_size))
        D = pair(feats[s:e], medoids)
        labels[s:e] = np.argmin(D, axis=1)
        del D
    return labels


# --- tiny auto-sizer (no manual knobs) ---
def _avail_ram_bytes_fallback():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return int(8e9)  # fallback 8 GB


def _metric_cost_factor(metric: str) -> float:
    m = (metric or "deltaE00").lower()
    if m == "l2": return 1.0
    if m == "deltae76": return 1.2
    return 1.8  # ΔE00 heavier


def _auto_pam_clara_params(n: int, k: int, *, metric="deltaE00") -> dict:
    import math
    n = int(max(1, n)); k = int(max(1, k))
    f = _metric_cost_factor(metric)

    # sample size: >= 300*k, grows with k and log10(n), scaled by metric cost
    base = int(round(k * (50 + 10 * math.log10(max(10, n)))))
    s_lo = max(300 * k, k)
    s_hi = min(int(0.05 * n), 200_000)
    sample_size = max(s_lo, min(int(base * f), s_hi))

    # restarts: 1..5 based on n (+1 if k big)
    if n <= 100_000:        num_samples = 1
    elif n <= 1_000_000:    num_samples = 3
    else:                   num_samples = 5
    if k >= 48:             num_samples = min(5, num_samples + 1)

    # validation set: 1.5× sample, min 50k, ≤ n
    val_size = min(n, max(int(1.5 * sample_size), 50_000))

    # block size for full assignment: ~15% of available RAM
    avail = _avail_ram_bytes_fallback()
    budget = max(256 * 1024 * 1024, int(0.15 * avail))   # ≥256MB
    blk = int(budget // max(1, 4 * k))                   # float32
    block_size = int(max(50_000, min(blk, 600_000)))

    return {
        "sample_size": sample_size,
        "num_samples": num_samples,
        "val_size": val_size,
        "block_size": block_size,
    }

# --- CLARA wrapper: sample->PAM->validate->assign full ---
def pam_clara(feats, pixels_rgb_u8, k, *, metric="deltaE00",
              sample_size=80000, num_samples=3, val_size=120000,
              block_size=120000, random_state=42, max_iter=100):
    rng = default_rng(random_state)
    n = feats.shape[0]
    sample_size = min(int(sample_size), n)
    val_size = min(int(val_size), n)

    best_cost = np.inf
    best_meds_global = None

    # validation subset (once)
    val_idx = np.sort(rng.choice(n, size=val_size, replace=False))
    feats_val = feats[val_idx]

    # distance selector for validation
    Dv = (_deltaE00_pairwise if metric=="deltaE00"
          else _deltaE76_pairwise if metric=="deltaE76"
          else (lambda X,Y: np.sqrt(np.maximum(_pairwise_sqdist(X,Y), 0.0))))

    for _ in range(int(num_samples)):
        samp_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        feats_s = feats[samp_idx]

        _, med_s = pam_cluster_metric(
            feats_s, k, metric=metric, max_iter=max_iter,
            random_state=int(rng.integers(0, 10_000_000))
        )
        meds_global = samp_idx[med_s]

        # validate: total nearest distance on held-out set
        val_cost = float(np.min(Dv(feats_val, feats[meds_global]), axis=1).sum())
        if val_cost < best_cost:
            best_cost = val_cost
            best_meds_global = meds_global

    # final assignment over ALL points (blockwise)
    labels = _assign_blockwise(feats, feats[best_meds_global],
                               metric=metric, block_size=block_size)
    return labels, best_meds_global


def _choose_scale(longest: int, *,
                  ok_min: int,
                  target: int,
                  choices=(2, 3, 4)) -> int | None:
    """
    Pick the smallest scale in `choices` that brings the longest side to at least ok_min,
    without needing to exceed `target` (we'll clamp down afterwards if we overshoot).
    Returns None if no upscaling is needed.
    """
    if ok_min <= longest < target:
        return None    # already acceptable per user rule
    if longest >= target:
        return None    # already >= 3000 -> don't upscale
    # Need to upscale from below ok_min
    for s in sorted(choices):
        if longest * s >= ok_min:
            return s
    return max(choices) if choices else None  # fallback


def _maybe_upscale_with_realesrgan(input_path: str,
                                   *,
                                   enable: bool,
                                   ok_min: int,
                                   target: int | None,
                                   bin_path: str,
                                   model_dir: str | None = None,
                                   model_name: str | None = None,
                                   choices=(2, 3, 4)) -> str:
    """
    If enabled and the longest side is < ok_min, upscale with Real-ESRGAN
    using realesrgan-ncnn-vulkan.

    Args:
        input_path: Path to input image.
        enable: Whether to perform upscaling.
        ok_min: Minimum acceptable longest dimension (px) before upscaling.
        target: Target longest side (unused here, kept for API compatibility).
        bin_path: Path to realesrgan-ncnn-vulkan executable.
        model_dir: Directory containing model .bin/.param files.
        model_name: Model name (e.g. "realesrgan-x4plus").
        choices: Allowed upscale factors (default: 2, 3, 4).
    """
    if not enable:
        return input_path

    try:
        with Image.open(input_path) as im:
            w, h = im.size
    except Exception as e:
        print(f"⚠️  Could not open input for pre-check: {e}. Skipping upscale.")
        return input_path

    longest = max(w, h)

    if longest >= ok_min:
        print(f"ℹ️  Upscale check: longest={longest}px ≥ {ok_min}px → no upscale.")
        return input_path

    # Determine scale factor
    scale = None
    for s in sorted(choices):
        if longest * s >= ok_min:
            scale = s
            break
    if scale is None:
        scale = max(choices)

    root, _ = os.path.splitext(os.path.abspath(input_path))
    up_path = f"{root}.upx{scale}.png"

    # Build the command
    cmd = [str(bin_path), "-s", str(scale), "-i", input_path, "-o", up_path]
    if model_dir:
        cmd += ["-m", str(model_dir)]
    if model_name:
        cmd += ["-n", str(model_name)]

    try:
        print(f"⏫ Running Real-ESRGAN: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"⚠️  Real-ESRGAN failed ({e}). Using original image.")
        return input_path

    try:
        with Image.open(up_path) as up:
            uw, uh = up.size
        print(f"✅ Upscaled to {uw}×{uh} (longest={max(uw, uh)}px).")
    except Exception:
        print(f"✅ Upscaled (file saved): {up_path}")

    return up_path


def _entries_to_vec(entries: List[Tuple[str,int]], base_order: List[str]) -> np.ndarray:
    v = np.zeros(len(base_order), dtype=int)
    for n, p in entries:
        v[base_order.index(n)] = int(p)
    return v

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(b >= a) and np.any(b > a)

def _extras(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(b - a, 0, None)

@dataclass
class BuildPolicy:
    max_deltaE: float = 8.0
    max_added_parts: int = 6
    max_added_pigments: int = 2
    max_new_pigments: int = 1
    min_added_fraction: float = 0.05
    max_chain_depth: int = 4
    parent_choice: str = "min_added_parts"   # "min_deltaE" | "min_new_pigments" | "min_added_parts"

def plan_build_order_configurable(parts_mat: np.ndarray,
                                  approx_rgb_uint8: np.ndarray,
                                  base_order: List[str],
                                  policy: BuildPolicy):
    """
    Neutral build-on planner. Returns:
      order, step_note, parent_map, extras_label
    """
    C, _ = parts_mat.shape
    totals = parts_mat.sum(axis=1)

    candidates = {i: [] for i in range(C)}
    for p in range(C):
        for c in range(C):
            if p == c:
                continue
            a, b = parts_mat[p], parts_mat[c]
            if not _dominates(a, b):
                continue
            dE = deltaE_lab(approx_rgb_uint8[p], approx_rgb_uint8[c])
            if dE > policy.max_deltaE:
                continue
            e = _extras(a, b)
            total_added = int(e.sum())
            if total_added > policy.max_added_parts:
                continue
            idxs = np.nonzero(e)[0].tolist()
            added_pigs = len(idxs)
            if added_pigs > policy.max_added_pigments:
                continue
            parent_support = set(np.nonzero(a)[0].tolist())
            new_pigs = len([i for i in idxs if i not in parent_support])
            if new_pigs > policy.max_new_pigments:
                continue
            if totals[p] > 0 and (total_added / max(1, totals[p])) < policy.min_added_fraction:
                continue

            if policy.parent_choice == "min_deltaE":
                score = (dE, total_added, added_pigs, new_pigs)
            elif policy.parent_choice == "min_new_pigments":
                score = (new_pigs, added_pigs, total_added, dE)
            else:
                score = (total_added, added_pigs, dE, new_pigs)

            label = " + ".join([f"{int(e[i])}× {base_order[i]}" for i in idxs])
            candidates[c].append((score, p, label))

    parent = {i: None for i in range(C)}
    extras_label = {i: "" for i in range(C)}
    for c in range(C):
        if candidates[c]:
            candidates[c].sort(key=lambda t: t[0])
            _, p, label = candidates[c][0]
            parent[c] = p
            extras_label[c] = label

    # chain depth
    depth = {i: 0 for i in range(C)}
    def _depth(i):
        if parent[i] is None: return 0
        if depth[i] != 0: return depth[i]
        d = 1 + _depth(parent[i]); depth[i] = d; return d
    for i in range(C): _depth(i)
    for i in range(C):
        if depth[i] > policy.max_chain_depth:
            parent[i] = None
            extras_label[i] = ""

    # children buckets
    children = {i: [] for i in range(C)}
    for c, p in parent.items():
        if p is not None: children[p].append(c)
    for p in children:
        children[p].sort(key=lambda c: (parts_mat[c].sum() - parts_mat[p].sum()))

    bases = [i for i in range(C) if parent[i] is None]
    bases.sort(key=lambda i: -int(totals[i]))

    order = []
    step_note = {}
    q = bases[:]
    while q:
        i = q.pop(0)
        order.append(i)
        if parent[i] is None:
            step_note[i] = f"Base mix for color #{i+1}"
        else:
            step_note[i] = f"Color #{i+1} = Color #{parent[i]+1} + {extras_label[i]}"
        q.extend(children[i])

    return order, step_note, parent, extras_label


def _levels_from_parents(parent: dict[int, int|None]) -> dict[int,int]:
    lvl = {}
    def rec(i):
        if i in lvl: return lvl[i]
        p = parent.get(i, None)
        lvl[i] = 0 if p is None else 1 + rec(p)
        return lvl[i]
    for i in parent.keys():
        rec(i)
    return lvl

def draw_build_graph_page(approx_rgb_uint8: np.ndarray,
                          parent: dict[int, int|None],
                          extras_label: dict[int,str],
                          *,
                          title: str = "Build Dependency Graph (Neutral)"):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_subplot(111)
    ax.set_title(title, pad=8, fontsize=12)
    ax.axis("off")

    levels = _levels_from_parents(parent)
    max_level = max(levels.values()) if levels else 0

    by_level = {}
    for i, lv in levels.items():
        by_level.setdefault(lv, []).append(i)
    for lv in by_level:
        by_level[lv].sort()

    pos = {}
    for lv in range(max_level + 1):
        nodes = by_level.get(lv, [])
        n = max(1, len(nodes))
        xs = np.linspace(0.12, 0.88, n)
        y = 0.88 - lv * (0.75 / max(1, max_level))
        for k, i in enumerate(nodes):
            pos[i] = (xs[k], y)

    # edges
    for c, p in parent.items():
        if p is None: continue
        x0, y0 = pos[p]; x1, y1 = pos[c]
        ax.annotate("", xy=(x1, y1-0.02), xytext=(x0, y0+0.02),
                    arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.85))
        xm, ym = (x0 + x1)/2, (y0 + y1)/2
        if extras_label.get(c):
            ax.text(xm, ym, extras_label[c], fontsize=6, ha="center", va="center", color="dimgray")

    # nodes
    for i, (x, y) in pos.items():
        rgb = (approx_rgb_uint8[i] / 255.0).tolist()
        circ = plt.Circle((x, y), 0.03, color=rgb, ec="black", lw=0.6)
        ax.add_patch(circ)
        ax.text(x, y + 0.055, f"{i+1}", ha="center", va="center", fontsize=8, fontweight="bold")

    ax.text(0.015, 0.03, "Arrow: add parts to reach child • Edge label: parts× pigment",
            fontsize=8, ha="left", va="bottom", transform=ax.transAxes)
    return fig


def _combo_key(combo_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(combo_names)

@lru_cache(maxsize=100_000)
def _cached_mix_color(model: str,
                      combo_key: Tuple[str, ...],
                      parts_tuple: Tuple[int, ...],
                      use_tinting_strength: bool) -> Tuple[float, float, float]:
    """
    Exact memoization for mixing. Cache key includes whether tinting strength is used.
    """
    base_rgbs = np.array([BASE_PALETTE[n] for n in combo_key], dtype=float)
    parts_arr = np.array(parts_tuple, dtype=float)

    if model == "km":
        if use_tinting_strength:
            rgb = mix_km_strength(parts_arr, base_rgbs, combo_key)
        else:
            rgb = mix_km_generic(parts_arr, base_rgbs)
    else:
        rgb = mix_color(parts_arr, base_rgbs, model, base_names=combo_key)

    return float(rgb[0]), float(rgb[1]), float(rgb[2])


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
    elif model == "learned":                      # <--- NEW
        return mix_learned(parts, base_rgbs)
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
        use_tinting_strength: bool = True,   # <--- add this
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:

    """
    Search for a small-integer “parts” recipe that approximates a target color using
    a subset of the given base pigments. The search enumerates all pigment
    combinations of size 1..max_components and all non-negative integer part
    allocations whose sum ≤ max_parts (excluding the all-zero vector).

    Scoring
    -------
    score = ΔE*ab(mix, target)
            + λ_components * (num_components - 1)
            + λ_parts * (sum(parts) / max_parts)

    Mixing models
    -------------
    model="km" uses the strength-aware hybrid KM if use_tinting_strength=True;
    otherwise falls back to a generic KM (no tinting multipliers).
     Other options ("linear", "lab", "subtractive") dispatch to their respective mixers.

    Caching
    -------
    Calls to the mixer are memoized via `_cached_mix_color(model, combo_names, parts)`
    so repeated (combo, parts) during the brute-force search do not recompute.
    The cache key is independent of `target_rgb`. Each process has its own cache.

    Parameters
    ----------
    target_rgb : Sequence[float]
        Target color in sRGB 0..255.
    base_names : Sequence[str]
        Names of the available base pigments (order defines the parts vector
        alignment). Must match keys in `BASE_PALETTE` (and optionally `STRENGTH`).
    max_parts : int, default 10
        Maximum allowed sum(parts) for any candidate recipe.
    max_components : int, default 3
        Maximum number of distinct pigments allowed in a recipe.
    model : {"km","linear","lab","subtractive"}, default "km"
        Mixing model to use (see above).
    prefer_simple_lambda_components : float, default 0.03
        Regularization weight penalizing more components.
    prefer_simple_lambda_parts : float, default 0.01
        Regularization weight penalizing larger total parts.

    Returns
    -------
    entries : List[Tuple[str, int]]
        The best recipe as (pigment_name, part_count) with zero-parts removed.
        Single-pigment results are normalized to at least 1 part.
    best_rgb : np.ndarray, shape (3,)
        The predicted mixed sRGB (0..255 floats) for the best recipe.
    best_err : float
        ΔE*ab between `best_rgb` and `target_rgb`.

    Notes
    -----
    - Relies on module-level `BASE_PALETTE` (sRGB for each pigment) and, for
      model="km", `STRENGTH` (tinting multipliers). Missing strengths default to 1.0.
    - The hybrid KM model is not strictly associative; “build-on-it” physical
      workflows are encouraged for painting, but the search always mixes from base
      pigments to keep predictions consistent.
    - Complexity: O(Σ_{m=1..max_components} C(N, m) * P(m, max_parts)), where
      P enumerates integer partitions with sum ≤ max_parts. The LRU cache reduces
      constant factors substantially in practice.
    """
    N = len(base_names)
    target = np.array(target_rgb, dtype=float)

    best_score = float("inf")
    best_err = float("inf")
    best_entries: List[Tuple[str, int]] = []
    best_rgb = target.copy()

    max_components = max(1, min(max_components, N, (max_parts if max_parts > 0 else 1)))

    for m in range(1, max_components + 1):
        for combo in itertools.combinations(range(N), m):
            combo_names = [base_names[i] for i in combo]
            for parts in enumerate_partitions_upto(max_parts, m):
                s = sum(parts)
                if s == 0:
                    continue
                parts_arr = np.array(parts, dtype=float)
                mix_rgb = np.array(
                    _cached_mix_color(
                        model,
                        _combo_key(combo_names),
                        tuple(int(x) for x in parts_arr),
                        bool(use_tinting_strength),
                    ),
                    dtype=float
                )
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


# --- worker-visible placeholders; will be populated at runtime ---
BASE_PALETTE = None
STRENGTH = None
USE_TINTING_STRENGTH = False  # <--- add

def _init_worker(palette_dict, strength_dict, use_tinting_strength_flag: bool):
    """
    Install palette/strength in worker globals.
    """
    global BASE_PALETTE, STRENGTH, USE_TINTING_STRENGTH
    BASE_PALETTE = palette_dict
    STRENGTH = strength_dict
    USE_TINTING_STRENGTH = bool(use_tinting_strength_flag)



def _recipe_worker(color_rgb_list,
                   base_names,
                   max_parts,
                   max_components,
                   model,
                   lambda_components,
                   lambda_parts,
                   use_tinting_strength_flag):   # <--- add this
    """
    Runs integer_mix_best for a single centroid color.
    Kept pickle-friendly for ProcessPoolExecutor.
    """
    color = np.array(color_rgb_list, dtype=float)
    entries, approx_rgb, err = integer_mix_best(
        color,
        base_names,
        max_parts=max_parts,
        max_components=max_components,
        model=model,
        prefer_simple_lambda_components=lambda_components,
        prefer_simple_lambda_parts=lambda_parts,
        use_tinting_strength=bool(use_tinting_strength_flag),   # <--- forward
    )
    # Return plain python types to be extra pickle-friendly
    return {
        "entries": [(str(n), int(p)) for (n, p) in entries],
        "approx_rgb": [float(x) for x in approx_rgb],
        "err": float(err),
    }


def _map_pre_brighten_pct_to_factor(pct: float) -> float:
    """Clamp 0..100 → factor 1.00..2.00."""
    p = max(0.0, min(100.0, float(pct)))
    return 1.0 + (p / 100.0)


# ---------------------------
# Main (DICT config, no argparse)
# ---------------------------
def main(config: dict | None = None):
    """
    Run the generator using a dict-based config.
    Pass only the keys you want to override; unspecified keys use DEFAULT_CONFIG.
    """
    global BASE_PALETTE, STRENGTH, USE_TINTING_STRENGTH

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    args = SimpleNamespace(**cfg)

    # --- NEW: pre-upscale gate ---
    args.input = _maybe_upscale_with_realesrgan(
        args.input,
        enable=bool(getattr(args, "enable_upscale", True)),
        ok_min=int(getattr(args, "upscale_ok_min_long", 2500)),
        target=None,
        bin_path=str(getattr(args, "realesrgan_bin", "realesrgan-ncnn-vulkan")),
        model_dir=str(getattr(args, "realesrgan_model_dir", "")),
        model_name=str(getattr(args, "realesrgan_model_name", "")),
        choices=tuple(getattr(args, "realesrgan_scale_choices", (2, 3, 4))),
    )

    # set global toggle from config (existing)
    USE_TINTING_STRENGTH = bool(args.use_tinting_strength)

    # Map the alias onto outline-mode (existing)
    if args.sketch_style:
        args.outline_mode = {"old": "image", "new": "labels", "both": "both"}[args.sketch_style]

    # -------------------------
    # Load + optional resize for clustering (unchanged, but now uses the possibly upscaled path)
    # -------------------------
    img = Image.open(args.input).convert("RGB")

    # --- PRE-BRIGHTEN: apply a 0..100% increase (mapped to factor 1.00..2.00)
    try:
        pct = float(getattr(args, "pre_brighten_pct", 0))
        if pct > 0:
            # use helper if added:
            factor = _map_pre_brighten_pct_to_factor(pct)
            factor = 1.0 + (max(0.0, min(100.0, pct)) / 100.0)
            img = ImageEnhance.Brightness(img).enhance(factor)
            print(f"✨ Pre-brighten applied: +{pct:.1f}% (factor {factor:.3f})")
        else:
            print("✨ Pre-brighten skipped (pre_brighten_pct=0).")
    except Exception as e:
        print(f"⚠️  Pre-brighten failed ({e}) — continuing with unmodified image.")

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

    if args.cluster_algo == "kmeans":
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

    elif args.cluster_algo == "pam":
        # Use Lab features for perceptual ΔE metrics
        if args.pam_metric in ("deltaE00", "deltaE76"):
            if args.cluster_space != "lab":
                print("⚠️ pam_metric is perceptual; forcing cluster_space='lab' for PAM.")

            def _rgbrows_to_labrows(arr_uint8):
                out = np.empty((arr_uint8.shape[0], 3), dtype=np.float32)
                for i, (r, g, b) in enumerate(arr_uint8):
                    out[i] = rgb8_to_lab(np.array([r, g, b], np.float32))
                return out

            feats = _rgbrows_to_labrows(pixels_small.astype(np.uint8))
        else:
            feats = pixels_small.astype(np.float32)  # L2 in RGB

        n = feats.shape[0]
        k = int(args.colors)

        auto = _auto_pam_clara_params(n, k, metric=str(args.pam_metric))
        sample_size = auto["sample_size"]
        num_samples = auto["num_samples"]
        val_size = auto["val_size"]
        block_size = auto["block_size"]

        print(f"ℹ️ PAM/CLARA(auto): n={n}, k={k}, metric={args.pam_metric}, "
              f"sample={sample_size}, samples={num_samples}, val={val_size}, block={block_size}")

        labels_flat, medoid_idx = pam_clara(
            feats, pixels_small, k,
            metric=str(args.pam_metric),
            sample_size=sample_size,
            num_samples=num_samples,
            val_size=val_size,
            block_size=block_size,
            random_state=int(getattr(args, 'pam_random_state', 42)),
            max_iter=100,
        )
        labels_small = labels_flat.reshape(Hs, Ws).astype(np.uint8)

        # Medoids are real pixels → use their RGB for downstream swatches/recipes
        medoid_rgbs = pixels_small[medoid_idx]
        centroids = np.clip(np.rint(medoid_rgbs), 0, 255).astype(np.uint8)

    # Upsample labels to full res
    labels_full = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_full = np.array(labels_full, dtype=np.uint8)

    # Region cleanup
    labels_full = cleanup_label_regions(
        labels_full,
        min_region_px=max(0, int(args.min_region_px)),
        min_region_pct=max(0.0, float(args.min_region_pct)),
    )

    # Build recipes (MIXED palette) — parallelized
    names = args.palette
    all_entries, all_recipes, approx_rgbs, deltaEs = [], [], [], []

    centroids_list = [c.astype(float).tolist() for c in centroids]

    if args.parallel and len(centroids_list) > 1:
        max_workers = int(args.workers) if args.workers else (os.cpu_count() or 1)
        # Guard against silly over-commit (optional, but polite)
        max_workers = max(1, min(max_workers, len(centroids_list)))

        tasks = []
        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(BASE_PALETTE, STRENGTH, bool(args.use_tinting_strength)),  # <--- add flag here
        ) as ex:
            for c in centroids_list:
                fut = ex.submit(
                    _recipe_worker,
                    c,
                    names,
                    int(args.max_parts),
                    int(args.components),
                    str(args.mix_model),
                    float(args.get("prefer_simple_lambda_components", 0.03) if isinstance(args, dict) else getattr(args,
                                                                                                                   "prefer_simple_lambda_components",
                                                                                                                   0.03) if hasattr(
                        args, "prefer_simple_lambda_components") else 0.03),
                    float(args.get("prefer_simple_lambda_parts", 0.01) if isinstance(args, dict) else getattr(args,
                                                                                                              "prefer_simple_lambda_parts",
                                                                                                              0.01) if hasattr(
                        args, "prefer_simple_lambda_parts") else 0.01),
                    bool(args.use_tinting_strength),  # <--- add here
                )

                tasks.append(fut)

            # preserve original centroid order as results return
            results = [t.result() for t in tasks]

        for res in results:
            entries = res["entries"]
            approx = np.array(res["approx_rgb"], dtype=float)
            err = res["err"]
            all_entries.append(entries)
            all_recipes.append(recipe_text(entries))
            approx_rgbs.append(approx)
            deltaEs.append(err)
    else:
        # Fallback: single-core
        for col in centroids.astype(np.float32):
            entries, approx_rgb, err = integer_mix_best(
                col,
                names,
                max_parts=args.max_parts,
                max_components=args.components,
                model=args.mix_model,
                use_tinting_strength=bool(args.use_tinting_strength),  # <--- add
                # If your integer_mix_best signature already includes these
                # defaults, you can omit them:
                # prefer_simple_lambda_components=0.03,
                # prefer_simple_lambda_parts=0.01,
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

        # ---- Optional: Build dependency graph page (neutral) ----
        if args.build_graph_page:
            base_order = args.palette
            parts_mat = np.stack([_entries_to_vec(e, base_order) for e in all_entries], axis=0)

            policy = BuildPolicy(
                max_deltaE=8.0,
                max_added_parts=6,
                max_added_pigments=2,
                max_new_pigments=1,
                min_added_fraction=0.05,
                max_chain_depth=4,
                parent_choice="min_added_parts",
            )

            _order, _steps, parent_map, extras_label = plan_build_order_configurable(
                parts_mat, approx_uint8, base_order, policy
            )

            fig = draw_build_graph_page(
                approx_uint8, parent_map, extras_label,
                title="Build Dependency Graph (Neutral, additive steps)"
            )
            pdf.savefig(fig, dpi=300);
            plt.close(fig)

        # Page 3 (Outline page uses the same image as the underlay)
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
        # Per-color pages (original order only)
        # -------------------------
        if args.per_color_frames:
            # Ensure multiply-underlay factor if outline exists
            a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
            if sketch_factor_rgb is None and outline_gray is not None:
                s = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
                sketch_factor_rgb = ((1.0 - a) + a * s)[..., None]

            # Old area-based order derived from step frames (unchanged)
            H, W = labels_full.shape
            area = np.array([(labels_full == i).sum() for i in range(args.colors)], dtype=np.int64)
            per_color_order = []
            for _title, idxs, _frame in frames_to_emit:
                for idx in sorted(idxs, key=lambda i: -int(area[i])):
                    if idx not in per_color_order:
                        per_color_order.append(idx)
            for i in range(args.colors):
                if i not in per_color_order:
                    per_color_order.append(i)

            # Neutral label for right-side note (optional)
            step_note_map = {i: f"Color #{i + 1}" for i in per_color_order}

            # Render per-color pages in the decided (original) order
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

                # Right side: color key & neutral note
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
                axR.text(0.05, 0.05, step_note_map.get(i, ""), fontsize=8, transform=axR.transAxes)

                pdf.savefig(fig, dpi=300);
                plt.close(fig)

                # Update cumulative mask for next page
                if args.per_color_cumulative:
                    prev_mask |= curr_mask
                else:
                    prev_mask[:] = False

        # -------------------------
        # Optional: Workflow sequence pages (EXTRA PAGES, does not touch per-color pages)
        # -------------------------
        if args.build_workflow_pages:
            base_order = args.palette
            parts_mat = np.stack([_entries_to_vec(e, base_order) for e in all_entries], axis=0)

            policy = BuildPolicy(
                max_deltaE=float(args.build_max_deltaE),
                max_added_parts=int(args.build_max_added_parts),
                max_added_pigments=int(args.build_max_added_pigments),
                max_new_pigments=int(args.build_max_new_pigments),
                min_added_fraction=float(args.build_min_added_fraction),
                max_chain_depth=int(args.build_max_chain_depth),
                parent_choice=str(args.build_parent_choice),
            )

            workflow_order, step_note_map, parent_map, _extras_label = plan_build_order_configurable(
                parts_mat, approx_uint8, base_order, policy
            )

            # --- Gate: only build workflow pages if a majority are derivable from the graph
            total_colors = parts_mat.shape[0]
            derivable_count = sum(1 for i in range(total_colors) if parent_map.get(i) is not None)
            majority_ok = (derivable_count / max(1, total_colors)) >= 0.5

            if not majority_ok:
                print(
                    f"ℹ️  Skipping workflow pages: only {derivable_count}/{total_colors} colors "
                    f"({100.0 * derivable_count / max(1, total_colors):.1f}%) were derivable from the graph."
                )
            else:
                print(
                    f"✅ Building workflow pages: {derivable_count}/{total_colors} colors "
                    f"({100.0 * derivable_count / max(1, total_colors):.1f}%) are derivable."
                )

                H, W = labels_full.shape
                prev_mask = np.zeros((H, W), dtype=bool)

                for rank, i in enumerate(workflow_order, start=1):
                    curr_mask = (labels_full == i)
                    frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                    # Reuse the same cumulative knob for now
                    if args.per_color_cumulative and args.prev_alpha > 0 and prev_mask.any():
                        sel_prev = prev_mask
                        white_f = 255.0
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

                    frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)

                    fig = new_fig(A4_LANDSCAPE)
                    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig, wspace=0.02)
                    axL = fig.add_subplot(gs[0, 0]);
                    axR = fig.add_subplot(gs[0, 1])

                    axL.imshow(frame_with_grid)
                    axL.set_title((f"Workflow • Step {rank}: Color #{i + 1} + Grid "
                                   f"{'(cumulative)' if args.per_color_cumulative else ''} "
                                   f"{'(outline multiply underlay)' if sketch_factor_rgb is not None else ''}"), pad=2)
                    axL.axis("off")

                    draw_color_key(
                        axR, centroids, all_recipes, all_entries, BASE_PALETTE,
                        used_indices=[i],
                        title=f"Color Key • Workflow Step {rank}",
                        tweaks=tweaks,
                        wrap_width=max(30, int(args.wrap * 0.7)),
                        show_components=not args.hide_components,
                        deltaEs=deltaEs,
                        left_pad=1.25, right_margin=0.18, text_gap=0.05,
                        approx_palette=approx_uint8
                    )
                    axR.text(0.05, 0.05, step_note_map.get(i, ""), fontsize=8, transform=axR.transAxes)

                    pdf.savefig(fig, dpi=300);
                    plt.close(fig)

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
    BASE_PALETTE = BASE_PALETTE  # assign the local dict to the module global
    STRENGTH = STRENGTH

    # ---------------------------
    # Main CLI
    # ---------------------------

    # --- NEW: central config with previous CLI defaults ---
    DEFAULT_CONFIG = {
        # --- Parallelization ---
        "parallel": True,     # turn off to force single-core
        "workers": None,      # None = os.cpu_count(); or set an int

        "input": "pics/4.jpg",  # was a required CLI arg; override as needed
        "pdf": "paint_by_numbers_guide.pdf",
        "colors": 28,
        "resize": None,  # e.g. (W, H)
        "cluster_space": "lab",  # {"lab","rgb"}
        "palette": list(BASE_PALETTE.keys()),
        "components": 5,
        "max_parts": 10,
        "mix_model": "learned",  # {"linear","lab","subtractive","km","learned"}
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
        # Policy knobs (balanced defaults)
        "build_max_deltaE": 8.0,
        "build_max_added_parts": 6,
        "build_max_added_pigments": 2,
        "build_max_new_pigments": 1,
        "build_min_added_fraction": 0.05,
        "build_max_chain_depth": 4,
        "build_parent_choice": "min_added_parts",  # "min_deltaE" | "min_new_pigments" | "min_added_parts"
        # --- Mixing behavior ---
        "use_tinting_strength": False,  # True = strength-aware KM; False = generic KM
        # --- Build-on graph (extra page) ---
        "build_graph_page": True,  # add a single neutral dependency-graph page
        "build_workflow_pages": True,  # if True, also generate workflow-based per-color pages
        # --- Optional pre-upscale with Real-ESRGAN ---
        "enable_upscale": True,  # turn on/off the pre-upscale
        "upscale_ok_min_long": 2500,  # if longest side >= this → no upscale
        "realesrgan_bin": "realesrgan-ncnn-vulkan-20220424-windows/realesrgan-ncnn-vulkan.exe",
        # (recommended) also add:
        "realesrgan_model_dir": "realesrgan-ncnn-vulkan-20220424-windows/models",
        "realesrgan_model_name": "realesrgan-x4plus", # matches your .bin/.param files
        "realesrgan_scale_choices": (2, 3, 4),  # allowed scale factors
        # --- Pre-brighten (applied AFTER upscaling, BEFORE analysis)
        "pre_brighten_pct": 10, # 0 = no change; 1..100 = percentage increase in brightness
        # --- Clustering (auto PAM+CLARA) ---
        "cluster_algo": "pam",  # {"kmeans","pam"}
        "pam_metric": "deltaE00",  # {"deltaE00","deltaE76","L2"}
        "pam_random_state": 42,  # reproducibility
    }

    main()
