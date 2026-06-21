from __future__ import annotations

import itertools
from functools import lru_cache
from typing import List, Sequence, Tuple

import mixbox as _mixbox
import numpy as np

from . import color as color_space

BASE_PALETTE = None
DARKEN_PER_PIGMENT = None

_MIXBOX_LATENTS: dict[tuple[int, int, int], list[float]] = {}


def darken_srgb(rgb, factor=0.8, gamma=2.2):
    """Darken an sRGB color by scaling its linear-light values."""
    r, g, b = [c / 255.0 for c in rgb]
    r_lin, g_lin, b_lin = (r ** gamma, g ** gamma, b ** gamma)
    r_lin *= factor
    g_lin *= factor
    b_lin *= factor

    def to_srgb(c_lin):
        c_lin = max(0.0, min(1.0, c_lin))
        return c_lin ** (1.0 / gamma)

    return tuple(int(round(to_srgb(c) * 255)) for c in (r_lin, g_lin, b_lin))


def _latent_for_rgb_u8(rgb_u8) -> list[float]:
    key = (int(rgb_u8[0]), int(rgb_u8[1]), int(rgb_u8[2]))
    z = _MIXBOX_LATENTS.get(key)
    if z is None:
        z = _mixbox.rgb_to_latent(key)
        _MIXBOX_LATENTS[key] = z
    return z


def mix_learned(
    parts: np.ndarray,
    base_rgbs: np.ndarray,
    base_names: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Mix colors in Mixbox latent space, then apply optional per-pigment darkening.
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

    r, g, b = _mixbox.latent_to_rgb(z_mix)
    rgb = np.array([int(r), int(g), int(b)], dtype=float)

    if base_names is not None and DARKEN_PER_PIGMENT is not None:
        factors = np.array([DARKEN_PER_PIGMENT.get(name, 1.0) for name in base_names], dtype=float)
        eff_factor = float(np.sum(w * factors))
        if abs(eff_factor - 1.0) > 1e-6:
            rgb = np.array(darken_srgb(rgb, factor=eff_factor), dtype=float)

    return rgb


def _combo_key(combo_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(combo_names)


@lru_cache(maxsize=100_000)
def _cached_mix_color(combo_key: Tuple[str, ...], parts_tuple: Tuple[int, ...]) -> Tuple[float, float, float]:
    """Memoized learned-model mixing for an exact pigment combo and parts tuple."""
    base_rgbs = np.array([BASE_PALETTE[n] for n in combo_key], dtype=float)
    parts_arr = np.array(parts_tuple, dtype=float)
    rgb = mix_learned(parts_arr, base_rgbs, combo_key)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def enumerate_partitions_upto(total: int, k: int):
    """Yield all k-tuples of nonnegative ints summing <= total, excluding all-zero."""
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
    prefer_simple_lambda_components: float = 0.03,
    prefer_simple_lambda_parts: float = 0.01,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Search integer-parts recipes using only the Mixbox learned mixing model.
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
                mix_rgb = np.array(
                    _cached_mix_color(
                        _combo_key(combo_names),
                        tuple(int(x) for x in parts),
                    ),
                    dtype=float,
                )
                err = color_space.deltaE_lab(mix_rgb, target)
                score = (
                    err
                    + prefer_simple_lambda_components * (m - 1)
                    + prefer_simple_lambda_parts * (s / float(max_parts))
                )
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
    """Human-readable recipe string, e.g. '2 parts Yellow + 1 part Black'."""
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries]) if entries else "-"


def _init_worker(palette_dict, darken_dict, delta_e_method: str = "colour_ciede2000"):
    """Install palette/darkening config in worker globals."""
    global BASE_PALETTE, DARKEN_PER_PIGMENT
    BASE_PALETTE = palette_dict
    DARKEN_PER_PIGMENT = darken_dict
    color_space.DELTA_E_METHOD = str(delta_e_method)


def _recipe_worker(
    color_rgb_list,
    base_names,
    max_parts,
    max_components,
    lambda_components,
    lambda_parts,
):
    """Runs integer_mix_best for a single centroid color."""
    color = np.array(color_rgb_list, dtype=float)
    entries, approx_rgb, err = integer_mix_best(
        color,
        base_names,
        max_parts=max_parts,
        max_components=max_components,
        prefer_simple_lambda_components=lambda_components,
        prefer_simple_lambda_parts=lambda_parts,
    )
    return {
        "entries": [(str(n), int(p)) for (n, p) in entries],
        "approx_rgb": [float(x) for x in approx_rgb],
        "err": float(err),
    }
