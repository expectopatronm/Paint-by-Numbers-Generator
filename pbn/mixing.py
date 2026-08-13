from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import mixbox as _mixbox
import numpy as np

from . import color as color_space

BASE_PALETTE = None
DARKEN_PER_PIGMENT = None

_MIXBOX_LATENTS: dict[tuple[int, int, int], list[float]] = {}


@dataclass(frozen=True)
class IntegerRecipeCatalog:
    """Precomputed colors and entries for every allowed integer recipe."""

    entries: tuple[tuple[tuple[str, int], ...], ...]
    rgbs: np.ndarray


def constrained_delta_e_recipe_index(
    catalog_rgbs: np.ndarray,
    target_rgb: Sequence[float],
    *,
    max_delta_l: float = 2.0,
) -> tuple[int, np.ndarray]:
    """Choose the lowest Delta E recipe inside an absolute L* guardrail.

    Args:
        catalog_rgbs: Predicted RGB colors for the recipe catalogue.
        target_rgb: RGB color to match.
        max_delta_l: Maximum acceptable absolute L* error.

    Returns:
        Selected catalogue index and the Delta E array for all recipes.
    """
    labs = color_space.rgb8_rows_to_lab(catalog_rgbs)
    target_lab = color_space.rgb8_to_lab(np.asarray(target_rgb, dtype=np.float32))
    value_errors = np.abs(labs[:, 0] - target_lab[0])
    errors = color_space.deltaE_lab_rows(catalog_rgbs, target_rgb)
    eligible = np.flatnonzero(value_errors <= max(0.0, float(max_delta_l)))
    if not len(eligible):
        best_value = value_errors.min()
        eligible = np.flatnonzero(value_errors <= best_value + 0.25)
    return int(eligible[np.argmin(errors[eligible])]), errors


def darken_srgb(rgb, factor=0.8, gamma=2.2):
    """Darken an sRGB color by scaling its linear-light values."""
    linear = np.power(np.asarray(rgb, dtype=float) / 255.0, gamma) * factor
    return tuple(np.rint(np.power(np.clip(linear, 0.0, 1.0), 1.0 / gamma) * 255).astype(int))


def _latent_for_rgb_u8(rgb_u8) -> list[float]:
    key = tuple(int(channel) for channel in rgb_u8)
    latent = _MIXBOX_LATENTS.get(key)
    if latent is None:
        latent = _mixbox.rgb_to_latent(key)
        _MIXBOX_LATENTS[key] = latent
    return latent


def mix_learned(
    parts: np.ndarray,
    base_rgbs: np.ndarray,
    base_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Mix paint colors in Mixbox latent space."""
    amounts = np.asarray(parts, dtype=float)
    if amounts.sum() <= 0:
        return base_rgbs[0].astype(float)

    weights = amounts / amounts.sum()
    latent = np.sum(
        [weight * np.asarray(_latent_for_rgb_u8(rgb)) for weight, rgb in zip(weights, base_rgbs)],
        axis=0,
    )
    rgb = np.asarray(_mixbox.latent_to_rgb(latent.tolist()), dtype=float)

    if base_names is not None and DARKEN_PER_PIGMENT is not None:
        factors = np.array([DARKEN_PER_PIGMENT.get(name, 1.0) for name in base_names])
        factor = float(weights @ factors)
        if not np.isclose(factor, 1.0):
            rgb = np.asarray(darken_srgb(rgb, factor=factor), dtype=float)
    return rgb


@lru_cache(maxsize=100_000)
def _cached_mix_color(names: Tuple[str, ...], parts: Tuple[int, ...]) -> Tuple[float, float, float]:
    """Return a memoized Mixbox prediction for an exact integer recipe."""
    base_rgbs = np.array([BASE_PALETTE[name] for name in names], dtype=float)
    rgb = mix_learned(np.asarray(parts, dtype=float), base_rgbs, names)
    return tuple(float(channel) for channel in rgb)


def build_integer_recipe_catalog(
    base_names: Sequence[str],
    *,
    max_parts: int = 10,
    max_components: int = 5,
    max_part_per_pigment: int = 5,
) -> IntegerRecipeCatalog:
    """Build every unique reduced recipe inside the practical limits."""
    names = list(base_names)
    max_parts = max(1, int(max_parts))
    max_part_per_pigment = max(1, min(int(max_part_per_pigment), max_parts))
    max_components = max(1, min(int(max_components), len(names), max_parts))
    candidates = []

    for component_count in range(1, max_components + 1):
        for combo in itertools.combinations(names, component_count):
            for parts in itertools.product(range(1, max_part_per_pigment + 1), repeat=component_count):
                if sum(parts) > max_parts or math.gcd(*parts) != 1:
                    continue
                entries = tuple(zip(combo, map(int, parts)))
                rgb = np.asarray(_cached_mix_color(tuple(combo), tuple(parts)), dtype=float)
                candidates.append((entries, rgb))

    candidates.sort(
        key=lambda item: (
            len(item[0]),
            sum(parts for _name, parts in item[0]),
            max(parts for _name, parts in item[0]),
            item[0],
        )
    )
    return IntegerRecipeCatalog(
        entries=tuple(entries for entries, _rgb in candidates),
        rgbs=np.stack([rgb for _entries, rgb in candidates]),
    )


def exhaustive_integer_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 10,
    max_components: int = 5,
    max_part_per_pigment: int = 5,
    max_delta_l: float = 2.0,
    catalog: IntegerRecipeCatalog | None = None,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """Return the lowest-Delta-E integer recipe inside the L* guardrail."""
    catalog = catalog or build_integer_recipe_catalog(
        base_names,
        max_parts=max_parts,
        max_components=max_components,
        max_part_per_pigment=max_part_per_pigment,
    )
    best_index, errors = constrained_delta_e_recipe_index(
        catalog.rgbs,
        target_rgb,
        max_delta_l=max_delta_l,
    )
    return (
        list(catalog.entries[best_index]),
        catalog.rgbs[best_index].copy(),
        float(errors[best_index]),
    )


def recipe_text(entries: List[Tuple[str, int]]) -> str:
    """Return a human-readable whole-parts recipe."""
    return " + ".join(
        f"{parts} {'part' if parts == 1 else 'parts'} {name}" for name, parts in entries
    )
