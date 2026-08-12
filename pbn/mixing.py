from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import mixbox as _mixbox
import numpy as np
from scipy.optimize import minimize

from . import color as color_space

BASE_PALETTE = None
DARKEN_PER_PIGMENT = None

_MIXBOX_LATENTS: dict[tuple[int, int, int], list[float]] = {}


@dataclass(frozen=True)
class IntegerRecipeCatalog:
    """Precomputed colours and entries for every allowed integer recipe."""

    entries: tuple[tuple[tuple[str, int], ...], ...]
    rgbs: np.ndarray


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


def build_integer_recipe_catalog(
    base_names: Sequence[str],
    *,
    max_parts: int = 10,
    max_components: int = 5,
    max_part_per_pigment: int = 5,
) -> IntegerRecipeCatalog:
    """Build every unique, reduced recipe within the practical limits.

    Args:
        base_names: Pigments available to the recipe search.
        max_parts: Maximum sum of all integer parts in one recipe.
        max_components: Maximum number of distinct pigments in one recipe.
        max_part_per_pigment: Maximum integer amount of any one pigment.

    Returns:
        A reusable catalogue sorted so equally accurate recipes prefer fewer
        pigments and fewer total parts.
    """
    names = list(base_names)
    max_parts = max(1, int(max_parts))
    max_part_per_pigment = max(1, min(int(max_part_per_pigment), max_parts))
    max_components = max(1, min(int(max_components), len(names), max_parts))
    candidates: list[tuple[tuple[tuple[str, int], ...], np.ndarray]] = []

    for component_count in range(1, max_components + 1):
        for combo in itertools.combinations(names, component_count):
            for parts in itertools.product(range(1, max_part_per_pigment + 1), repeat=component_count):
                if sum(parts) > max_parts or math.gcd(*parts) != 1:
                    continue
                rgb = np.array(_cached_mix_color(tuple(combo), tuple(parts)), dtype=float)
                entries = tuple((combo[i], int(parts[i])) for i in range(component_count))
                candidates.append((entries, rgb))

    candidates.sort(key=lambda item: (len(item[0]), sum(p for _name, p in item[0]),
                                      max(p for _name, p in item[0]), item[0]))
    return IntegerRecipeCatalog(
        entries=tuple(entries for entries, _rgb in candidates),
        rgbs=np.stack([rgb for _entries, rgb in candidates], axis=0),
    )


def exhaustive_integer_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 10,
    max_components: int = 5,
    max_part_per_pigment: int = 5,
    catalog: IntegerRecipeCatalog | None = None,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """Return the globally best recipe from the bounded integer catalogue."""
    if catalog is None:
        catalog = build_integer_recipe_catalog(
            base_names,
            max_parts=max_parts,
            max_components=max_components,
            max_part_per_pigment=max_part_per_pigment,
        )
    errors = color_space.deltaE_lab_rows(catalog.rgbs, target_rgb)
    best_index = int(np.argmin(errors))
    best_rgb = catalog.rgbs[best_index].copy()
    best_error = color_space.deltaE_lab(best_rgb, target_rgb)
    return list(catalog.entries[best_index]), best_rgb, best_error


def _normalize_weights(raw_weights: Sequence[float]) -> np.ndarray:
    weights = np.clip(np.asarray(raw_weights, dtype=float), 0.0, 1.0)
    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / max(1, len(weights))
    return weights / total


def _continuous_combo_best(
    target: np.ndarray,
    combo_names: Sequence[str],
    *,
    seed: int = 0,
    maxiter: int = 120,
    starts: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    base_rgbs = np.array([BASE_PALETTE[n] for n in combo_names], dtype=float)

    if len(combo_names) == 1:
        weights = np.array([1.0], dtype=float)
        rgb = mix_learned(weights, base_rgbs, combo_names)
        return weights, rgb, color_space.deltaE_lab(rgb, target)

    def objective(weights):
        rgb = mix_learned(weights, base_rgbs, combo_names)
        return color_space.deltaE_lab(rgb, target)

    rng = np.random.default_rng(seed)
    start_points = [np.ones(len(combo_names), dtype=float) / len(combo_names)]
    for i in range(len(combo_names)):
        one_hot = np.zeros(len(combo_names), dtype=float)
        one_hot[i] = 1.0
        start_points.append(0.75 * one_hot + 0.25 / len(combo_names))
    for _ in range(max(0, starts - len(start_points))):
        start_points.append(rng.dirichlet(np.ones(len(combo_names))))

    best_weights = start_points[0]
    best_err = float("inf")
    for start in start_points[:max(1, starts)]:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(combo_names),
            constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
            options={"maxiter": maxiter, "ftol": 1e-4, "disp": False},
        )
        weights = _normalize_weights(result.x if result.success else start)
        err = objective(weights)
        if err < best_err:
            best_err = err
            best_weights = weights

    weights = best_weights
    rgb = mix_learned(weights, base_rgbs, combo_names)
    return weights, rgb, color_space.deltaE_lab(rgb, target)


def _quantized_parts_candidates(weights: np.ndarray, max_parts: int):
    active = np.where(weights > 1e-4)[0]
    if len(active) == 0:
        active = np.array([int(np.argmax(weights))])

    for total_parts in range(max(1, len(active)), max_parts + 1):
        raw = weights * total_parts
        parts = np.floor(raw).astype(int)
        parts[active] = np.maximum(parts[active], 1)

        while int(parts.sum()) > total_parts:
            removable = np.where(parts > 1)[0]
            if len(removable) == 0:
                break
            j = min(removable, key=lambda idx: raw[idx] / max(1, parts[idx] - 1))
            parts[j] -= 1

        while int(parts.sum()) < total_parts:
            j = int(np.argmax(raw - parts))
            parts[j] += 1

        if int(parts.sum()) <= max_parts and int(parts.sum()) > 0:
            yield tuple(int(x) for x in parts)


def _parts_to_entries(combo_names: Sequence[str], parts: Sequence[int]) -> List[Tuple[str, int]]:
    return [(combo_names[i], int(p)) for i, p in enumerate(parts) if p > 0]


def _random_composition(rng: np.random.Generator, total: int, k: int) -> np.ndarray:
    if k <= 1:
        return np.array([total], dtype=int)
    cuts = sorted(rng.choice(total + k - 1, k - 1, replace=False))
    vals = []
    prev = -1
    for cut in cuts + [total + k - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return np.array(vals, dtype=int)


def _local_improve_parts(
    target: np.ndarray,
    combo_names: Sequence[str],
    parts: np.ndarray,
    *,
    max_components: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    base_rgbs = np.array([BASE_PALETTE[n] for n in combo_names], dtype=float)

    def evaluate(candidate_parts):
        rgb = mix_learned(np.array(candidate_parts, dtype=float), base_rgbs, combo_names)
        return color_space.deltaE_lab(rgb, target), rgb

    best_parts = np.array(parts, dtype=int)
    best_err, best_rgb = evaluate(best_parts)
    improved = True
    while improved:
        improved = False
        active_count = int(np.count_nonzero(best_parts))
        for src in range(len(best_parts)):
            if best_parts[src] <= 0:
                continue
            for dst in range(len(best_parts)):
                if src == dst:
                    continue
                if best_parts[dst] == 0 and active_count >= max_components:
                    continue
                candidate = best_parts.copy()
                candidate[src] -= 1
                candidate[dst] += 1
                err, rgb = evaluate(candidate)
                if err + 1e-9 < best_err:
                    best_parts, best_err, best_rgb = candidate, err, rgb
                    improved = True
                    active_count = int(np.count_nonzero(best_parts))
    return best_parts, best_rgb, best_err


def _normalize_recipe_parts(parts: Sequence[int], max_parts: int, max_components: int) -> np.ndarray:
    arr = np.maximum(np.array(parts, dtype=int), 0)
    max_parts = max(1, int(max_parts))
    max_components = max(1, int(max_components))

    while int(np.count_nonzero(arr)) > max_components:
        nonzero = np.flatnonzero(arr > 0)
        arr[int(min(nonzero, key=lambda i: arr[i]))] = 0

    while int(arr.sum()) > max_parts:
        nonzero = np.flatnonzero(arr > 0)
        arr[int(max(nonzero, key=lambda i: arr[i]))] -= 1

    if int(arr.sum()) <= 0:
        arr[0] = 1
    return arr


def _random_recipe_parts(
    rng: np.random.Generator,
    n: int,
    max_parts: int,
    max_components: int,
    ranked_indices: Sequence[int] | None = None,
) -> np.ndarray:
    k = int(rng.integers(1, min(n, max_components) + 1))
    if ranked_indices is not None and rng.random() < 0.75:
        pool_size = min(n, max(k + 2, max_components))
        pool = np.array(ranked_indices[:pool_size], dtype=int)
        chosen = rng.choice(pool, size=k, replace=False)
    else:
        chosen = rng.choice(n, size=k, replace=False)

    total = int(rng.integers(k, max_parts + 1))
    local = np.ones(k, dtype=int)
    for _ in range(total - k):
        local[int(rng.integers(0, k))] += 1

    parts = np.zeros(n, dtype=int)
    parts[chosen] = local
    return parts


def _hillclimb_refine_parts(
    target: np.ndarray,
    names: Sequence[str],
    start_parts: Sequence[int],
    *,
    max_parts: int,
    max_components: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    base_rgbs = np.array([BASE_PALETTE[n] for n in names], dtype=float)
    cache: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}

    def evaluate(candidate_parts: Sequence[int]) -> tuple[float, np.ndarray]:
        key = tuple(int(x) for x in candidate_parts)
        cached = cache.get(key)
        if cached is None:
            if sum(key) <= 0:
                cached = (float("inf"), target.copy())
            else:
                rgb = mix_learned(np.array(key, dtype=float), base_rgbs, names)
                cached = (color_space.deltaE_lab(rgb, target), rgb)
            cache[key] = cached
        return cached

    best_parts = _normalize_recipe_parts(start_parts, max_parts, max_components)
    best_err, best_rgb = evaluate(best_parts)

    improved = True
    while improved:
        improved = False
        active = np.flatnonzero(best_parts > 0)
        candidates = []

        for src in active:
            for dst in range(len(best_parts)):
                if int(src) == dst:
                    continue
                if best_parts[dst] == 0 and int(np.count_nonzero(best_parts)) >= max_components:
                    continue
                candidate = best_parts.copy()
                candidate[int(src)] -= 1
                candidate[dst] += 1
                candidates.append(candidate)

        if int(best_parts.sum()) < max_parts:
            for dst in range(len(best_parts)):
                if best_parts[dst] == 0 and int(np.count_nonzero(best_parts)) >= max_components:
                    continue
                candidate = best_parts.copy()
                candidate[dst] += 1
                candidates.append(candidate)

        if int(best_parts.sum()) > 1:
            for src in active:
                candidate = best_parts.copy()
                candidate[int(src)] -= 1
                candidates.append(candidate)

        for candidate in candidates:
            candidate = _normalize_recipe_parts(candidate, max_parts, max_components)
            err, rgb = evaluate(candidate)
            if err + 1e-12 < best_err:
                best_parts, best_err, best_rgb = candidate, err, rgb
                improved = True

    return best_parts, best_rgb, best_err


def multistart_hillclimb_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 20,
    max_components: int = 10,
    starts: int = 45,
    seed_entries: Sequence[Tuple[str, int]] | None = None,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Search integer-parts recipes with seeded multistart coordinate descent.

    Starts from the existing recipe and several random whole-part recipes, then
    repeatedly applies one-part transfer/add/remove moves until locally optimal.
    """
    target = np.array(target_rgb, dtype=float)
    names = list(base_names)
    rng = np.random.default_rng(seed)
    max_parts = max(1, int(max_parts))
    max_components = max(1, min(int(max_components), len(names), max_parts))

    start_vectors: list[np.ndarray] = []
    if seed_entries:
        seed_parts = np.zeros(len(names), dtype=int)
        name_to_index = {name: i for i, name in enumerate(names)}
        for name, amount in seed_entries:
            if name in name_to_index:
                seed_parts[name_to_index[name]] += int(amount)
        if int(seed_parts.sum()) > 0:
            start_vectors.append(seed_parts)

    single_scores = []
    for i, name in enumerate(names):
        err = color_space.deltaE_lab(np.array(BASE_PALETTE[name], dtype=float), target)
        single_scores.append((err, i))
        one = np.zeros(len(names), dtype=int)
        one[i] = min(max_parts, max(1, max_parts // 2))
        start_vectors.append(one)
    ranked_indices = [i for _err, i in sorted(single_scores)]

    while len(start_vectors) < max(1, int(starts)):
        start_vectors.append(
            _random_recipe_parts(
                rng,
                len(names),
                max_parts,
                max_components,
                ranked_indices=ranked_indices,
            )
        )

    best_parts = None
    best_rgb = target.copy()
    best_err = float("inf")
    seen: set[tuple[int, ...]] = set()
    for start_parts in start_vectors[:max(1, int(starts))]:
        key = tuple(int(x) for x in _normalize_recipe_parts(start_parts, max_parts, max_components))
        if key in seen:
            continue
        seen.add(key)
        parts, rgb, err = _hillclimb_refine_parts(
            target,
            names,
            key,
            max_parts=max_parts,
            max_components=max_components,
        )
        if err < best_err:
            best_parts, best_rgb, best_err = parts, rgb, err

    if best_parts is None:
        best_parts = np.zeros(len(names), dtype=int)
        best_parts[ranked_indices[0]] = 1
        base_rgbs = np.array([BASE_PALETTE[n] for n in names], dtype=float)
        best_rgb = mix_learned(best_parts, base_rgbs, names)
        best_err = color_space.deltaE_lab(best_rgb, target)

    return _parts_to_entries(names, best_parts), best_rgb, best_err


def stochastic_integer_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 20,
    max_components: int = 10,
    samples: int = 12000,
    seed_entries: Sequence[Tuple[str, int]] | None = None,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Randomized integer-parts search with one-part local refinement.

    This is useful after continuous optimization because it optimizes the actual
    integer recipe directly instead of trusting a simple fractional-to-integer rounding.
    """
    target = np.array(target_rgb, dtype=float)
    rng = np.random.default_rng(seed)
    names = list(base_names)
    base_rgbs = np.array([BASE_PALETTE[n] for n in names], dtype=float)
    max_parts = max(1, int(max_parts))
    max_components = max(1, min(int(max_components), len(names), max_parts))

    best_err = float("inf")
    best_parts = None
    best_rgb = target.copy()
    top_candidates: list[tuple[float, np.ndarray]] = []
    max_refine_candidates = 24

    def evaluate_raw(parts):
        rgb = mix_learned(np.array(parts, dtype=float), base_rgbs, names)
        return color_space.deltaE_lab(rgb, target), rgb

    def consider(parts):
        if int(parts.sum()) <= 0 or int(np.count_nonzero(parts)) > max_components:
            return
        err, _rgb = evaluate_raw(parts)
        candidate = np.array(parts, dtype=int)
        if len(top_candidates) < max_refine_candidates:
            top_candidates.append((err, candidate))
            top_candidates.sort(key=lambda item: item[0])
        elif err < top_candidates[-1][0]:
            top_candidates[-1] = (err, candidate)
            top_candidates.sort(key=lambda item: item[0])

    if seed_entries:
        parts = np.zeros(len(names), dtype=int)
        name_to_index = {name: i for i, name in enumerate(names)}
        for name, amount in seed_entries:
            if name in name_to_index:
                parts[name_to_index[name]] += int(amount)
        if 0 < int(parts.sum()) <= max_parts:
            consider(parts)

    single_scores = []
    for i, name in enumerate(names):
        err = color_space.deltaE_lab(np.array(BASE_PALETTE[name], dtype=float), target)
        single_scores.append((err, i))
    ranked_indices = [i for _err, i in sorted(single_scores)]

    for sample_index in range(max(1, int(samples))):
        if rng.random() < 0.7:
            chosen_count = int(rng.integers(1, max_components + 1))
            pool_size = min(len(names), max(max_components, chosen_count + 2))
            pool = ranked_indices[:pool_size]
            chosen = rng.choice(pool, size=chosen_count, replace=False)
        else:
            chosen_count = int(rng.integers(1, max_components + 1))
            chosen = rng.choice(len(names), size=chosen_count, replace=False)

        total_parts = int(rng.integers(max(1, chosen_count), max_parts + 1))
        local_parts = _random_composition(rng, total_parts, chosen_count)
        if np.count_nonzero(local_parts) == 0:
            continue

        parts = np.zeros(len(names), dtype=int)
        for idx, amount in zip(chosen, local_parts):
            parts[int(idx)] = int(amount)
        if np.count_nonzero(parts) == 0:
            continue
        consider(parts)

    seen = set()
    for _raw_err, candidate_parts in top_candidates:
        key = tuple(int(x) for x in candidate_parts)
        if key in seen:
            continue
        seen.add(key)
        improved_parts, rgb, err = _local_improve_parts(
            target,
            names,
            candidate_parts,
            max_components=max_components,
        )
        if err < best_err:
            best_err = err
            best_parts = improved_parts
            best_rgb = rgb

    if best_parts is None:
        best_parts = np.zeros(len(names), dtype=int)
        best_parts[int(np.argmin([score for score, _idx in single_scores]))] = 1
        best_rgb = mix_learned(best_parts, base_rgbs, names)
        best_err = color_space.deltaE_lab(best_rgb, target)

    return _parts_to_entries(names, best_parts), best_rgb, best_err


def genetic_integer_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 20,
    max_components: int = 10,
    population: int = 180,
    generations: int = 160,
    seed_entries: Sequence[Tuple[str, int]] | None = None,
    seed: int = 0,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Genetic search over integer paint recipes.

    Recipes are integer vectors. The GA uses elitism, weighted parent selection,
    crossover, mutation, and a final one-part local refinement.
    """
    target = np.array(target_rgb, dtype=float)
    rng = np.random.default_rng(seed)
    names = list(base_names)
    base_rgbs = np.array([BASE_PALETTE[n] for n in names], dtype=float)
    max_parts = max(1, int(max_parts))
    max_components = max(1, min(int(max_components), len(names), max_parts))
    population = max(8, int(population))
    generations = max(1, int(generations))

    def evaluate(parts):
        rgb = mix_learned(np.array(parts, dtype=float), base_rgbs, names)
        return color_space.deltaE_lab(rgb, target), rgb

    def normalize(parts):
        parts = np.maximum(np.array(parts, dtype=int), 0)
        while int(parts.sum()) > max_parts:
            nonzero = np.flatnonzero(parts > 0)
            parts[int(rng.choice(nonzero))] -= 1
        while int(np.count_nonzero(parts)) > max_components:
            nonzero = np.flatnonzero(parts > 0)
            smallest = min(nonzero, key=lambda i: parts[i])
            parts[int(smallest)] = 0
        if int(parts.sum()) <= 0:
            parts[int(rng.integers(0, len(parts)))] = 1
        return parts

    def random_parts():
        chosen_count = int(rng.integers(1, max_components + 1))
        chosen = rng.choice(len(names), size=chosen_count, replace=False)
        total_parts = int(rng.integers(chosen_count, max_parts + 1))
        local = np.ones(chosen_count, dtype=int)
        for _ in range(total_parts - chosen_count):
            local[int(rng.integers(0, chosen_count))] += 1
        parts = np.zeros(len(names), dtype=int)
        parts[chosen] = local
        return parts

    def mutate(parts):
        candidate = np.array(parts, dtype=int).copy()
        if rng.random() < 0.55 and int(candidate.sum()) > 0:
            srcs = np.flatnonzero(candidate > 0)
            src = int(rng.choice(srcs))
            dst = int(rng.integers(0, len(candidate)))
            if src != dst:
                candidate[src] -= 1
                candidate[dst] += 1
        elif rng.random() < 0.75 and int(candidate.sum()) < max_parts:
            candidate[int(rng.integers(0, len(candidate)))] += 1
        elif int(candidate.sum()) > 1:
            srcs = np.flatnonzero(candidate > 0)
            candidate[int(rng.choice(srcs))] -= 1
        else:
            candidate[int(rng.integers(0, len(candidate)))] += 1
        return normalize(candidate)

    pop = []
    if seed_entries:
        seed_parts = np.zeros(len(names), dtype=int)
        name_to_index = {name: i for i, name in enumerate(names)}
        for name, amount in seed_entries:
            if name in name_to_index:
                seed_parts[name_to_index[name]] += int(amount)
        if int(seed_parts.sum()) > 0:
            pop.append(normalize(seed_parts))
    while len(pop) < population:
        pop.append(random_parts())

    best_parts = pop[0].copy()
    best_err, best_rgb = evaluate(best_parts)

    for _ in range(generations):
        scored = []
        for parts in pop:
            err, rgb = evaluate(parts)
            scored.append((err, parts, rgb))
            if err < best_err:
                best_parts, best_err, best_rgb = parts.copy(), err, rgb
        scored.sort(key=lambda item: item[0])
        elite_count = max(2, population // 8)
        next_pop = [parts.copy() for _err, parts, _rgb in scored[:elite_count]]
        weights = np.array([1.0 / (0.05 + err) for err, _parts, _rgb in scored], dtype=float)
        weights = weights / weights.sum()

        while len(next_pop) < population:
            p1 = scored[int(rng.choice(len(scored), p=weights))][1]
            p2 = scored[int(rng.choice(len(scored), p=weights))][1]
            mask = rng.random(len(names)) < 0.5
            child = np.where(mask, p1, p2)
            if rng.random() < 0.85:
                child = mutate(child)
            next_pop.append(normalize(child))
        pop = next_pop

    best_parts, best_rgb, best_err = _local_improve_parts(
        target,
        names,
        best_parts,
        max_components=max_components,
    )
    return _parts_to_entries(names, best_parts), best_rgb, best_err


def continuous_mix_best(
    target_rgb: Sequence[float],
    base_names: Sequence[str],
    *,
    max_parts: int = 20,
    max_components: int = 5,
    candidate_pigments: int = 8,
    max_combos: int = 32,
    seed_entries: Sequence[Tuple[str, int]] | None = None,
    seed: int = 0,
    maxiter: int = 120,
    starts: int = 5,
) -> Tuple[List[Tuple[str, int]], np.ndarray, float]:
    """
    Search continuous pigment weights, then quantize the best result back to integer parts.

    This is intended as a high-error fallback for integer_mix_best. It searches a reduced
    set of plausible pigment combinations to keep runtime bounded.
    """
    target = np.array(target_rgb, dtype=float)
    max_components = max(1, min(max_components, len(base_names)))
    max_parts = max(1, int(max_parts))

    single_scores = []
    for name in base_names:
        rgb = np.array(BASE_PALETTE[name], dtype=float)
        single_scores.append((color_space.deltaE_lab(rgb, target), name))
    single_score_by_name = {name: err for err, name in single_scores}
    ranked_names = [name for _err, name in sorted(single_scores)]
    search_names = ranked_names[:max(1, min(candidate_pigments, len(ranked_names)))]

    seed_combo = tuple()
    seed_parts_by_combo = {}
    if seed_entries:
        seed_combo = tuple(n for n, p in seed_entries if p > 0)
        seed_parts_by_combo[seed_combo] = tuple(int(p) for n, p in seed_entries if p > 0)
        for name in seed_combo:
            if name in base_names and name not in search_names:
                search_names.append(name)

    combos = []
    for m in range(1, max_components + 1):
        for combo in itertools.combinations(search_names, m):
            combos.append(combo)
    if seed_combo and seed_combo not in combos and len(seed_combo) <= max_components:
        combos.insert(0, seed_combo)
    full_candidate_combo = tuple(search_names[:max_components])
    if len(full_candidate_combo) > 1 and full_candidate_combo not in combos:
        combos.insert(0, full_candidate_combo)

    def combo_rank(combo):
        return (
            0 if combo == seed_combo else 1,
            0 if combo == full_candidate_combo else 1,
            sum(single_score_by_name[name] for name in combo) / len(combo),
            len(combo),
        )

    combos = sorted(set(combos), key=combo_rank)[:max(1, int(max_combos))]

    best_err = float("inf")
    best_entries: List[Tuple[str, int]] = []
    best_rgb = target.copy()

    for combo_index, combo in enumerate(combos):
        weights, _continuous_rgb, _continuous_err = _continuous_combo_best(
            target,
            combo,
            seed=seed + combo_index,
            maxiter=maxiter,
            starts=starts,
        )
        base_rgbs = np.array([BASE_PALETTE[n] for n in combo], dtype=float)
        seen_parts = set()
        if combo in seed_parts_by_combo:
            seed_parts = seed_parts_by_combo[combo]
            if sum(seed_parts) <= max_parts:
                seen_parts.add(seed_parts)
                rgb = mix_learned(np.array(seed_parts, dtype=float), base_rgbs, combo)
                err = color_space.deltaE_lab(rgb, target)
                if err < best_err:
                    best_err = err
                    best_rgb = rgb
                    best_entries = [(combo[i], int(p)) for i, p in enumerate(seed_parts) if p > 0]
        for parts in _quantized_parts_candidates(weights, max_parts):
            if parts in seen_parts:
                continue
            seen_parts.add(parts)
            rgb = mix_learned(np.array(parts, dtype=float), base_rgbs, combo)
            err = color_space.deltaE_lab(rgb, target)
            if err < best_err:
                best_err = err
                best_rgb = rgb
                best_entries = [(combo[i], int(p)) for i, p in enumerate(parts) if p > 0]

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
    recipe_hillclimb_starts=45,
    recipe_hillclimb_seed=0,
):
    """Runs multistart hill-climb recipe search for a single centroid color."""
    color = np.array(color_rgb_list, dtype=float)
    del lambda_components, lambda_parts
    entries, approx_rgb, err = multistart_hillclimb_mix_best(
        color,
        base_names,
        max_parts=max_parts,
        max_components=max_components,
        starts=int(recipe_hillclimb_starts),
        seed=int(recipe_hillclimb_seed),
    )

    return {
        "entries": [(str(n), int(p)) for (n, p) in entries],
        "approx_rgb": [float(x) for x in approx_rgb],
        "err": float(err),
    }
