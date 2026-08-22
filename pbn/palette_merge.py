from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import color as color_space
from .mixing import IntegerRecipeCatalog, exhaustive_integer_mix_best


@dataclass(frozen=True)
class PaletteMergeResult:
    """Final palette after validated perceptual recipe merging."""

    labels: np.ndarray
    targets: np.ndarray
    entries: tuple[tuple[tuple[str, int], ...], ...]
    rgbs: np.ndarray
    delta_es: np.ndarray
    merged_from: tuple[tuple[int, ...], ...]


@dataclass
class _MergeGroup:
    members: tuple[int, ...]
    target_lab: np.ndarray
    entries: tuple[tuple[str, int], ...]
    rgb: np.ndarray
    delta_e: float
    reference_rgbs: np.ndarray


def _compatible(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_delta_e: float,
    max_delta_l: float,
) -> bool:
    """Return whether every cross-group color pair is perceptually close."""
    for color_a in first:
        for color_b in second:
            if abs(color_space.Lstar_from_rgb(color_a) - color_space.Lstar_from_rgb(color_b)) > max_delta_l:
                return False
            if color_space.deltaE_lab(color_a, color_b) > max_delta_e:
                return False
    return True


def merge_similar_recipes(
    labels: np.ndarray,
    targets: np.ndarray,
    entries: Sequence[Sequence[tuple[str, int]]],
    predicted_rgbs: np.ndarray,
    delta_es: Sequence[float],
    catalog: IntegerRecipeCatalog,
    base_names: Sequence[str],
    *,
    max_delta_e: float = 1.5,
    max_delta_l: float = 1.0,
    recipe_max_delta_l: float = 2.0,
    min_colors: int = 1,
) -> PaletteMergeResult:
    """Merge paint recipes whose predicted colors are perceptually redundant.

    This is complete-link agglomerative merging with an additional recipe
    validation step. A proposed group is accepted only when its recalculated
    integer recipe remains within both thresholds of every original predicted
    paint color in that group. Consequently, ``colors`` can act as an upper
    bound without allowing similarity chains to collapse distinct colors.

    Args:
        labels: Compact zero-based label map.
        targets: Representative source RGB for every label.
        entries: Existing integer recipe entries for every label.
        predicted_rgbs: Existing predicted mixed RGB for every label.
        delta_es: Existing recipe-to-target CIEDE2000 errors.
        catalog: Exhaustive valid integer recipe catalog.
        base_names: Active base-paint names.
        max_delta_e: Largest pairwise CIEDE2000 distance allowed in a group.
        max_delta_l: Largest pairwise absolute L* difference allowed in a group.
        recipe_max_delta_l: Value guardrail used when recalculating a merged recipe.
        min_colors: Lower bound on the automatically reduced palette size.

    Returns:
        Compact labels and recalculated recipes for the reduced palette.
    """
    label_map = np.asarray(labels)
    colors = len(predicted_rgbs)
    if colors <= 1 or max_delta_e <= 0 or max_delta_l < 0:
        return PaletteMergeResult(
            labels=label_map.copy(),
            targets=np.asarray(targets, dtype=float).copy(),
            entries=tuple(tuple(recipe) for recipe in entries),
            rgbs=np.asarray(predicted_rgbs, dtype=float).copy(),
            delta_es=np.asarray(delta_es, dtype=float).copy(),
            merged_from=tuple((index,) for index in range(colors)),
        )

    counts = np.bincount(label_map.reshape(-1), minlength=colors).astype(float)
    target_labs = color_space.rgb8_rows_to_lab(np.asarray(targets, dtype=float))
    groups = [
        _MergeGroup(
            members=(index,),
            target_lab=target_labs[index],
            entries=tuple(entries[index]),
            rgb=np.asarray(predicted_rgbs[index], dtype=float),
            delta_e=float(delta_es[index]),
            reference_rgbs=np.asarray(predicted_rgbs[index], dtype=float)[None, :],
        )
        for index in range(colors)
    ]
    min_colors = max(1, min(int(min_colors), colors))

    while len(groups) > min_colors:
        candidates = sorted(
            (
                color_space.deltaE_lab(groups[left].rgb, groups[right].rgb),
                left,
                right,
            )
            for left in range(len(groups))
            for right in range(left + 1, len(groups))
        )
        accepted = False
        for distance, left, right in candidates:
            if distance > max_delta_e:
                break
            group_a, group_b = groups[left], groups[right]
            if not _compatible(
                group_a.reference_rgbs,
                group_b.reference_rgbs,
                max_delta_e=max_delta_e,
                max_delta_l=max_delta_l,
            ):
                continue

            members = tuple(sorted(group_a.members + group_b.members))
            member_counts = counts[list(members)]
            merged_lab = np.average(target_labs[list(members)], axis=0, weights=member_counts)
            merged_target = np.clip(color_space.lab_to_rgb8(merged_lab), 0, 255)
            new_entries, new_rgb, new_error = exhaustive_integer_mix_best(
                merged_target,
                base_names,
                max_delta_l=recipe_max_delta_l,
                catalog=catalog,
            )
            references = np.vstack((group_a.reference_rgbs, group_b.reference_rgbs))
            if not _compatible(
                np.asarray(new_rgb, dtype=float)[None, :],
                references,
                max_delta_e=max_delta_e,
                max_delta_l=max_delta_l,
            ):
                continue

            groups[left] = _MergeGroup(
                members=members,
                target_lab=merged_lab,
                entries=tuple(new_entries),
                rgb=np.asarray(new_rgb, dtype=float),
                delta_e=float(new_error),
                reference_rgbs=references,
            )
            del groups[right]
            accepted = True
            break
        if not accepted:
            break

    groups.sort(key=lambda group: min(group.members))
    remap = np.empty(colors, dtype=np.uint8)
    for new_index, group in enumerate(groups):
        remap[list(group.members)] = new_index

    return PaletteMergeResult(
        labels=remap[label_map],
        targets=np.vstack([np.clip(color_space.lab_to_rgb8(group.target_lab), 0, 255) for group in groups]),
        entries=tuple(group.entries for group in groups),
        rgbs=np.vstack([group.rgb for group in groups]),
        delta_es=np.asarray([group.delta_e for group in groups], dtype=float),
        merged_from=tuple(group.members for group in groups),
    )
