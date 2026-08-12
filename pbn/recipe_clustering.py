from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000

from . import color as color_space
from .mixing import IntegerRecipeCatalog


def _delta_e_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return CIEDE2000 distances between every point and centre."""
    return deltaE_ciede2000(points[:, np.newaxis, :], centers[np.newaxis, :, :])


def _distinct_catalog(catalog: IntegerRecipeCatalog) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct RGB colours and their original catalogue indices."""
    _unique, first = np.unique(catalog.rgbs.astype(np.uint8), axis=0, return_index=True)
    indices = np.sort(first)
    return catalog.rgbs[indices], indices


def _nearest_distinct(target_labs: np.ndarray, catalog_labs: np.ndarray) -> np.ndarray:
    """Map initial centres to distinct attainable colours."""
    selected: list[int] = []
    used: set[int] = set()
    for target in target_labs:
        distances = deltaE_ciede2000(catalog_labs, target[np.newaxis, :])
        for index in np.argsort(distances):
            candidate = int(index)
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                break
    return np.array(selected, dtype=int)


def recipe_aware_cluster(
    rgb: np.ndarray,
    colors: int,
    catalog: IntegerRecipeCatalog,
    *,
    random_state: int = 42,
    iterations: int = 8,
    shortlist: int = 384,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Cluster an image using only colours attainable by integer recipes.

    Args:
        rgb: RGB image array.
        colors: Number of paint colours to select.
        catalog: Valid integer recipes and their predicted colours.
        random_state: Reproducible K-Means initializer seed.
        iterations: Maximum constrained centre-update passes.
        shortlist: Nearby catalogue candidates evaluated per cluster.

    Returns:
        Pixel labels, selected RGB palette, and corresponding catalogue indices.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    unique_rgb, inverse, counts = np.unique(
        image.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True
    )
    pixel_labs = color_space.rgb8_rows_to_lab(unique_rgb)
    distinct_rgb, catalog_indices = _distinct_catalog(catalog)
    catalog_labs = color_space.rgb8_rows_to_lab(distinct_rgb)
    colors = max(1, min(int(colors), len(distinct_rgb), len(unique_rgb)))

    initializer = KMeans(n_clusters=colors, random_state=random_state, n_init=8)
    initializer.fit(pixel_labs, sample_weight=counts)
    selected = _nearest_distinct(initializer.cluster_centers_, catalog_labs)

    for _ in range(max(1, int(iterations))):
        distances = _delta_e_matrix(pixel_labs, catalog_labs[selected])
        labels = np.argmin(distances, axis=1)
        updated = selected.copy()
        reserved: set[int] = set()
        cluster_order = np.argsort([
            -int(counts[labels == cluster].sum()) for cluster in range(colors)
        ])

        for cluster in cluster_order:
            mask = labels == int(cluster)
            if not np.any(mask):
                continue
            cluster_labs = pixel_labs[mask]
            cluster_counts = counts[mask]
            centroid = np.average(cluster_labs, axis=0, weights=cluster_counts)
            euclidean = np.linalg.norm(catalog_labs - centroid, axis=1)
            candidate_count = min(max(1, int(shortlist)), len(euclidean))
            candidates = np.argpartition(euclidean, candidate_count - 1)[:candidate_count]
            costs = (
                deltaE_ciede2000(
                    cluster_labs[np.newaxis, :, :],
                    catalog_labs[candidates, np.newaxis, :],
                )
                @ cluster_counts
            )
            for position in np.argsort(costs):
                candidate = int(candidates[int(position)])
                if candidate not in reserved:
                    updated[int(cluster)] = candidate
                    reserved.add(candidate)
                    break

        if np.array_equal(updated, selected):
            break
        selected = updated

    final_distances = _delta_e_matrix(pixel_labs, catalog_labs[selected])
    unique_labels = np.argmin(final_distances, axis=1).astype(np.uint8)
    labels_full = unique_labels[inverse].reshape(image.shape[:2])
    selected_catalog_indices = tuple(int(catalog_indices[index]) for index in selected)
    palette = np.clip(np.rint(catalog.rgbs[list(selected_catalog_indices)]), 0, 255).astype(np.uint8)
    return labels_full, palette, selected_catalog_indices
