#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbn.image_ops import cleanup_label_regions, potts_smooth_labels


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def resize_max(rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return rgb
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.array(Image.fromarray(rgb).resize(new_size, Image.BILINEAR))


def crop_rgb(rgb: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return rgb
    x, y, w, h = crop
    H, W = rgb.shape[:2]
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(x0 + 1, min(W, x0 + w))
    y1 = max(y0 + 1, min(H, y0 + h))
    return rgb[y0:y1, x0:x1]


def rgb_to_lab_features(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def kmeans_labels(rgb: np.ndarray, colors: int, random_state: int = 42):
    h, w = rgb.shape[:2]
    feats = rgb_to_lab_features(rgb)
    km = KMeans(n_clusters=colors, random_state=random_state, n_init=8)
    labels = km.fit_predict(feats).reshape(h, w).astype(np.int32)
    centers_lab = km.cluster_centers_.astype(np.uint8).reshape(1, colors, 3)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(colors, 3)
    quantized = centers_rgb[labels].astype(np.uint8)
    return labels, quantized


def quantize_from_labels(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    for lab in np.unique(labels):
        mask = labels == lab
        if not np.any(mask):
            continue
        color = np.median(rgb[mask], axis=0)
        out[mask] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return out


def spatial_kmeans_labels(rgb: np.ndarray, colors: int, spatial_weight: float, random_state: int = 42):
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xy_scale = max(h, w) / 255.0
    feats = np.dstack([
        lab,
        (xx / xy_scale) * float(spatial_weight),
        (yy / xy_scale) * float(spatial_weight),
    ]).reshape(-1, 5)
    km = KMeans(n_clusters=colors, random_state=random_state, n_init=8)
    labels = km.fit_predict(feats).reshape(h, w).astype(np.int32)
    return labels, quantize_from_labels(rgb, labels)


def meanshift_prefilter(rgb: np.ndarray, spatial_radius: float, color_radius: float) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.pyrMeanShiftFiltering(bgr, sp=spatial_radius, sr=color_radius)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def cluster_ids_by_area(labels: np.ndarray, limit: int) -> list[int]:
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [int(ids[i]) for i in order[:limit]]


def mask_image(labels: np.ndarray, cluster_id: int) -> np.ndarray:
    mask = labels == cluster_id
    out = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    out[mask] = np.array([35, 35, 35], dtype=np.uint8)
    return out


def build_preview(
    rgb: np.ndarray,
    *,
    colors: int,
    clusters_to_show: int,
    meanshift_sp: float,
    meanshift_sr: float,
    spatial_weight: float,
    mrf_beta: float,
    mrf_iterations: int,
    cleanup_min_px: int,
    cleanup_min_pct: float,
    output: str,
) -> None:
    raw_labels, raw_quant = kmeans_labels(rgb, colors)
    ms_rgb = meanshift_prefilter(rgb, meanshift_sp, meanshift_sr)
    ms_labels, ms_quant = kmeans_labels(ms_rgb, colors)
    spatial_labels, spatial_quant = spatial_kmeans_labels(rgb, colors, spatial_weight)
    mrf_labels = potts_smooth_labels(rgb, raw_labels, beta=mrf_beta, iterations=mrf_iterations)
    mrf_quant = quantize_from_labels(rgb, mrf_labels)
    clean_labels = cleanup_label_regions(
        raw_labels.astype(np.uint8),
        min_region_px=int(cleanup_min_px),
        min_region_pct=float(cleanup_min_pct),
    ).astype(np.int32)
    clean_quant = quantize_from_labels(rgb, clean_labels)

    variants = [
        ("Original", rgb),
        ("Raw K-Means quantized", raw_quant),
        ("Mean-shift + K-Means quantized", ms_quant),
        ("Spatial K-Means quantized", spatial_quant),
        ("Potts/MRF-smoothed K-Means", mrf_quant),
        ("Connected-component cleanup", clean_quant),
    ]

    label_variants = [
        ("Raw K-Means labels", raw_labels),
        ("Mean-shift labels", ms_labels),
        ("Spatial K-Means labels", spatial_labels),
        ("Potts/MRF labels", mrf_labels),
        ("Component-cleaned labels", clean_labels),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (title, image) in zip(axes.reshape(-1), variants):
        ax.imshow(image, cmap="tab20" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Quantized outputs: colors={colors}, mean-shift sp={meanshift_sp}/sr={meanshift_sr}, "
        f"spatial_weight={spatial_weight}, mrf_beta={mrf_beta}, cleanup_px={cleanup_min_px}",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

    mask_output = out.with_name(out.stem + "_masks" + out.suffix)
    rows = max(1, clusters_to_show)
    cols = len(label_variants) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    ids_by_variant = [(title, labels, cluster_ids_by_area(labels, rows)) for title, labels in label_variants]
    for row in range(rows):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Original reference")
        axes[row, 0].axis("off")
        for col, (title, labels, ids) in enumerate(ids_by_variant, start=1):
            cid = ids[row]
            axes[row, col].imshow(mask_image(labels, cid))
            axes[row, col].set_title(f"{title}\nmask #{cid}")
            axes[row, col].axis("off")
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Largest cluster masks: speckle comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(mask_output, dpi=160)
    plt.close(fig)
    print(f"Saved preview to {out}")
    print(f"Saved mask preview to {mask_output}")


def parse_crop(value: str | None):
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview clustering speckle without running the full PDF pipeline.")
    parser.add_argument("--input", default="pics/33.jpg", help="Input image to cluster.")
    parser.add_argument("--output", default="outputs/cluster_preview.png", help="Preview image path.")
    parser.add_argument("--colors", type=int, default=15, help="Number of K-Means colors.")
    parser.add_argument("--resize-max", type=int, default=900, help="Resize longest side for fast preview; 0 disables.")
    parser.add_argument("--crop", type=parse_crop, default=None, help="Optional crop as x,y,w,h after resizing.")
    parser.add_argument("--clusters-to-show", type=int, default=2, help="How many large cluster masks to show.")
    parser.add_argument("--meanshift-sp", type=float, default=12.0, help="Mean-shift spatial radius.")
    parser.add_argument("--meanshift-sr", type=float, default=18.0, help="Mean-shift color radius.")
    parser.add_argument("--spatial-weight", type=float, default=4.0, help="Spatial XY weight for spatial K-Means.")
    parser.add_argument("--mrf-beta", type=float, default=7.0, help="Neighbor disagreement penalty for Potts smoothing.")
    parser.add_argument("--mrf-iterations", type=int, default=6, help="ICM iterations for Potts smoothing.")
    parser.add_argument("--cleanup-min-px", type=int, default=80, help="Minimum connected component size.")
    parser.add_argument("--cleanup-min-pct", type=float, default=0.0, help="Minimum component size as image percent.")
    args = parser.parse_args()

    rgb = load_rgb(args.input)
    rgb = resize_max(rgb, args.resize_max)
    rgb = crop_rgb(rgb, args.crop)
    build_preview(
        rgb,
        colors=args.colors,
        clusters_to_show=max(1, args.clusters_to_show),
        meanshift_sp=args.meanshift_sp,
        meanshift_sr=args.meanshift_sr,
        spatial_weight=args.spatial_weight,
        mrf_beta=args.mrf_beta,
        mrf_iterations=args.mrf_iterations,
        cleanup_min_px=args.cleanup_min_px,
        cleanup_min_pct=args.cleanup_min_pct,
        output=args.output,
    )


if __name__ == "__main__":
    main()
