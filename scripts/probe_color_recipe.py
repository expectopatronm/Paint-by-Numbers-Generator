from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbn import color as color_space
from pbn import mixing
from pbn.color import deltaE_lab, lab_to_rgb8, rgb_to_hsv
from pbn.config import BASE_PALETTE, DARKEN_PER_PIGMENT, DEFAULT_CONFIG
from pbn.image_ops import _map_stencil_brightness_slider
from pbn.mixing import continuous_mix_best, integer_mix_best, recipe_text, stochastic_integer_mix_best


def rgbrows_to_labrows(arr_uint8: np.ndarray) -> np.ndarray:
    rgb = arr_uint8.astype(np.float32) / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151520, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    t = xyz / white
    eps = (6 / 29) ** 3
    f = np.where(t > eps, np.cbrt(t), (1 / 3) * (29 / 6) ** 2 * t + 4 / 29)
    lab = np.empty_like(f, dtype=np.float32)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])
    return lab


def load_cluster_targets(config: dict) -> np.ndarray:
    args = SimpleNamespace(**config)
    img = Image.open(args.input)
    img = ImageOps.exif_transpose(img).convert("RGB")

    pct = float(getattr(args, "pre_brighten_pct", 0))
    if pct > 0:
        factor = _map_stencil_brightness_slider(pct / 100.0)
        factor = 1.0 + (max(0.0, min(100.0, pct)) / 100.0)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if args.resize:
        width, height = map(int, args.resize)
        img = img.resize((width, height), resample=Image.BILINEAR)

    pixels = np.array(img).reshape((-1, 3)).astype(np.uint8)
    feats = rgbrows_to_labrows(pixels)
    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=8)
    kmeans.fit(feats)

    centroids_rgb = [np.clip(np.rint(lab_to_rgb8(lab)), 0, 255) for lab in kmeans.cluster_centers_.astype(np.float32)]
    return np.array(centroids_rgb, dtype=np.uint8)


def load_recipe_cache(path: str | Path):
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def cache_entry_for_color(cache: dict, color_number: int):
    for entry in cache.get("color_targets", []):
        if int(entry.get("color_number", -1)) == int(color_number):
            return entry
    return None


def print_result(label: str, entries, rgb, err):
    rgb_i = tuple(np.rint(rgb).astype(int))
    hsv = rgb_to_hsv(rgb)
    print(f"\n{label}", flush=True)
    print("-" * len(label), flush=True)
    print(f"Recipe: {recipe_text(entries)}", flush=True)
    print(f"Predicted RGB: {rgb_i}", flush=True)
    print(f"HSV: h={hsv[0]:.1f}, s={hsv[1]:.3f}, v={hsv[2]:.3f}", flush=True)
    print(f"Delta E: {err:.4f}", flush=True)


def parse_recipe(text: str):
    entries = []
    if not text:
        return entries
    for chunk in text.split(","):
        name, amount = chunk.rsplit(":", 1)
        entries.append((name.strip(), int(amount.strip())))
    return entries


def entries_from_cache(entry: dict):
    return [(str(item["pigment"]), int(item["parts"])) for item in entry.get("recipe_entries", [])]


def main():
    parser = argparse.ArgumentParser(description="Probe recipe optimization for one paint-by-numbers color.")
    parser.add_argument("--color", type=int, default=20, help="1-based color number from the PDF.")
    parser.add_argument("--target-rgb", nargs=3, type=int, default=None, metavar=("R", "G", "B"))
    parser.add_argument("--cache", default="", help="Recipe target cache JSON. Defaults to config recipe_cache.")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--max-parts", type=int, default=20)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--candidate-pigments", type=int, default=11)
    parser.add_argument("--max-combos", type=int, default=32)
    parser.add_argument("--optimizer-iterations", type=int, default=180)
    parser.add_argument("--optimizer-starts", type=int, default=8)
    parser.add_argument("--stochastic-samples", type=int, default=12000)
    parser.add_argument("--integer-max-parts", type=int, default=None)
    parser.add_argument("--integer-components", type=int, default=None)
    parser.add_argument("--skip-integer", action="store_true")
    parser.add_argument(
        "--seed-recipe",
        default="",
        help="Comma-separated recipe, e.g. burnt_sienna:1,cobalt_blue:2,titanium_white:5",
    )
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    mixing.BASE_PALETTE = BASE_PALETTE
    mixing.DARKEN_PER_PIGMENT = DARKEN_PER_PIGMENT
    color_space.DELTA_E_METHOD = str(cfg.get("delta_e_method", "colour_ciede2000"))

    cache = None if args.ignore_cache else load_recipe_cache(args.cache or cfg.get("recipe_cache", "outputs/recipe_targets.json"))
    cache_entry = cache_entry_for_color(cache, args.color) if cache else None

    if args.target_rgb is not None:
        target = np.array(args.target_rgb, dtype=float)
    elif cache_entry is not None:
        target = np.array(cache_entry["target_rgb"], dtype=float)
        print(f"Loaded target from cache: {args.cache or cfg.get('recipe_cache')}", flush=True)
    else:
        centroids = load_cluster_targets(cfg)
        if args.color < 1 or args.color > len(centroids):
            raise SystemExit(f"--color must be between 1 and {len(centroids)}")
        target = centroids[args.color - 1].astype(float)

    print(f"Target color #{args.color}: {tuple(np.rint(target).astype(int))}", flush=True)
    print(f"Palette size: {len(cfg['palette'])}", flush=True)

    seed_entries = parse_recipe(args.seed_recipe)
    if not seed_entries and cache_entry is not None:
        seed_entries = entries_from_cache(cache_entry)
        if seed_entries:
            print(f"Loaded seed recipe from cache: {recipe_text(seed_entries)}", flush=True)
    if args.skip_integer:
        if not seed_entries:
            raise SystemExit("--skip-integer requires --seed-recipe")
        names = cfg["palette"]
        base = np.array([BASE_PALETTE[n] for n, _p in seed_entries], dtype=float)
        parts = np.array([p for _n, p in seed_entries], dtype=float)
        rgb = mixing.mix_learned(parts, base, [n for n, _p in seed_entries])
        err = deltaE_lab(rgb, target)
        entries = seed_entries
        print_result("Seed recipe baseline", entries, rgb, err)
    else:
        integer_max_parts = args.integer_max_parts or int(cfg["max_parts"])
        integer_components = args.integer_components or int(cfg["components"])
        entries, rgb, err = integer_mix_best(
            target,
            cfg["palette"],
            max_parts=integer_max_parts,
            max_components=integer_components,
        )
        print_result(f"Integer baseline ({integer_max_parts} parts, {integer_components} components)", entries, rgb, err)

    cont_entries, cont_rgb, cont_err = continuous_mix_best(
        target,
        cfg["palette"],
        max_parts=args.max_parts,
        max_components=args.components,
        candidate_pigments=args.candidate_pigments,
        max_combos=args.max_combos,
        seed_entries=entries,
        maxiter=args.optimizer_iterations,
        starts=args.optimizer_starts,
    )
    print_result(
        f"Continuous fallback ({args.max_parts} parts, {args.components} components)",
        cont_entries,
        cont_rgb,
        cont_err,
    )

    stochastic_entries, stochastic_rgb, stochastic_err = stochastic_integer_mix_best(
        target,
        cfg["palette"],
        max_parts=args.max_parts,
        max_components=args.components,
        samples=args.stochastic_samples,
        seed_entries=cont_entries if cont_err < err else entries,
    )
    print_result(
        f"Stochastic integer fallback ({args.max_parts} parts, {args.components} components)",
        stochastic_entries,
        stochastic_rgb,
        stochastic_err,
    )

    best_err = min(err, cont_err, stochastic_err)
    if best_err <= 1.0:
        print("\nPASS: Delta E is <= 1.0", flush=True)
    else:
        print("\nMISS: Delta E is still > 1.0", flush=True)


if __name__ == "__main__":
    main()
