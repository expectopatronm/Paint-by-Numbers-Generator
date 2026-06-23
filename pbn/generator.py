from __future__ import annotations

import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageEnhance, ImageOps
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from . import color as color_space
from . import mixing
from .build_order import BuildPolicy, _entries_to_vec, draw_build_graph_page, plan_build_order_configurable
from .color import Lstar_from_rgb, deltaE_lab, lab_to_rgb8, rgb8_to_lab
from .config import BASE_PALETTE, DARKEN_PER_PIGMENT, DEFAULT_CONFIG
from .image_ops import (
    _adjust_brightness_rgb,
    _adjust_sharpness_rgb,
    _apply_clean_stencil_rgb,
    _map_stencil_brightness_slider,
    add_grid_to_rgb,
    adjacent_color_order,
    build_value_tweaks,
    cleanup_label_regions,
    group_classic_exclusive,
    group_value5_exclusive,
    load_external_sketch_gray,
    original_edge_sketch_with_grid,
    potts_smooth_labels,
    render_image_sketch_gray,
)
from .imprimatura import _prev_highlight_rgb, choose_imprimatura_target_from_image
from .integrations import _maybe_upscale_input, rmbg_alpha_matte
from .mixing import _init_worker, _recipe_worker, genetic_integer_mix_best, integer_mix_best, recipe_text
from .pdf_render import draw_color_key, new_fig
from .svg_trace import _auto_grid_step, run_centerline_trace

DELTA_E_METHOD = "colour_ciede2000"

# ---------------------------
# Main (DICT config, no argparse)
# ---------------------------
def main(config: dict | None = None):
    """
    Run the generator using a dict-based config.
    Pass only the keys you want to override; unspecified keys use DEFAULT_CONFIG.
    """
    global BASE_PALETTE, DELTA_E_METHOD

    t0 = time.perf_counter()  # <<< start timer

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    args = SimpleNamespace(**cfg)

    # --- Pre-upscale / restoration gate ---
    args.input = _maybe_upscale_input(args.input, args)

    # set global toggles from config (existing)
    DELTA_E_METHOD = str(getattr(args, "delta_e_method", "colour_ciede2000"))
    mixing.BASE_PALETTE = BASE_PALETTE
    mixing.DARKEN_PER_PIGMENT = DARKEN_PER_PIGMENT
    color_space.DELTA_E_METHOD = DELTA_E_METHOD

    # Map the alias onto outline-mode (label-boundary modes were removed).
    if args.sketch_style:
        if str(args.sketch_style).lower() not in ("old", "image"):
            print(f"sketch_style={args.sketch_style!r} is deprecated; using image sketch instead.")
        args.outline_mode = "image"

    # -------------------------
    # Load twice: one for OUTLINE (no pre-brighten), one for COLORING (pre-brighten)
    # -------------------------
    print("[1/6] Loading image and preparing inputs...")
    img_outline = Image.open(args.input)  # upscaled (if enabled), NOT pre-brightened
    img_outline = ImageOps.exif_transpose(img_outline).convert("RGB")
    img = img_outline.copy()  # this copy will be pre-brightened for clustering/coloring

    # --- PRE-BRIGHTEN: apply a 0..100% increase (mapped to factor 1.00..2.00)
    try:
        pct = float(getattr(args, "pre_brighten_pct", 0))
        if pct > 0:
            factor = _map_stencil_brightness_slider(pct / 100.0)  # or keep your factor logic as-is
            factor = 1.0 + (max(0.0, min(100.0, pct)) / 100.0)
            img = ImageEnhance.Brightness(img).enhance(factor)
            print(f"Pre-brighten applied: +{pct:.1f}")
        else:
            print("Pre-brighten skipped (pre_brighten_pct=0).")
    except Exception as e:
        print(f"Pre-brighten failed ({e}) — continuing with unmodified image.")

    orig_w, orig_h = img.size

    # If grid_step is "auto" ...
    if (getattr(args, "grid_step", None) in (None, "auto")) or (
            isinstance(args.grid_step, (int, float)) and args.grid_step <= 0):
        args.grid_step = _auto_grid_step(orig_w, getattr(args, "grid_min_cols", 5))

    # COLORING (pre-brightened) tensors
    rgb_full = np.array(img)
    bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)

    # OUTLINE (upscaled, non-brightened) tensors
    rgb_outline_full = np.array(img_outline)
    bgr_outline_full = cv2.cvtColor(rgb_outline_full, cv2.COLOR_RGB2BGR)

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

    # K-Means clustering in perceptual LAB space.
    def rgbrow_to_labrows(arr_uint8):
        labs = []
        for r, g, b in arr_uint8:
            lab = rgb8_to_lab(np.array([r, g, b], dtype=np.float32))
            labs.append(lab)
        return np.array(labs, dtype=np.float32)

    print("[2/6] Clustering image colors...")
    feats = rgbrow_to_labrows(pixels_small.astype(np.uint8))
    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=8)
    kmeans.fit(feats)
    labels_small = kmeans.labels_.reshape(Hs, Ws).astype(np.uint8)

    centroids_lab = kmeans.cluster_centers_.astype(np.float32)
    centroids_rgb = [np.clip(np.rint(lab_to_rgb8(lab)), 0, 255) for lab in centroids_lab]
    centroids = np.array(centroids_rgb, dtype=np.uint8)

    # Upsample labels to full res
    labels_full = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_full = np.array(labels_full, dtype=np.uint8)

    print("[3/6] Smoothing and cleaning color regions...")
    if bool(getattr(args, "mrf_smoothing", True)):
        print(
            "Applying Potts/MRF label smoothing "
            f"(beta={float(args.mrf_beta)}, iterations={int(args.mrf_iterations)})."
        )
        labels_full = potts_smooth_labels(
            rgb_full,
            labels_full,
            centroids_rgb_u8=centroids,
            beta=float(args.mrf_beta),
            iterations=int(args.mrf_iterations),
        )

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

    print(f"[4/6] Building paint recipes for {len(centroids_list)} colors...")
    if args.parallel and len(centroids_list) > 1:
        max_workers = int(args.workers) if args.workers else (os.cpu_count() or 1)
        # Guard against silly over-commit (optional, but polite)
        max_workers = max(1, min(max_workers, len(centroids_list)))

        tasks = {}
        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(
                    BASE_PALETTE,
                    DARKEN_PER_PIGMENT,
                    str(args.delta_e_method),
                ),
        ) as ex:
            for idx, c in enumerate(centroids_list):
                fut = ex.submit(
                    _recipe_worker,
                    c,
                    names,
                    int(args.max_parts),
                    int(args.components),
                    float(args.get("prefer_simple_lambda_components", 0.03) if isinstance(args, dict) else getattr(args,
                                                                                                                   "prefer_simple_lambda_components",
                                                                                                                   0.03) if hasattr(
                        args, "prefer_simple_lambda_components") else 0.03),
                    float(args.get("prefer_simple_lambda_parts", 0.01) if isinstance(args, dict) else getattr(args,
                                                                                                              "prefer_simple_lambda_parts",
                                                                                                              0.01) if hasattr(
                        args, "prefer_simple_lambda_parts") else 0.01),
                    bool(getattr(args, "genetic_retry_enabled", True)),
                    float(getattr(args, "genetic_retry_delta_e", 1.0)),
                    int(getattr(args, "genetic_retry_max_parts", 20)),
                    int(getattr(args, "genetic_retry_components", 10)),
                    int(getattr(args, "genetic_retry_population", 180)),
                    int(getattr(args, "genetic_retry_generations", 160)),
                )

                tasks[fut] = idx

            results = [None] * len(centroids_list)
            for fut in tqdm(
                as_completed(tasks),
                total=len(tasks),
                desc="Recipes",
                unit="color",
            ):
                results[tasks[fut]] = fut.result()

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
        for col in tqdm(centroids.astype(np.float32), total=len(centroids), desc="Recipes", unit="color"):
            entries, approx_rgb, err = integer_mix_best(
                col,
                names,
                max_parts=args.max_parts,
                max_components=args.components,
            )
            if bool(getattr(args, "genetic_retry_enabled", True)) and err > float(getattr(args, "genetic_retry_delta_e", 1.0)):
                genetic_entries, genetic_rgb, genetic_err = genetic_integer_mix_best(
                    col,
                    names,
                    max_parts=int(getattr(args, "genetic_retry_max_parts", 20)),
                    max_components=int(getattr(args, "genetic_retry_components", 10)),
                    population=int(getattr(args, "genetic_retry_population", 180)),
                    generations=int(getattr(args, "genetic_retry_generations", 160)),
                    seed_entries=entries,
                )
                if genetic_err < err:
                    entries, approx_rgb, err = genetic_entries, genetic_rgb, genetic_err
            all_entries.append(entries)
            all_recipes.append(recipe_text(entries))
            approx_rgbs.append(np.array(approx_rgb, dtype=float))
            deltaEs.append(err)

    approx_uint8 = np.clip(np.rint(np.array(approx_rgbs)), 0, 255).astype(np.uint8)

    if bool(getattr(args, "write_recipe_cache", True)):
        cache_path = str(getattr(args, "recipe_cache", "outputs/recipe_targets.json"))
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            cache_payload = {
                "input": str(args.input),
                "colors": int(args.colors),
                "palette": list(args.palette),
                "max_parts": int(args.max_parts),
                "components": int(args.components),
                "delta_e_method": str(args.delta_e_method),
                "genetic_retry_enabled": bool(getattr(args, "genetic_retry_enabled", True)),
                "genetic_retry_delta_e": float(getattr(args, "genetic_retry_delta_e", 1.0)),
                "genetic_retry_max_parts": int(getattr(args, "genetic_retry_max_parts", 20)),
                "genetic_retry_components": int(getattr(args, "genetic_retry_components", 10)),
                "genetic_retry_population": int(getattr(args, "genetic_retry_population", 180)),
                "genetic_retry_generations": int(getattr(args, "genetic_retry_generations", 160)),
                "color_targets": [
                    {
                        "color_number": int(i + 1),
                        "target_rgb": [int(x) for x in np.rint(centroids[i]).astype(int).tolist()],
                        "recipe_entries": [{"pigment": str(n), "parts": int(p)} for n, p in all_entries[i]],
                        "recipe_text": str(all_recipes[i]),
                        "predicted_rgb": [int(x) for x in np.rint(approx_uint8[i]).astype(int).tolist()],
                        "delta_e": float(deltaEs[i]),
                    }
                    for i in range(args.colors)
                ],
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)
            print(f"Recipe target cache saved: {cache_path}")
        except Exception as e:
            print(f"Recipe target cache skipped ({e}).")

    # PBN image from MIXED palette using the final smoothed/cleaned labels.
    print("[5/6] Planning painting stages...")
    pbn_image = approx_uint8[labels_full]

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
    # Outline prep (external sketch or image-edge sketch)
    # -------------------------
    uses_external_sketch = bool(getattr(args, "external_sketch", None))
    if uses_external_sketch:
        external_sketch_path = str(getattr(args, "external_sketch"))
        outline_gray = load_external_sketch_gray(external_sketch_path, (orig_w, orig_h))
        sketch_gray = outline_gray
        print(f"Using external sketch file: {external_sketch_path}")
    else:
        outline_gray = render_image_sketch_gray(bgr_outline_full, args)
        sketch_gray = outline_gray

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
    print("[6/6] Rendering PDF pages...")
    A4_LANDSCAPE = (11.69, 8.27)
    with (PdfPages(args.pdf) as pdf):
        # Page 1
        fig = new_fig(A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1.0, 1.55], figure=fig, wspace=0.01, hspace=0.03)
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[:, 1])
        ax1.imshow(rgb_full); ax1.set_title("Original", pad=2); ax1.axis("off")
        ax2.imshow(pbn_image)
        ax2.set_title(f"Paint by Numbers ({args.colors} colors) • cluster=LAB K-Means • mixmodel=learned • max parts≤{args.max_parts}", pad=2)
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

        imp_mode = getattr(args, "imprimatura_mode", "match_light")
        imp_target_rgb = choose_imprimatura_target_from_image(
            rgb_full, mode=imp_mode, target_L=50.0, chroma_s=0.25
        )

        imp_entries, imp_rgb, imp_dE = integer_mix_best(
            imp_target_rgb,
            args.palette,
            max_parts=6,
            max_components=3,
        )
        imp_rgb = np.clip(np.rint(imp_rgb), 0, 255).astype(np.uint8)
        imp_L = Lstar_from_rgb(imp_rgb)

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
                title="Build Dependency Graph (Neutral, additive steps)",
                imprimatura={
                    "rgb": imp_rgb,
                    "recipe_text": recipe_text(imp_entries),
                    "Lstar": Lstar_from_rgb(imp_rgb),
                    "deltaE": deltaE_lab(imp_rgb, imp_target_rgb),
                }
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
                mode_tag = "external" if uses_external_sketch else "image"
                ax.imshow(ref_with_grid)
                ax.set_title(f"Clean Stencil Outline + Grid ({mode_tag}) (step={args.grid_step}px)", pad=2)
                ax.axis("off")
                pdf.savefig(fig, dpi=300);
                plt.close(fig)

            else:
                # Legacy behavior (no clean stencil): keep the old special page for image-edges;
                # otherwise show the raw outline/combined outline.
                if uses_external_sketch:
                    ref_rgb = cv2.cvtColor(outline_gray, cv2.COLOR_GRAY2RGB)
                    ref_with_grid = add_grid_to_rgb(ref_rgb, grid_step=args.grid_step, grid_color=200)
                    ax.imshow(ref_with_grid)
                    ax.set_title(f"External Sketch + Grid (step={args.grid_step}px)", pad=2)
                    ax.axis("off")
                    pdf.savefig(fig, dpi=300);
                    plt.close(fig)
                else:
                    legacy_page = original_edge_sketch_with_grid(
                        img_outline,
                        grid_step=args.grid_step,
                        grid_color=200,
                        canny_high_pct=float(args.edge_percentile),
                    )

                    ax.imshow(legacy_page, cmap='gray')
                    ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px)", pad=2)
                    ax.axis("off")
                    pdf.savefig(fig, dpi=300);
                    plt.close(fig)

        # Step pages
        for title, idxs, frame in tqdm(frames_to_emit, desc="Stage pages", unit="page"):
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

        # --- Optional: build FG/BG masks via RMBG ---
        fg_mask = bg_mask = None
        if bool(getattr(args, "separate_fg_bg", False)):
            try:
                matte = rmbg_alpha_matte(
                    args.input,
                    model_dir=str(getattr(args, "rmbg_model_dir", "rmbg")),
                    device=getattr(args, "rmbg_device", None),
                    target_size=tuple(getattr(args, "rmbg_target_size", (1024, 1024))),
                )
                thr = float(getattr(args, "rmbg_alpha_threshold", 0.5))
                fg_mask = (matte >= thr)
                bg_mask = ~fg_mask
                print(f"RMBG matte ready (threshold={thr}). "
                      f"FG px={int(fg_mask.sum())}, BG px={int(bg_mask.sum())}")
            except Exception as e:
                print(f"RMBG failed ({e}). Proceeding without FG/BG split.")
                fg_mask = bg_mask = None

        # -------------------------
        # Per-color pages (configurable order)
        # -------------------------
        if args.per_color_frames:

            # Ensure multiply-underlay factor if outline exists (unchanged)
            a = float(np.clip(args.sketch_alpha, 0.0, 1.0))
            if sketch_factor_rgb is None and outline_gray is not None:
                s = np.clip(outline_gray.astype(np.float32) / 255.0, 0.0, 1.0)
                sketch_factor_rgb = ((1.0 - a) + a * s)[..., None]

            H, W = labels_full.shape
            area = np.array([(labels_full == i).sum() for i in range(args.colors)], dtype=np.int64)

            def painting_so_far_with_grid(done_mask: np.ndarray) -> np.ndarray:
                progress_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)
                if done_mask.any():
                    progress_rgb[done_mask] = pbn_image[done_mask]
                if sketch_factor_rgb is not None:
                    progress_f = np.clip(progress_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                    progress_rgb = (np.clip(progress_f * sketch_factor_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                return add_grid_to_rgb(progress_rgb, grid_step=args.grid_step, grid_color=200)

            # --- Decide per-color order mode ---
            order_mode = getattr(args, "per_color_order_mode")
            if order_mode is None:
                order_mode = "adjacent"

            # --- Build per_color_order according to mode ---
            per_color_order: List[int] = []

            if order_mode == "adjacent":
                # Grow from already-painted regions whenever possible.
                per_color_order = adjacent_color_order(labels_full, approx_uint8, area)
            else:
                # Old behavior: follow the stepwise frames order
                for _title, idxs, _frame in frames_to_emit:
                    for idx in sorted(idxs, key=lambda i: -int(area[i])):
                        if idx not in per_color_order:
                            per_color_order.append(idx)

            # Safety: ensure all colors are in the list
            for i in range(args.colors):
                if i not in per_color_order:
                    per_color_order.append(i)

            # When FG/BG split is on, split the selected order into two tracks.
            if bool(getattr(args, "separate_fg_bg", False)) and (fg_mask is not None):
                # Helper: does color i have any pixels in FG or BG?
                def has_pixels(i, mask):
                    return int(np.logical_and(labels_full == i, mask).sum()) > 0

                bg_order, fg_order = [], []
                for idx in per_color_order:
                    if has_pixels(idx, bg_mask):
                        bg_order.append(idx)
                    if has_pixels(idx, fg_mask):
                        fg_order.append(idx)

                def render_pages(order, which_mask_name, which_mask, seed_prev_mask=None):
                    prev_mask = (seed_prev_mask.copy()
                                 if seed_prev_mask is not None
                                 else np.zeros((H, W), dtype=bool))
                    for i in tqdm(order, desc=f"Per-color pages ({which_mask_name})", unit="page"):
                        # pixels for this color restricted to which_mask
                        curr_mask = np.logical_and(labels_full == i, which_mask)

                        # skip empty
                        if not curr_mask.any():
                            continue

                        frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                        # Optional cumulative display (includes seeded BG if provided)
                        if args.per_color_cumulative and args.prev_alpha > 0 and prev_mask.any():
                            sel_prev = prev_mask
                            hl = _prev_highlight_rgb(args)
                            alpha = float(getattr(args, "prev_alpha", 0.50))
                            white_f = 255.0

                            if hl is not None:
                                frame_rgb[sel_prev] = (
                                        (1.0 - alpha) * white_f + alpha * hl.astype(np.float32)
                                ).round().astype(np.uint8)
                            else:
                                frame_rgb[sel_prev] = (
                                        (1.0 - alpha) * white_f + alpha * pbn_image[sel_prev].astype(np.float32)
                                ).round().astype(np.uint8)

                        # Current color regions (restricted to FG or BG)
                        frame_rgb[curr_mask] = pbn_image[curr_mask]

                        # Multiply blend with outline if present
                        if sketch_factor_rgb is not None:
                            frame_f = np.clip(frame_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                            composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                            composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
                        else:
                            composite_u8 = frame_rgb

                        frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)
                        after_mask = np.logical_or(prev_mask, curr_mask)

                        # Layout
                        fig = new_fig((11.69, 8.27))
                        gs = GridSpec(
                            2, 3,
                            width_ratios=[2.7, 1.0, 1.15],
                            height_ratios=[1, 1],
                            figure=fig,
                            wspace=0.025,
                            hspace=0.04
                        )

                        axL = fig.add_subplot(gs[:, 0])
                        axBefore = fig.add_subplot(gs[0, 1])
                        axAfter = fig.add_subplot(gs[1, 1])
                        axKey = fig.add_subplot(gs[:, 2])

                        # Left: current per-color frame
                        axL.imshow(frame_with_grid)
                        role = "Background" if which_mask_name == "bg" else "Foreground"
                        axL.set_title(
                            (f"Per-Color • #{i + 1} ({role}) + Grid "
                             f"{'(cumulative, prevα=' + str(args.prev_alpha) + ')' if args.per_color_cumulative else ''} "
                             f"{'(outline multiply underlay)' if sketch_factor_rgb is not None else ''}"),
                            pad=2
                        )
                        axL.axis("off")

                        # Middle column: painting state before and after this page's color.
                        axBefore.imshow(painting_so_far_with_grid(prev_mask))
                        axBefore.set_title("Painting so far", fontsize=9, pad=4)
                        axBefore.axis("off")

                        axAfter.imshow(painting_so_far_with_grid(after_mask))
                        axAfter.set_title("After this color", fontsize=9, pad=4)
                        axAfter.axis("off")

                        # Bottom-right: existing color key panel
                        draw_color_key(
                            axKey, centroids, all_recipes, all_entries, BASE_PALETTE,
                            used_indices=[i],
                            title=f"Color Key • Color #{i + 1} ({role})",
                            tweaks=tweaks,
                            wrap_width=max(30, int(args.wrap * 0.7)),
                            show_components=not args.hide_components,
                            deltaEs=deltaEs,
                            left_pad=1.25, right_margin=0.18, text_gap=0.05,
                            approx_palette=approx_uint8
                        )
                        axKey.text(0.05, 0.05, f"Color #{i + 1}", fontsize=8, transform=axKey.transAxes)
                        axKey.axis("off")

                        pdf.savefig(fig, dpi=300)
                        plt.close(fig)

                        # Update cumulative mask for next page in this track
                        if args.per_color_cumulative:
                            prev_mask |= curr_mask
                        else:
                            prev_mask[:] = False

                    # hand back whatever we've accumulated (so BG can seed FG)
                    return prev_mask

                # First emit *background* pages and capture their cumulative mask
                bg_cumulative = render_pages(bg_order, "bg", bg_mask)

                # Then emit *foreground* pages, including all background done so far
                render_pages(fg_order, "fg", fg_mask, seed_prev_mask=bg_cumulative)


            else:
                # --- Original behavior (no split) ---
                prev_mask = np.zeros((H, W), dtype=bool)
                for i in tqdm(per_color_order, desc="Per-color pages", unit="page"):
                    curr_mask = (labels_full == i)
                    frame_rgb = np.full_like(pbn_image, 255, dtype=np.uint8)

                    if args.per_color_cumulative and args.prev_alpha > 0 and prev_mask.any():
                        sel_prev = prev_mask
                        hl = _prev_highlight_rgb(args)
                        alpha = float(getattr(args, "prev_alpha", 0.50))
                        white_f = 255.0

                        if hl is not None:
                            # Blend WHITE with the chosen neon highlight color, not with original colors
                            # (shape-broadcasting works because sel_prev masks a Nx3 view)
                            frame_rgb[sel_prev] = (
                                    (1.0 - alpha) * white_f + alpha * hl.astype(np.float32)
                            ).round().astype(np.uint8)
                        else:
                            # Original behavior: blend WHITE with the previous regions' original colors
                            frame_rgb[sel_prev] = (
                                    (1.0 - alpha) * white_f + alpha * pbn_image[sel_prev].astype(np.float32)
                            ).round().astype(np.uint8)

                    frame_rgb[curr_mask] = pbn_image[curr_mask]

                    if sketch_factor_rgb is not None:
                        frame_f = np.clip(frame_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
                        composite = np.clip(frame_f * sketch_factor_rgb, 0.0, 1.0)
                        composite_u8 = (composite * 255.0 + 0.5).astype(np.uint8)
                    else:
                        composite_u8 = frame_rgb

                    frame_with_grid = add_grid_to_rgb(composite_u8, grid_step=args.grid_step, grid_color=200)
                    after_mask = np.logical_or(prev_mask, curr_mask)

                    fig = new_fig((11.69, 8.27))
                    gs = GridSpec(
                        2, 3,
                        width_ratios=[2.7, 1.0, 1.15],
                        height_ratios=[1, 1],
                        figure=fig,
                        wspace=0.025,
                        hspace=0.04
                    )

                    axL = fig.add_subplot(gs[:, 0])  # left spans both rows
                    axBefore = fig.add_subplot(gs[0, 1])
                    axAfter = fig.add_subplot(gs[1, 1])
                    axKey = fig.add_subplot(gs[:, 2])

                    # Left: current per-color frame
                    axL.imshow(frame_with_grid)
                    axL.set_title((f"Per-Color • #{i + 1} + Grid "
                                   f"{'(cumulative, prevα=' + str(args.prev_alpha) + ')' if args.per_color_cumulative else ''} "
                                   f"{'(outline multiply underlay)' if sketch_factor_rgb is not None else ''}"), pad=2)
                    axL.axis("off")

                    # Middle column: painting state before and after this page's color.
                    axBefore.imshow(painting_so_far_with_grid(prev_mask))
                    axBefore.set_title("Painting so far", fontsize=9, pad=4)
                    axBefore.axis("off")

                    axAfter.imshow(painting_so_far_with_grid(after_mask))
                    axAfter.set_title("After this color", fontsize=9, pad=4)
                    axAfter.axis("off")

                    # Bottom-right: what you already had (color key)
                    draw_color_key(
                        axKey, centroids, all_recipes, all_entries, BASE_PALETTE,
                        used_indices=[i],
                        title=f"Color Key • Color #{i + 1}",
                        tweaks=tweaks,
                        wrap_width=max(30, int(args.wrap * 0.7)),
                        show_components=not args.hide_components,
                        deltaEs=deltaEs,
                        left_pad=1.25, right_margin=0.18, text_gap=0.05,
                        approx_palette=approx_uint8
                    )
                    axKey.text(0.05, 0.05, f"Color #{i + 1}", fontsize=8, transform=axKey.transAxes)
                    axKey.axis("off")

                    pdf.savefig(fig, dpi=300)
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

    print(f"Saved A4 landscape PDF to {args.pdf} (frame-mode={args.frame_mode}, outline={args.outline_mode})")

    # --------------------------------------------------
    # Optional: export centerline SVG for Inkscape plotting
    # --------------------------------------------------
    if args.export_centerline_svg:
        # Attach stencil to args so the helper can access it
        args.outline_gray = locals().get("outline_gray", None)
        run_centerline_trace(args)

    # <<< end timer + print
    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.2f}s")

