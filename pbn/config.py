from __future__ import annotations

# Base tube pigment palette
# ---------------------------
BASE_PALETTE = {
    "alizarin_crimson": (82, 17, 28),
    "burnt_sienna": (86, 35, 33),
    "burnt_umber": (44, 28, 22),
    "cobalt_blue": (36, 42, 85),
    "indian_yellow": (219, 125, 21),
    # "indigo": (14, 17, 33),
    "ivory_black": (15, 18, 22),
    "olive_green": (73, 88, 37),
    "paynes_gray": (13, 20, 25),
    "titanium_white": (247, 247, 241),
    "vandyke_brown": (27, 20, 13),
    "yellow_ochre": (187, 128, 35)
}

# Dark earths + blacks (burnt_sienna, burnt_umber, ivory_black, paynes_gray, vandyke_brown) at 0.75:
# These will reliably make mixes dry darker.
# Using them heavily will overshoot into darkness unless balanced with white.
# Strong chromatic colors (alizarin_crimson, cobalt_blue, indian_yellow, olive_green, yellow_ochre) at 1.0:
# These don’t add extra darkening or lightening in your model.
# They behave like “honest citizens” in terms of value.
# Titanium white at 1.25:
# Acts as a stabilizer and lightener in the drying model.
# Makes white a powerful tool for keeping mixes from sinking into too much darkness.
# Indirectly encourages “use a bit more white” in recipes that need accurate dry values.
# So from a “real painter” perspective, your numbers encode this story:
# Dark earths and blacks: be careful, they dry darker than they look.
# Most strong hues and midtones: what you see wet is roughly what you get dry (value-wise).
# White: reliable, even a bit of a life-saver—if in doubt, a little more white will help your dry value not collapse into darkness.
DARKEN_PER_PIGMENT = {
    "titanium_white": 1.0,
    "alizarin_crimson": 1.0,
    "cobalt_blue": 1.0,
    "indian_yellow": 1.0,
    "olive_green": 1.0,
    "yellow_ochre": 1.0,
    "burnt_sienna": 1.0,
    "burnt_umber": 1.0,
    # "indigo": 1.0,
    "ivory_black": 1.0,
    "paynes_gray": 1.0,
    "vandyke_brown": 1.0,
}


# ---------------------------
# Main CLI
# ---------------------------
DEFAULT_CONFIG = {
    # ------------------------------------------------------------------
    # 1) EXECUTION / PERFORMANCE
    # ------------------------------------------------------------------
    # parallel:
    #   - If True, recipes for clustered colors are computed in parallel
    #     using ProcessPoolExecutor.
    #   - If False, everything runs single-core (useful for debugging or
    #     platforms where multiprocessing is problematic).
    # workers:
    #   - How many worker processes to spawn when parallel=True.
    #   - None → use os.cpu_count() (or a similar heuristic).
    #   - You can set an explicit integer to limit CPU use.
    "parallel": True,
    "workers": None,
    # ------------------------------------------------------------------
    # 2) INPUT / OUTPUT PATHS
    # ------------------------------------------------------------------
    # input:
    #   - Path to the source image to convert into a paint-by-numbers guide.
    #   - Can be overridden per-call via config.
    # pdf:
    #   - Output filename for the multi-page A4 landscape PDF guide.
    # external_sketch:
    #   - Optional path to a user-supplied sketch/stencil image.
    #   - When set, this grayscale image is resized to the source image
    #     size and used anywhere the generated outline/sketch would be
    #     used: outline page, frame underlay, per-color underlay, and
    #     centerline SVG tracing.
    #   - Use black/dark lines on a white/light background.
    "input": "pics/33.jpg",
    "pdf": "paint_by_numbers_guide.pdf",
    "external_sketch": "pics/33_sketch.png",
    # recipe_cache:
    #   - JSON cache of cluster target colors and recipe results.
    #   - Written after recipe generation so optimizer experiments can
    #     target a color number without rerunning K-Means or rendering a PDF.
    "write_recipe_cache": True,
    "recipe_cache": "outputs/recipe_targets.json",
    # ------------------------------------------------------------------
    # 3) CANVAS GEOMETRY & PRINT LAYOUT (FOR CENTERLINE SVG CANVAS)
    # ------------------------------------------------------------------
    # canvas_dimensions_mm:
    #   - Physical paper size in millimeters for the “canvas” SVG
    #     (centerline layout), as (width_mm, height_mm).
    # canvas_long_margin_mm:
    #   - Margin applied on BOTH ends of the LONGEST side of the canvas
    #     when fitting the grid + centerlines.
    #   - The short side is fitted edge-to-edge; only the long side gets
    #     this margin.
    # canvas_rotation_deg:
    #   - Rotation applied when placing the pixel artwork onto the canvas
    #     for the centerline SVG.
    #   - Supported values: 0 or 90 degrees (others are clamped to 0).
    "canvas_dimensions_mm": (240, 300),
    "canvas_long_margin_mm": 5.0,
    "canvas_rotation_deg": 0,
    # ------------------------------------------------------------------
    # 4) GLOBAL GRID SETTINGS (USED ON PDF PAGES & SVG GRID)
    # ------------------------------------------------------------------
    # grid_step:
    #   - Grid spacing in pixels for PDF pages and SVG overlays.
    #   - "auto" or <=0/None → automatically choose a step such that
    #     at least grid_min_cols boxes fit across the image width.
    # grid_min_cols:
    #   - Minimum number of columns that the auto grid will try to
    #     produce horizontally. Larger values → finer grid.
    "grid_step": "auto",
    "grid_min_cols": 7,
    # ------------------------------------------------------------------
    # 5) CLUSTERING & COLOR QUANTIZATION
    # ------------------------------------------------------------------
    # colors:
    #   - Number of color clusters to extract from the image.
    #   - Controls how many numbered “paint regions” you get.
    # resize:
    #   - Optional (width, height) tuple to resize the image BEFORE
    #     clustering (for speed and noise reduction).
    #   - None → use the full-resolution image for clustering.
    # Clustering is fixed to K-Means in perceptual CIELAB space.
    # Potts/MRF smoothing below handles speckle reduction after clustering.
    "colors": 20,
    "resize": None,  # e.g. (W, H)
    # mrf_smoothing:
    #   - If True, apply Potts/MRF smoothing after clustering and label
    #     upsampling, before tiny-region cleanup and PDF rendering.
    #   - This reduces isolated speckling while keeping label boundaries
    #     tied to the original clustered colors.
    # mrf_beta:
    #   - Strength of the neighbor-agreement penalty.
    #   - Higher values smooth more aggressively; lower values preserve
    #     more fine texture.
    # mrf_iterations:
    #   - Number of smoothing passes. Usually 3-6 is enough.
    "mrf_smoothing": True,
    "mrf_beta": 7.0,
    "mrf_iterations": 4,
    # ------------------------------------------------------------------
    # 6) PALETTE & MIXING / RECIPE SEARCH
    # ------------------------------------------------------------------
    # palette:
    #   - Ordered list of pigment names (keys into BASE_PALETTE,
    #     and optionally DARKEN_PER_PIGMENT).
    #   - The order determines the index mapping for recipe vectors.
    # components:
    #   - Maximum number of distinct pigments allowed in any single recipe.
    #   - Higher values allow more complex mixes but increase search cost.
    # max_parts:
    #   - Maximum sum of integer “parts” in a recipe.
    #   - For example, with max_parts=10, recipes like 3+3+4 are allowed,
    #     but 5+5+3 would be rejected (sum=13).
    # Mixing always uses the Mixbox learned latent-space model with optional
    # per-pigment darkening for dried-paint behavior.
    "palette": list(BASE_PALETTE.keys()),
    "components": 5,
    "max_parts": 10,
    # genetic_retry_enabled:
    #   - If True, run a genetic integer optimizer when the normal exhaustive
    #     integer search returns a recipe above genetic_retry_delta_e.
    #   - This keeps the old deterministic search as the first pass, then
    #     uses a larger integer recipe space only for difficult colors.
    # genetic_retry_delta_e:
    #   - Trigger threshold. Recipes with Delta E above this value get the
    #     genetic fallback search.
    # genetic_retry_max_parts:
    #   - Maximum total parts allowed in a genetic fallback recipe.
    # genetic_retry_components:
    #   - Maximum number of pigments allowed in a genetic fallback recipe.
    # genetic_retry_population / genetic_retry_generations:
    #   - Search budget for the genetic optimizer. Higher values can improve
    #     hard colors but increase runtime.
    "genetic_retry_enabled": True,
    "genetic_retry_delta_e": 1.0,
    "genetic_retry_max_parts": 20,
    "genetic_retry_components": 10,
    "genetic_retry_population": 180,
    "genetic_retry_generations": 160,
    # delta_e_method:
    #   - "colour_ciede2000" uses the colour-science implementation of
    #     CIEDE2000 for recipe scoring (default).
    #   - "skimage_ciede2000" uses scikit-image's CIEDE2000.
    #   - "cie76" uses plain Euclidean Lab distance for old comparisons.
    "delta_e_method": "colour_ciede2000",  # {"colour_ciede2000", "skimage_ciede2000", "cie76"}
    # ------------------------------------------------------------------
    # 7) FRAME SEQUENCING / HIGH-LEVEL PAINT ORDER
    # ------------------------------------------------------------------
    # frame_mode:
    #   - Selects which multi-step “frame sequence” to render in the PDF:
    #       "classic"  → highlight/dark/mid/neutrals/etc in separate frames.
    #       "value5"   → five value bands (deep/core/mid/half/high) only.
    #       "both"     → emit both classic and value-5 sequences.
    #       "combined" → hybrid value-based + classic sequence designed
    #                    for a practical paint workflow.
    # per_color_frames:
    #   - If True, create additional “per-color” pages where each page
    #     shows only one color’s regions (optionally cumulative).
    #   - If False, skip these per-color pages.
    # per_color_order_mode:
    #   - Ordering of colors when generating per-color pages:
    #       "stepwise":
    #           Follow the order implied by the chosen frame_mode
    #           (older behavior; respects the frame painting sequence).
    #       "adjacent":
    #           Use adjacency rings from the border inward; within each
    #           ring, sort dark→light and large areas first (more spatially
    #           coherent progression).
    #   - None (if used) would be interpreted as:
    #         - "adjacent" if frame_mode == "adjacent"
    #         - "stepwise" otherwise (kept for backward-compat).
    "frame_mode": "combined",  # {"classic", "value5", "both", "combined"}
    "per_color_frames": True,
    "per_color_order_mode": "stepwise",
    # ------------------------------------------------------------------
    # 8) COLOR-KEY RENDERING & TEXT LAYOUT
    # ------------------------------------------------------------------
    # wrap:
    #   - Base text-wrap width for recipe descriptions (in characters).
    #   - Individual rows are scaled relative to available width, but this
    #     is the starting point for line wrapping.
    # hide_components:
    #   - If True, hide the component pigment swatches on color-key pages
    #     (only show the mixed color and text).
    #   - If False, show small swatches for each component pigment used
    #     in the recipe.
    "wrap": 55,
    "hide_components": False,
    # ------------------------------------------------------------------
    # 9) GRID / OUTLINE OVERLAY & SKETCH BLENDING
    # ------------------------------------------------------------------
    # edge_percentile:
    #   - Percentile used to determine high Canny thresholds from gradient
    #     magnitudes in the OLD edge-based sketch pipeline.
    #   - Higher values → fewer, stronger edges.
    # outline_mode:
    #   - How to compute the outline passed into the PDF pages:
    #       "image" → legacy method based on image edges / pencil sketch.
    #   - Label-boundary/closed-region outline generation has been removed.
    # sketch_style:
    #   - High-level alias that overrides outline_mode when set:
    #       "old" → outline_mode="image"
    #   - None → use outline_mode directly.
    # sketch_alpha:
    #   - Blending factor (0..1) when multiply-blending the outline
    #     into the colored frames:
    #       0.0 → outline disabled in frames.
    #       1.0 → strong ink-like multiplication.
    "edge_percentile": 90.0,
    "outline_mode": "image",
    "sketch_style": None,  # {"old"}; overrides outline_mode
    "sketch_alpha": 0.25,
    # ------------------------------------------------------------------
    # 10) PER-COLOR PAGE ACCUMULATION / PREVIOUS-AREA HIGHLIGHTING
    # ------------------------------------------------------------------
    # per_color_cumulative:
    #   - If True, each per-color page shows not only the current color
    #     but also the regions painted in previous per-color pages (with
    #     a special highlight treatment).
    #   - If False, each per-color page shows only that color’s regions.
    # prev_alpha:
    #   - Opacity used when blending previous regions into a per-color
    #     page background:
    #       0.0 → previous regions invisible.
    #       1.0 → previous regions at full highlight strength.
    # prev_highlight_mode:
    #   - How to render previous regions on per-color pages:
    #       "none"        → darken/whiten only (no neon coloration).
    #       "neon_orange" → blend white with a neon orange overlay.
    #       "neon_green"  → blend white with a neon green overlay.
    #       "custom"      → blend white with prev_highlight_rgb color.
    # prev_highlight_rgb:
    #   - Custom highlight RGB used only when prev_highlight_mode=="custom".
    #   - Ignored otherwise.
    "per_color_cumulative": True,
    "prev_alpha": 0.10,
    "prev_highlight_mode": "neon_green",  # {"none", "neon_orange", "neon_green", "custom"}
    "prev_highlight_rgb": (255, 90, 0),
    # ------------------------------------------------------------------
    # 11) REGION CLEANUP (MINIMUM REGION SIZE)
    # ------------------------------------------------------------------
    # min_region_px:
    #   - Minimum area in pixels for any connected label component.
    #   - Components smaller than this are merged into a neighboring label
    #     based on local majority voting.
    # min_region_pct:
    #   - Minimum area as a percentage of the total image pixels.
    #   - The effective threshold is max(min_region_px,
    #     min_region_pct/100 * total_pixels).
    #   - Set both to 0 to disable small-region cleanup.
    "min_region_px": 0,
    "min_region_pct": 0.0,
    # ------------------------------------------------------------------
    # 12) CLEAN STENCIL PIPELINE (OUTLINE POST-PROCESSING)
    # ------------------------------------------------------------------
    # apply_clean_stencil:
    #   - If True, apply an adaptive threshold + light erosion pipeline
    #     to produce a crisp, printable stencil from the outline_gray
    #     image, followed by brightness/sharpness adjustments.
    #   - If False, use the raw outline_gray.
    # stencil_brightness:
    #   - Slider value in 0..1 for brightness adjustment of the stencil
    #     (mapped internally to a 0.5..2.0 factor).
    # stencil_sharpness:
    #   - Slider value in 0..1 for sharpness adjustment of the stencil
    #     (mapped internally to a 0.5..3.0 factor).
    # stencil_block_size:
    #   - Odd kernel size for cv2.adaptiveThreshold (local window).
    #   - Must be >=3; larger values capture slower illumination changes.
    # stencil_C:
    #   - Bias term (C) for cv2.adaptiveThreshold. Higher values usually
    #     produce slightly lighter stencils (fewer black pixels).
    "apply_clean_stencil": False,
    "stencil_brightness": 1.0,  # sliders 0..1 (mapped internally)
    "stencil_sharpness": 1.0,
    "stencil_block_size": 11,
    "stencil_C": 2,
    # ------------------------------------------------------------------
    # 13) BUILD-ON POLICY (NEUTRAL DEPENDENCY GRAPH)
    # ------------------------------------------------------------------
    # The following knobs control how the “build-on” dependency graph is
    # constructed for colors (optional extra page).
    # build_max_deltaE:
    #   - Maximum allowed ΔE*ab between parent and child color to be
    #     considered a valid “add parts” step.
    # build_max_added_parts:
    #   - Maximum total number of extra parts allowed between parent and
    #     child recipes.
    # build_max_added_pigments:
    #   - Maximum number of new Δ>0 pigments introduced at that step.
    # build_max_new_pigments:
    #   - Maximum number of pigments that were completely absent in the
    #     parent but appear in the child.
    # build_min_added_fraction:
    #   - Minimum added-parts/parent-total ratio to avoid trivial changes
    #     (e.g. +1 part on a 50-part parent).
    # build_max_chain_depth:
    #   - Upper limit on allowed parent→child chain length (colors deeper
    #     than this have their parent link removed).
    # build_parent_choice:
    #   - Tie-break preference when multiple parents are available:
    #       "min_deltaE"       → prioritize smallest ΔE.
    #       "min_new_pigments" → prioritize minimal introduction of
    #                            completely new pigments.
    #       "min_added_parts"  → prioritize smallest total added parts.
    # build_graph_page:
    #   - If True, emit an extra A4 page showing the dependency graph and
    #     optional imprimatura swatch/recipe.
    "build_max_deltaE": 8.0,
    "build_max_added_parts": 6,
    "build_max_added_pigments": 2,
    "build_max_new_pigments": 1,
    "build_min_added_fraction": 0.05,
    "build_max_chain_depth": 4,
    "build_parent_choice": "min_added_parts",
    "build_graph_page": False,
    # ------------------------------------------------------------------
    # 14) SUPIR PRE-UPSCALE
    # ------------------------------------------------------------------
    # enable_upscale:
    #   - If True, run SUPIR as a pre-processing step when the image is
    #     too small.
    #   - For undersized images, SUPIR failures stop processing instead
    #     of falling back to the original image.
    #   - If False, do not attempt upscaling.
    # upscale_ok_min_long:
    #   - Minimum acceptable longest side (in pixels) for the input image.
    #   - If longest_side >= this value, no upscaling is performed.
    #   - If longest_side < this value, SUPIR chooses the smallest
    #     supir_upscale_choices factor that reaches the threshold.
    "enable_upscale": True,
    "upscale_ok_min_long": 3000,
    "supir_repo_dir": "SUPIR",
    "supir_python": "venv/Scripts/python.exe",
    "supir_model_dir": "supir_models",
    "supir_engine": "SUPIR",
    "supir_sampler": "Ultimate Perception",
    "supir_prioritizing": "Quality",
    "supir_texture_richness": 1.0,
    "supir_creativity": 0.0,
    "supir_image_description": "",
    "supir_sign": "Q",  # {"Q", "F"}; Q = general quality, F = lighter degradation fidelity
    "supir_upscale": "auto",
    "supir_upscale_choices": (1, 2, 3, 4),
    "supir_min_size": 1024,
    "supir_edm_steps": 50,
    "supir_s_noise": 1.02,
    "supir_s_cfg": 6.0,
    "supir_spt_linear_cfg": 3.0,
    "supir_s_stage2": 0.93,
    "supir_color_fix_type": "Wavelet",  # {"None", "AdaIn", "Wavelet"}
    "supir_a_prompt": "",
    "supir_n_prompt": "",
    "supir_ae_dtype": "bf16",
    "supir_diff_dtype": "fp16",
    "supir_use_tile_vae": True,
    "supir_loading_half_params": False,
    "supir_extra_args": [],
    # ------------------------------------------------------------------
    # 15) PRE-BRIGHTENING (BEFORE CLUSTERING / COLORING)
    # ------------------------------------------------------------------
    # pre_brighten_pct:
    #   - Percentage increase in brightness applied AFTER any upscaling
    #     but BEFORE clustering/color analysis.
    #   - 0   → no change.
    #   - 1–100 → mapped to a factor ~= 1.0–2.0 depending on the
    #             internal slider mapping.
    "pre_brighten_pct": 0,
    # ------------------------------------------------------------------
    # 16) IMPRIMATURA (TONED GROUND) SELECTION
    # ------------------------------------------------------------------
    # imprimatura_mode:
    #   - Strategy for auto-picking a suggested imprimatura (toned ground)
    #     color from the image:
    #       "match_light"        → use hue from highlight pixels
    #                              (top ~20% in L*).
    #       "complement_dominant"→ use the complement of the dominant
    #                              mid-tone hue in the scene.
    #       "neutral_warm"       → fixed warm-neutral brown-paper style
    #                              target hue (fallback when scene is
    #                              ambiguous).
    "imprimatura_mode": "match_light",
    # ------------------------------------------------------------------
    # 17) FOREGROUND / BACKGROUND SPLIT (RMBG-2.0)
    # ------------------------------------------------------------------
    # separate_fg_bg:
    #   - If True, run background removal (RMBG-2.0) on the input image
    #     to create a foreground and background matte.
    #   - Per-color pages are then split into two tracks:
    #       1) Background colors (BG).
    #       2) Foreground colors (FG), seeded with already-painted BG.
    #   - If False, the per-color logic ignores any FG/BG distinction.
    # rmbg_model_dir:
    #   - Directory containing BRIA RMBG-2.0 weights (Hugging Face layout).
    # rmbg_device:
    #   - Device for RMBG inference:
    #       "cuda" → use GPU if available.
    #       "cpu"  → force CPU.
    #       None   → auto-detect.
    # rmbg_alpha_threshold:
    #   - Threshold on the [0..1] alpha matte to decide FG vs BG:
    #       matte >= threshold → foreground.
    #       matte <  threshold → background.
    # rmbg_target_size:
    #   - Size (width, height) for RMBG preprocessing; many models expect a
    #     ~1024×1024 square or similar.
    "separate_fg_bg": False,
    "rmbg_model_dir": "rmbg",
    "rmbg_device": "cuda",
    "rmbg_alpha_threshold": 0.5,
    "rmbg_target_size": (1024, 1024),
    # ------------------------------------------------------------------
    # 18) CENTERLINE TRACE (VECTOR STENCIL FOR PLOTTERS)
    # ------------------------------------------------------------------
    # export_centerline_svg:
    #   - If True, generate a single-stroke centerline SVG from the final
    #     cleaned stencil outline (optionally with a grid).
    # centerline_output:
    #   - Path to the primary centerline SVG file.
    # centerline_blur:
    #   - Gaussian blur radius (kernel size in pixels) applied before
    #     thresholding for the centerline extraction.
    #   - 0 or 1 → minimal blur (more detail, more noise).
    # centerline_threshold:
    #   - Manual threshold in [0..255] for binarizing the stencil.
    #   - None → use Otsu’s method (if centerline_otsu=True).
    # centerline_otsu:
    #   - If True, use Otsu automatic threshold; if False and
    #     centerline_threshold is not None, use the manual threshold.
    # centerline_dilate:
    #   - Number of dilation iterations after thresholding to connect
    #     small gaps before skeletonization.
    # centerline_simplify:
    #   - Epsilon for cv2.approxPolyDP polyline simplification on the
    #     extracted centerlines:
    #       0  → no simplification (raw contours).
    #      >0  → smoother, fewer points; too high may simplify away detail.
    "export_centerline_svg": True,
    "centerline_output": "centerline_output.svg",
    "centerline_blur": 1,
    "centerline_threshold": None,
    "centerline_otsu": True,
    "centerline_dilate": 0,
    "centerline_simplify": 0,
}

