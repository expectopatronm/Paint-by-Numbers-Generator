from __future__ import annotations

from typing import Dict, List, Tuple
import textwrap as _tw

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .color import Lstar_from_rgb

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
