from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .color import deltaE_lab

def _entries_to_vec(entries: List[Tuple[str,int]], base_order: List[str]) -> np.ndarray:
    v = np.zeros(len(base_order), dtype=int)
    for n, p in entries:
        v[base_order.index(n)] = int(p)
    return v

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(b >= a) and np.any(b > a)

def _extras(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(b - a, 0, None)

@dataclass
class BuildPolicy:
    max_deltaE: float = 8.0
    max_added_parts: int = 6
    max_added_pigments: int = 2
    max_new_pigments: int = 1
    min_added_fraction: float = 0.05
    max_chain_depth: int = 4
    parent_choice: str = "min_added_parts"   # "min_deltaE" | "min_new_pigments" | "min_added_parts"

def plan_build_order_configurable(parts_mat: np.ndarray,
                                  approx_rgb_uint8: np.ndarray,
                                  base_order: List[str],
                                  policy: BuildPolicy):
    """
    Neutral build-on planner. Returns:
      order, step_note, parent_map, extras_label
    """
    C, _ = parts_mat.shape
    totals = parts_mat.sum(axis=1)

    candidates = {i: [] for i in range(C)}
    for p in range(C):
        for c in range(C):
            if p == c:
                continue
            a, b = parts_mat[p], parts_mat[c]
            if not _dominates(a, b):
                continue
            dE = deltaE_lab(approx_rgb_uint8[p], approx_rgb_uint8[c])
            if dE > policy.max_deltaE:
                continue
            e = _extras(a, b)
            total_added = int(e.sum())
            if total_added > policy.max_added_parts:
                continue
            idxs = np.nonzero(e)[0].tolist()
            added_pigs = len(idxs)
            if added_pigs > policy.max_added_pigments:
                continue
            parent_support = set(np.nonzero(a)[0].tolist())
            new_pigs = len([i for i in idxs if i not in parent_support])
            if new_pigs > policy.max_new_pigments:
                continue
            if totals[p] > 0 and (total_added / max(1, totals[p])) < policy.min_added_fraction:
                continue

            if policy.parent_choice == "min_deltaE":
                score = (dE, total_added, added_pigs, new_pigs)
            elif policy.parent_choice == "min_new_pigments":
                score = (new_pigs, added_pigs, total_added, dE)
            else:
                score = (total_added, added_pigs, dE, new_pigs)

            label = " + ".join([f"{int(e[i])}× {base_order[i]}" for i in idxs])
            candidates[c].append((score, p, label))

    parent = {i: None for i in range(C)}
    extras_label = {i: "" for i in range(C)}
    for c in range(C):
        if candidates[c]:
            candidates[c].sort(key=lambda t: t[0])
            _, p, label = candidates[c][0]
            parent[c] = p
            extras_label[c] = label

    # chain depth
    depth = {i: 0 for i in range(C)}
    def _depth(i):
        if parent[i] is None: return 0
        if depth[i] != 0: return depth[i]
        d = 1 + _depth(parent[i]); depth[i] = d; return d
    for i in range(C): _depth(i)
    for i in range(C):
        if depth[i] > policy.max_chain_depth:
            parent[i] = None
            extras_label[i] = ""

    # children buckets
    children = {i: [] for i in range(C)}
    for c, p in parent.items():
        if p is not None: children[p].append(c)
    for p in children:
        children[p].sort(key=lambda c: (parts_mat[c].sum() - parts_mat[p].sum()))

    bases = [i for i in range(C) if parent[i] is None]
    bases.sort(key=lambda i: -int(totals[i]))

    order = []
    step_note = {}
    q = bases[:]
    while q:
        i = q.pop(0)
        order.append(i)
        if parent[i] is None:
            step_note[i] = f"Base mix for color #{i+1}"
        else:
            step_note[i] = f"Color #{i+1} = Color #{parent[i]+1} + {extras_label[i]}"
        q.extend(children[i])

    return order, step_note, parent, extras_label


def _levels_from_parents(parent: dict[int, int|None]) -> dict[int,int]:
    lvl = {}
    def rec(i):
        if i in lvl: return lvl[i]
        p = parent.get(i, None)
        lvl[i] = 0 if p is None else 1 + rec(p)
        return lvl[i]
    for i in parent.keys():
        rec(i)
    return lvl


def draw_build_graph_page(approx_rgb_uint8: np.ndarray,
                          parent: dict[int, int|None],
                          extras_label: dict[int,str],
                          *,
                          title: str = "Build Dependency Graph (Neutral)",
                          imprimatura: dict | None = None):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_subplot(111)
    ax.set_title(title, pad=8, fontsize=12)
    ax.axis("off")

    levels = _levels_from_parents(parent)
    max_level = max(levels.values()) if levels else 0

    by_level = {}
    for i, lv in levels.items():
        by_level.setdefault(lv, []).append(i)
    for lv in by_level:
        by_level[lv].sort()

    pos = {}
    for lv in range(max_level + 1):
        nodes = by_level.get(lv, [])
        n = max(1, len(nodes))
        xs = np.linspace(0.12, 0.88, n)
        y = 0.88 - lv * (0.75 / max(1, max_level))
        for k, i in enumerate(nodes):
            pos[i] = (xs[k], y)

    # edges
    for c, p in parent.items():
        if p is None: continue
        x0, y0 = pos[p]; x1, y1 = pos[c]
        ax.annotate("", xy=(x1, y1-0.02), xytext=(x0, y0+0.02),
                    arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.85))
        xm, ym = (x0 + x1)/2, (y0 + y1)/2
        if extras_label.get(c):
            ax.text(xm, ym, extras_label[c], fontsize=6, ha="center", va="center", color="dimgray")

    # nodes
    for i, (x, y) in pos.items():
        rgb = (approx_rgb_uint8[i] / 255.0).tolist()
        circ = plt.Circle((x, y), 0.03, color=rgb, ec="black", lw=0.6)
        ax.add_patch(circ)
        ax.text(x, y + 0.055, f"{i+1}", ha="center", va="center", fontsize=8, fontweight="bold")

    ax.text(0.015, 0.03, "Arrow: add parts to reach child • Edge label: parts× pigment",
            fontsize=8, ha="left", va="bottom", transform=ax.transAxes)

    # --- Optional imprimatura panel (swatch + recipe) ---
    if imprimatura is not None:
        # Panel position (in axes fraction coords)
        x0, y0, w, h = 0.67, 0.05, 0.30, 0.22
        ax.add_patch(Rectangle((x0, y0), w, h,
                               transform=ax.transAxes, facecolor="white", edgecolor="black", lw=0.6))
        ax.text(x0 + 0.015, y0 + h - 0.06, "Imprimatura (toned ground)", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        # Swatch
        sw = imprimatura.get("rgb", np.array([190,150,110], dtype=np.uint8)) / 255.0
        ax.add_patch(Rectangle((x0 + 0.015, y0 + 0.02), 0.12, 0.12,
                               transform=ax.transAxes, facecolor=tuple(sw), edgecolor="black", lw=0.4))
        # Text
        recipe = imprimatura.get("recipe_text", "—")
        L = imprimatura.get("Lstar", None)
        de = imprimatura.get("deltaE", None)
        lines = [f"Recipe: {recipe}"]
        if L is not None: lines.append(f"L*≈{L:.1f} (mid-tone)")
        if de is not None: lines.append(f"ΔE to target≈{de:.2f}")
        lines.append("Tip: apply as a thin, transparent wash.")
        ax.text(x0 + 0.15, y0 + 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=8, va="bottom")

    return fig

