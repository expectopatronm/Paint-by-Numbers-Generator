
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from scipy.optimize import nnls
import textwrap

# -----------------------------
# Base palette (approximate sRGB for tube colors)
# -----------------------------
BASE_PALETTE = {
    "Titanium White": (245, 245, 245),
    "Lemon Yellow": (250, 239, 80),
    "Vermillion Red": (214, 66, 50),
    "Carmine": (170, 25, 60),
    "Ultramarine": (25, 50, 140),
    "Pthalo Green": (20, 100, 70),
    "Yellow Ochre": (196, 158, 84),
    "Lamp Black": (20, 20, 20),
}

# ---------- Color space helpers ----------
def srgb_to_linear_arr(rgb_arr):
    rgb_arr = rgb_arr / 255.0
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055)/1.055) ** 2.4)

def linear_to_srgb_arr(lin):
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055)

def relative_luminance(rgb):
    lin = srgb_to_linear_arr(np.array(rgb, dtype=float))
    return 0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]

def Lstar_from_rgb(rgb):
    Y = relative_luminance(rgb)
    eps = 216/24389
    kappa = 24389/27
    if Y > eps:
        fY = np.cbrt(Y)
        L = 116*fY - 16
    else:
        L = kappa*Y
    return float(max(0, min(100, L)))

def build_palette_matrix(names):
    cols = []
    for n in names:
        srgb = np.array(BASE_PALETTE[n], dtype=float)
        cols.append(srgb_to_linear_arr(srgb))
    return np.stack(cols, axis=1)

# ---------- Mixing solver ----------
def solve_mix_weights(target_rgb, names):
    A = build_palette_matrix(names)
    b = srgb_to_linear_arr(np.array(target_rgb, dtype=float))
    w, _ = nnls(A, b)
    if w.sum() == 0:
        w = np.ones(len(names))
    w = w / w.sum()
    recon_lin = A @ w
    recon_rgb = np.clip(255 * linear_to_srgb_arr(recon_lin), 0, 255)
    err = float(np.linalg.norm(recon_rgb - np.array(target_rgb, float)))
    return w, err, recon_rgb

def simplify_parts(weights, names, max_components=4, scale=12):
    idx = np.argsort(weights)[::-1]
    idx = idx[weights[idx] > 1e-3][:max_components]
    sel_w = weights[idx]
    if sel_w.sum() == 0:
        sel_w = np.ones_like(sel_w)
    sel_w = sel_w / sel_w.sum()
    parts = np.rint(sel_w * scale).astype(int)
    parts[parts == 0] = 1
    return [(names[i], int(parts[j])) for j, i in enumerate(idx)]

def recipe_text(entries):
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries])

# ---------- Grouping helpers ----------
def rgb_to_hsv(rgb):
    rgb = np.array(rgb, dtype=float) / 255.0
    mx = rgb.max(); mn = rgb.min()
    diff = mx - mn
    if diff == 0:
        h = 0.0
    elif mx == rgb[0]:
        h = (60 * ((rgb[1]-rgb[2]) / diff) + 360) % 360
    elif mx == rgb[1]:
        h = (60 * ((rgb[2]-rgb[0]) / diff) + 120) % 360
    else:
        h = (60 * ((rgb[0]-rgb[1]) / diff) + 240) % 360
    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v

def auto_group_clusters(palette):
    n = len(palette)
    lums = np.array([relative_luminance(c) for c in palette])
    sats = np.array([rgb_to_hsv(c)[1] for c in palette])
    q25 = np.quantile(lums, 0.25)
    q80 = np.quantile(lums, 0.80)
    darks = [i for i in range(n) if lums[i] <= q25]
    highs = [i for i in range(n) if lums[i] >= q80]
    neutrals = [i for i in range(n) if (sats[i] <= 0.20) and (i not in highs)]
    mids = [i for i in range(n) if i not in darks and i not in highs and i not in neutrals]
    return {"darks": darks, "mids": mids, "neutrals": neutrals, "highs": highs}

# ---------- Value tweak suggestions ----------
def build_value_tweaks(palette, recipes_text):
    groups = {}
    for i, r in enumerate(recipes_text):
        groups.setdefault(r, []).append(i)
    tweaks = {i: "" for i in range(len(palette))}
    for r, indices in groups.items():
        if len(indices) <= 1:
            continue
        Ls = np.array([Lstar_from_rgb(palette[i]) for i in indices])
        L_mean = Ls.mean()
        for ci, L in zip(indices, Ls):
            delta = L - L_mean
            if delta > 0.5:
                tweaks[ci] = "Value tweak: + tiny White"
            elif delta < -0.5:
                tweaks[ci] = "Value tweak: + tiny Black"
            else:
                tweaks[ci] = "Value tweak: none (base)"
    return tweaks

# ---------- Drawing helpers (with wrapping) ----------
def draw_color_key(ax, palette, recipes, entries_per_color, base_palette, used_indices=None,
                   title="Color Key • Ratios + Component Paints", tweaks=None, wrap_width=55):
    if used_indices is None:
        used_indices = list(range(len(palette)))
    if tweaks is None:
        tweaks = {i: "" for i in range(len(palette))}

    base_order = list(base_palette.keys())

    for row_idx, ci in enumerate(used_indices):
        color = palette[ci]
        recipe = recipes[ci]
        entries = entries_per_color[ci]

        ax.add_patch(Rectangle((0, row_idx), 1, 1, color=(color/255), ec="k", lw=0.2))
        Lstar = Lstar_from_rgb(color)
        tweak_str = f"  • L*={Lstar:.1f}"
        if tweaks.get(ci, ""):
            tweak_str += f"  • {tweaks[ci]}"

        # Wrap the text to avoid overflow
        text_str = f"{ci+1}: {recipe}{tweak_str}"
        wrapped = textwrap.fill(text_str, width=wrap_width)
        ax.text(1.3, row_idx+0.5, wrapped, va="center", fontsize=8, wrap=True)

        # Component swatches in fixed order
        comp_names = [n for (n, _) in entries]
        base_x = 13.5
        col_pos = 0
        for name in base_order:
            if name in comp_names:
                comp_rgb = np.array(base_palette[name]) / 255.0
                ax.add_patch(Rectangle((base_x + col_pos*0.8, row_idx), 0.7, 1, color=comp_rgb, ec="k", lw=0.2))
                col_pos += 1

    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, len(used_indices))
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="A4 landscape PDF with wrapped legend text and value tweaks.")
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("--pdf", default="paint_by_numbers_guide.pdf", help="Output PDF path")
    parser.add_argument("--colors", type=int, default=15, help="Number of colors for KMeans")
    parser.add_argument("--resize", type=int, nargs=2, default=[120, 120], metavar=("W", "H"),
                        help="Resize (width height) used ONLY for clustering speed")
    parser.add_argument("--palette", nargs="*", default=list(BASE_PALETTE.keys()),
                        help="Subset/order of base paints to use (default: all)")
    parser.add_argument("--components", type=int, default=4, help="Max number of component paints to show per recipe")
    parser.add_argument("--wrap", type=int, default=55, help="Wrap width for legend text")
    args = parser.parse_args()

    # Load full original
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # Downsample ONLY for clustering
    img_small = img.resize(tuple(args.resize), resample=Image.BILINEAR)
    data_small = np.array(img_small)
    pixels_small = data_small.reshape((-1, 3))

    # KMeans on downsampled data
    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=5).fit(pixels_small)
    labels_small = kmeans.labels_
    palette = kmeans.cluster_centers_.astype(int)

    # Small segmented image and upscaled to original
    seg_small = palette[labels_small].reshape(data_small.shape).astype(np.uint8)
    seg = Image.fromarray(seg_small).resize((orig_w, orig_h), resample=Image.NEAREST)
    seg = np.array(seg)

    # Recipes & weights
    names = args.palette
    all_weights, all_entries, all_recipes = [], [], []
    for col in palette:
        w, _, _ = solve_mix_weights(col, names)
        entries = simplify_parts(w, names, max_components=args.components, scale=12)
        rec = recipe_text(entries)
        all_weights.append(w); all_entries.append(entries); all_recipes.append(rec)

    # Value tweaks
    tweaks = build_value_tweaks(palette, all_recipes)

    # Auto progress grouping
    groups = auto_group_clusters(palette)
    progress_order = [
        ("Frame 1 – Shadows / Dark Blocks", groups["darks"]),
        ("Frame 2 – Mid-tone Masses",       groups["mids"]),
        ("Frame 3 – Neutrals / Background",  groups["neutrals"]),
        ("Frame 4 – Highlights",             groups["highs"]),
        ("Frame 5 – Completed",              list(range(args.colors))),
    ]

    # Prepare frames
    def mask_for_clusters(seg, palette, cluster_indices):
        H, W, _ = seg.shape
        pixels = seg.reshape(-1, 3)
        pal_subset = np.array([palette[i] for i in cluster_indices])
        if pal_subset.size == 0:
            return np.zeros((H, W), dtype=bool)
        eq = (pixels[:, None, :] == pal_subset[None, :, :]).all(axis=2)
        mask_flat = eq.any(axis=1)
        return mask_flat.reshape(H, W)

    progress_frames = []
    for title, idxs in progress_order:
        if len(idxs) == 0 and "Completed" not in title:
            continue
        mask = mask_for_clusters(seg, palette, idxs)
        frame_img = np.where(mask[:, :, None], seg, 255).astype(np.uint8)
        progress_frames.append((title, idxs, frame_img))

    # ----- Build PDF (A4 Landscape) -----
    A4_LANDSCAPE = (11.69, 8.27)
    with PdfPages(args.pdf) as pdf:
        # Page 1: Original above PBN (left), overall Key (right spanning)
        fig = plt.figure(figsize=A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1,1.6], figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1])

        ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
        ax2.imshow(seg); ax2.set_title(f"Paint by Numbers ({args.colors} colors)"); ax2.axis("off")
        draw_color_key(ax3, palette, all_recipes, all_entries, BASE_PALETTE,
                       used_indices=list(range(args.colors)),
                       title="Color Key • All Clusters (with L* + Value Tweaks)",
                       tweaks=tweaks, wrap_width=args.wrap)

        plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

        # Per-frame pages with wrapped legend text
        for title, idxs, frame in progress_frames:
            fig = plt.figure(figsize=A4_LANDSCAPE)
            gs = GridSpec(1, 2, width_ratios=[1, 1.6], figure=fig)
            axL = fig.add_subplot(gs[0,0]); axR = fig.add_subplot(gs[0,1])

            axL.imshow(frame); axL.set_title(title); axL.axis("off")
            draw_color_key(axR, palette, all_recipes, all_entries, BASE_PALETTE,
                           used_indices=idxs,
                           title=f"Color Key • {title} (with L* + Value Tweaks)",
                           tweaks=tweaks, wrap_width=args.wrap)

            plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

    print(f"✅ Saved A4 landscape PDF with wrapped legend text to {args.pdf}")

if __name__ == "__main__":
    main()
