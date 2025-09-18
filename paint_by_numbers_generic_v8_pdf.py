
import argparse
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
import textwrap as _tw

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

# -----------------------------
# Color-space utilities
# -----------------------------
def srgb_to_linear_arr(rgb_arr):
    rgb_arr = np.clip(rgb_arr / 255.0, 0, 1)
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055)/1.055) ** 2.4)

def linear_to_srgb_arr(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055)

# XYZ/Lab conversions (D65, sRGB)
def srgb_to_xyz(rgb):
    lin = srgb_to_linear_arr(rgb/255.0)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return M @ lin

def xyz_to_srgb(xyz):
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])
    lin = M @ xyz
    srgb = np.clip(linear_to_srgb_arr(lin), 0, 1)
    return srgb * 255.0

def xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0]/Xn, xyz[1]/Yn, xyz[2]/Zn
    def f(t):
        return np.where(t > (6/29)**3, np.cbrt(t), (1/3)*(29/6)**2 * t + 4/29)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.array([L, a, b])

def lab_to_xyz(lab):
    L, a, b = lab
    Yn = 1.0; Xn = 0.95047; Zn = 1.08883
    fy = (L + 16)/116
    fx = fy + a/500
    fz = fy - b/200
    def finv(t):
        return np.where(t > 6/29, t**3, (3*(6/29)**2)*(t - 4/29))
    x = Xn * finv(fx)
    y = Yn * finv(fy)
    z = Zn * finv(fz)
    return np.array([x, y, z])

def rgb_to_lab(rgb):
    return xyz_to_lab(srgb_to_xyz(rgb))

def lab_to_rgb(lab):
    return xyz_to_srgb(lab_to_xyz(lab))

def relative_luminance(rgb):
    lin = srgb_to_linear_arr(np.array(rgb, dtype=float))
    return 0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2]

def Lstar_from_rgb(rgb):
    return float(np.clip(rgb_to_lab(np.array(rgb, float))[0], 0, 100))

# -----------------------------
# Mixing models
# -----------------------------
def mix_linear(parts, base_rgbs):
    w = parts / np.sum(parts)
    lin = np.sum(srgb_to_linear_arr(base_rgbs.T) * w, axis=1)
    return np.clip(255*linear_to_srgb_arr(lin), 0, 255)

def mix_lab(parts, base_rgbs):
    w = parts / np.sum(parts)
    labs = np.array([rgb_to_lab(c) for c in base_rgbs])
    lab = np.sum(labs.T * w, axis=1)
    return np.clip(lab_to_rgb(lab), 0, 255)

def mix_subtractive(parts, base_rgbs):
    w = parts / np.sum(parts)
    c = (base_rgbs/255.0)
    res = 1.0 - np.prod((1.0 - c) ** w[:, None], axis=0)
    return np.clip(res*255.0, 0, 255)

def mix_km_generic(parts, base_rgbs):
    w = parts / np.sum(parts)
    R = np.clip(base_rgbs/255.0, 1e-4, 1.0)
    A = -np.log(R)
    A_mix = np.sum(A.T * w, axis=1)
    R_mix = np.exp(-A_mix)
    return np.clip(R_mix*255.0, 0, 255)

def mix_color(parts, base_rgbs, model):
    if model == "linear":
        return mix_linear(parts, base_rgbs)
    elif model == "lab":
        return mix_lab(parts, base_rgbs)
    elif model == "subtractive":
        return mix_subtractive(parts, base_rgbs)
    elif model == "km":
        return mix_km_generic(parts, base_rgbs)
    else:
        return mix_linear(parts, base_rgbs)

# -----------------------------
# Integer optimizer (minimizes ΔE in Lab)
# -----------------------------
def enumerate_partitions(total, k):
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in enumerate_partitions(total - i, k - 1):
            yield (i,) + rest

def deltaE_lab(rgb1, rgb2):
    return float(np.linalg.norm(rgb_to_lab(rgb1) - rgb_to_lab(rgb2)))

def integer_mix_best(target_rgb, base_names, max_parts=5, max_components=3, model="km"):
    base_rgbs_full = np.array([BASE_PALETTE[n] for n in base_names], dtype=float)
    target = np.array(target_rgb, dtype=float)

    best_err = float('inf')
    best_entries = []
    best_rgb = target

    N = len(base_names)
    max_components = min(max_components, N, max_parts if max_parts>0 else 1)

    for m in range(1, max_components + 1):
        for combo in itertools.combinations(range(N), m):
            for parts in enumerate_partitions(max_parts, m):
                if sum(parts) != max_parts or all(p == 0 for p in parts):
                    continue
                parts_arr = np.array(parts, dtype=float)
                base_rgbs = base_rgbs_full[list(combo)]
                mix_rgb = mix_color(parts_arr, base_rgbs, model)
                err = deltaE_lab(mix_rgb, target)
                if err < best_err:
                    best_err = err
                    best_rgb = mix_rgb
                    best_entries = [(base_names[i], int(p)) for i, p in zip(combo, parts) if p > 0]

    if len(best_entries) == 1:
        n, _ = best_entries[0]
        best_entries = [(n, 1)]
    return best_entries, best_rgb, best_err

def recipe_text(entries):
    return " + ".join([f"{p} part{'s' if p != 1 else ''} {n}" for n, p in entries]) if entries else "—"

# -----------------------------
# Grouping, tweaks, sketch
# -----------------------------
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

def original_edge_sketch_with_grid(img, grid_step=80, threshold_percentile=75.0):
    arr = np.array(img.convert("RGB"))
    gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    gx = ndi.sobel(gray, axis=1, mode='reflect')
    gy = ndi.sobel(gray, axis=0, mode='reflect')
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    t = np.percentile(mag, threshold_percentile)
    edges = (mag >= t).astype(np.uint8) * 255
    sketch = 255 - edges
    h, w = sketch.shape
    grid_color = 200
    for x in range(0, w, grid_step):
        sketch[:, x:x+1] = grid_color
    for y in range(0, h, grid_step):
        sketch[y:y+1, :] = grid_color
    return Image.fromarray(sketch.astype(np.uint8))

# -----------------------------
# Drawing (single swatch, optional components)
# -----------------------------
def draw_color_key(ax, target_palette, recipes, entries_per_color, base_palette, used_indices=None,
                   title="Color Key • Ratios + Component Paints", tweaks=None, wrap_width=55,
                   show_components=True, deltaEs=None):
    if used_indices is None:
        used_indices = list(range(len(target_palette)))
    if tweaks is None:
        tweaks = {i: "" for i in range(len(target_palette))}

    base_order = list(base_palette.keys())

    for row_idx, ci in enumerate(used_indices):
        target_color = target_palette[ci]
        recipe = recipes[ci]
        entries = entries_per_color[ci]

        # Single square swatch of target color
        ax.add_patch(Rectangle((0, row_idx), 1, 1, color=(target_color/255), ec="k", lw=0.2))
        # Row number overlay
        ax.text(0.5, row_idx+0.5, f"{ci+1}", ha="center", va="center", fontsize=8, color="black",
                bbox=dict(facecolor=(1,1,1,0.45), edgecolor='none', boxstyle='round,pad=0.1'))

        Lstar = Lstar_from_rgb(target_color)
        tweak_str = f"  • L*={Lstar:.1f}"
        if deltaEs is not None:
            tweak_str += f"  • ΔE≈{deltaEs[ci]:.1f}"
        if tweaks.get(ci, ""):
            tweak_str += f"  • {tweaks[ci]}"

        text_str = f"{ci+1}: {recipe}{tweak_str}"
        wrapped = _tw.fill(text_str, width=wrap_width)
        ax.text(1.3, row_idx+0.5, wrapped, va="center", fontsize=8, wrap=True)

        if show_components:
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
    ax.set_title(title + "  (single swatch = target color)")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="A4 PDF with robust PBN recoloring, index-based masks, and advanced mixing.")
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("--pdf", default="paint_by_numbers_guide.pdf", help="Output PDF path")
    parser.add_argument("--colors", type=int, default=20, help="Number of clusters for KMeans")
    parser.add_argument("--resize", type=int, nargs=2, default=[120, 120], metavar=("W", "H"),
                        help="Working size for clustering (speed/quality tradeoff)")
    parser.add_argument("--palette", nargs="*", default=list(BASE_PALETTE.keys()))
    parser.add_argument("--components", type=int, default=3, help="Max component paints per recipe (integer mode)")
    parser.add_argument("--max-parts", type=int, default=10, help="Total integer parts per recipe (cap)")
    parser.add_argument("--mix-model", choices=["linear","lab","subtractive","km"], default="km",
                        help="Mixing model for integer recipe evaluation (default: km)")
    parser.add_argument("--wrap", type=int, default=55, help="Wrap width for legend text")
    parser.add_argument("--grid-step", type=int, default=80, help="Grid spacing in pixels for edge sketch page")
    parser.add_argument("--edge-percentile", type=float, default=85.0, help="Percentile threshold for edge detection")
    parser.add_argument("--hide-components", action="store_true", help="Hide component swatches on the right")
    args = parser.parse_args()

    # Load original
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # Edge sketch page
    sketch_img = original_edge_sketch_with_grid(img, grid_step=args.grid_step, threshold_percentile=args.edge_percentile)

    # --- KMeans (on downsample) ---
    img_small = img.resize(tuple(args.resize), resample=Image.BILINEAR)
    data_small = np.array(img_small)
    Hs, Ws, _ = data_small.shape
    pixels_small = data_small.reshape((-1, 3))

    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=5).fit(pixels_small)
    labels_small = kmeans.labels_.reshape(Hs, Ws).astype(np.uint8)  # label map at small res
    centroids = kmeans.cluster_centers_.astype(float)               # target cluster colors (float)
    target_palette = centroids.astype(np.uint8)                     # for display

    # --- Integer-optimized recipes per cluster ---
    names = args.palette
    all_entries, all_recipes, approx_rgbs, deltaEs = [], [], [], []
    for col in centroids:
        entries, approx_rgb, err = integer_mix_best(col, names,
                                                    max_parts=args.max_parts,
                                                    max_components=args.components,
                                                    model=args.mix_model)
        all_entries.append(entries)
        all_recipes.append(recipe_text(entries))
        approx_rgbs.append(np.array(approx_rgb, dtype=float))
        deltaEs.append(err)
    approx_rgbs = np.array(approx_rgbs, dtype=float)  # (K, 3)
    approx_uint8 = np.clip(np.rint(approx_rgbs), 0, 255).astype(np.uint8)

    # --- Recolor segmentation by integer-mix colors (depends on parts/components/model) ---
    seg_mixed_small = approx_uint8[labels_small]  # (Hs, Ws, 3)
    # Upscale label map and recolored PBN to original size
    labels_orig = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_orig = np.array(labels_orig, dtype=np.uint8)
    pbn_image = Image.fromarray(seg_mixed_small).resize((orig_w, orig_h), resample=Image.NEAREST)
    pbn_image = np.array(pbn_image, dtype=np.uint8)

    # Grouping for frames (based on target palette tonality/saturation)
    groups = auto_group_clusters(target_palette)
    progress_order = [
        ("Frame 1 – Shadows / Dark Blocks", groups["darks"]),
        ("Frame 2 – Mid-tone Masses",       groups["mids"]),
        ("Frame 3 – Neutrals / Background", groups["neutrals"]),
        ("Frame 4 – Highlights",            groups["highs"]),
        ("Frame 5 – Completed",             list(range(args.colors))),
    ]

    # Build per-frame images using **label indices** (robust to color rounding)
    progress_frames = []
    for title, idxs in progress_order:
        if len(idxs) == 0 and "Completed" not in title:
            continue
        mask = np.isin(labels_orig, np.array(idxs, dtype=np.uint8))
        frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
        progress_frames.append((title, idxs, frame_img))

    # Value tweaks grouping by identical recipes
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

    tweaks = build_value_tweaks(target_palette, all_recipes)

    # ----- Build PDF (A4 Landscape) -----
    A4_LANDSCAPE = (11.69, 8.27)
    with PdfPages(args.pdf) as pdf:
        # Page 1: Overview (Original above PBN, full key on right)
        fig = plt.figure(figsize=A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1,1.6], figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1])
        ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
        ax2.imshow(pbn_image); ax2.set_title(f"Paint by Numbers ({args.colors} colors) • model={args.mix_model} • max parts={args.max_parts}"); ax2.axis("off")
        draw_color_key(ax3, target_palette, all_recipes, all_entries, BASE_PALETTE,
                       used_indices=list(range(args.colors)),
                       title=f"Color Key • All Clusters",
                       tweaks=tweaks, wrap_width=args.wrap,
                       show_components=not args.hide_components,
                       deltaEs=deltaEs)
        plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

        # Page 2: Edge sketch
        fig = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig.add_subplot(111)
        ax.imshow(sketch_img, cmap='gray')
        ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px, percentile={args.edge_percentile:.0f})")
        ax.axis("off")
        plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

        # Frame pages
        for title, idxs, frame in progress_frames:
            fig = plt.figure(figsize=A4_LANDSCAPE)
            gs = GridSpec(1, 2, width_ratios=[1, 1.6], figure=fig)
            axL = fig.add_subplot(gs[0,0]); axR = fig.add_subplot(gs[0,1])
            axL.imshow(frame); axL.set_title(title); axL.axis("off")
            draw_color_key(axR, target_palette, all_recipes, all_entries, BASE_PALETTE,
                           used_indices=idxs,
                           title=f"Color Key • {title}",
                           tweaks=tweaks, wrap_width=args.wrap,
                           show_components=not args.hide_components,
                           deltaEs=deltaEs)
            plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

    print(f"✅ Saved A4 landscape PDF to {args.pdf} using mix-model={args.mix_model}, max-parts={args.max_parts}")

if __name__ == "__main__":
    main()
