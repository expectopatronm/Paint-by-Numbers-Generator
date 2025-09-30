import argparse
import itertools
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import textwrap as _tw

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

def srgb_to_linear_arr(rgb_arr):
    rgb_arr = np.clip(rgb_arr / 255.0, 0, 1)
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055)/1.055) ** 2.4)

def linear_to_srgb_arr(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055)

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

def group_classic(palette):
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

def group_value5(palette):
    L = np.array([relative_luminance(c) for c in palette])
    q10, q25, q70, q85 = np.quantile(L, [0.10, 0.25, 0.70, 0.85])
    deep = [i for i in range(len(palette)) if L[i] <= q10]
    core = [i for i in range(len(palette)) if (q10 < L[i] <= q25)]
    mids = [i for i in range(len(palette)) if (q25 < L[i] <= q70)]
    half = [i for i in range(len(palette)) if (q70 < L[i] <= q85)]
    highs = [i for i in range(len(palette)) if L[i] > q85]
    return {"deep": deep, "core": core, "mids": mids, "half": half, "highs": highs}

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

def ensure_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

def im2float01(img_u8): return img_u8.astype(np.float32) / 255.0
def float01_to_u8(imgf): return (np.clip(imgf, 0, 1) * 255.0 + 0.5).astype(np.uint8)
def lerp(a, b, t): return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def clahe_gray(gray_u8, clip=2.0, tiles=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return clahe.apply(gray_u8)

def canny_from_gradients(gray_u8, low_high_ratio=0.35, high_pct=90):
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy).ravel()
    high = float(np.percentile(mag, high_pct))
    high = np.clip(high, 10, 255)
    low = max(5.0, high * low_high_ratio)
    return int(low), int(high)

def remove_small_components(bin_u8, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    out = np.zeros_like(bin_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def size_norm(short_side, frac, odd=True, minv=3):
    k = max(minv, int(round(short_side * frac)))
    if odd: k |= 1
    return k

def illumination_flatten(gray_u8, smin, strength01):
    if strength01 <= 0:
        return gray_u8
    sigma = smin * lerp(0.03, 0.08, strength01)
    base = cv2.GaussianBlur(gray_u8, (0,0), sigma)
    g = im2float01(gray_u8); b = im2float01(base)
    flat = np.clip(g / (b + 1e-4), 0, 2.5)
    flat = flat / flat.max() if flat.max() > 0 else flat
    return float01_to_u8(flat)

def bilateral_edge_aware(gray_u8, strength01):
    if strength01 <= 0:
        return gray_u8
    sigma_color = lerp(10, 80, strength01)
    sigma_space = lerp(3, 12, strength01)
    return cv2.bilateralFilter(gray_u8, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def _auto_edge_mask(edge_strength_u8, target_fg=0.04, min_fg=0.01, max_fg=0.08, iters=8):
    es = edge_strength_u8.astype(np.uint8)
    H, W = es.shape[:2]; N = H * W
    nz = es[es > 0]
    if nz.size == 0:
        return np.zeros_like(es, dtype=np.uint8)
    lo, hi = 50.0, 98.0
    best = None
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        T = np.percentile(nz, p)
        _, binm = cv2.threshold(es, int(T), 255, cv2.THRESH_BINARY)
        fg = np.count_nonzero(binm) / float(N)
        best = binm
        if fg < min_fg:
            lo = 45.0; hi = p
        elif fg > max_fg:
            lo = p; hi = 99.0
        else:
            break
    return best

# ==============================
# Pencil sketch core
# ==============================
def pencil_readable_norm(
    bgr,
    sketchiness01=0.99,
    softness01=0.1,
    highlight_clip01=0.99,
    edge_boost01=0.99,
    texture_suppression01=0.1,
    illumination01=0.1,
    despeckle01=0.25,
    stroke01=0.1,
    line_floor01=0.99,
    use_clahe=True,
    gamma_midtones=0.99
):
    gray = ensure_gray(bgr)
    h, w = gray.shape[:2]; smin = min(h, w)

    gray = illumination_flatten(gray, smin, illumination01)
    if use_clahe:
        gray = clahe_gray(gray, clip=lerp(1.3, 2.0, softness01), tiles=8)

    blur_sharp = cv2.GaussianBlur(gray, (0,0), smin*0.003)
    sharp = cv2.addWeighted(gray, 1.4, blur_sharp, -0.4, 0)

    g_s = bilateral_edge_aware(sharp, texture_suppression01)
    gf  = im2float01(g_s)
    gf = np.power(np.clip(gf, 0, 1), gamma_midtones)

    inv = 1.0 - gf
    sigma = smin * lerp(0.006, 0.016, softness01)
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma, sigmaY=sigma)
    denom = np.maximum(1e-4, 1.0 - blur)
    dodge = np.clip(gf / denom, 0, 1)
    dodge = np.minimum(dodge, lerp(0.90, 0.975, highlight_clip01))

    low, high = canny_from_gradients(
        g_s,
        low_high_ratio=lerp(0.32, 0.50, 1.0 - sketchiness01),
        high_pct=int(lerp(90, 97, sketchiness01))
    )
    can = cv2.Canny(g_s, low, high)

    sigma1 = smin * lerp(0.003, 0.010, sketchiness01)
    sigma2 = sigma1 * 1.6
    g1 = cv2.GaussianBlur(gf, (0,0), sigma1)
    g2 = cv2.GaussianBlur(gf, (0,0), sigma2)
    dog = g1 - g2
    tau = lerp(0.9, 1.18, sketchiness01)
    phi = lerp(8.0, 22.0, sketchiness01 + edge_boost01*0.3)
    xdog = 1.0 - (0.5 * (1 + np.tanh(phi * (dog - tau))))
    xdog_u8 = float01_to_u8(xdog)

    edge_mix = cv2.max(can, xdog_u8)
    target_fg = lerp(0.025, 0.065, sketchiness01)
    edge_bin = _auto_edge_mask(edge_mix, target_fg=target_fg,
                               min_fg=0.015, max_fg=0.09)

    if despeckle01 > 0:
        min_area = int(lerp(0, 0.0020, despeckle01) * (h*w))
        edge_bin = remove_small_components(edge_bin, min_area)
    k = size_norm(smin, lerp(0.0015, 0.0040, stroke01), odd=False, minv=2)
    edge_bin = cv2.dilate(edge_bin, np.ones((k, k), np.uint8), 1)

    edge_mask = edge_bin.astype(np.float32) / 255.0
    ink_floor = 1.0 - (line_floor01 * edge_mask)
    tone_edge_mul = 1.0 - 0.40 * edge_mask
    pencil = np.minimum(dodge * tone_edge_mul, ink_floor)

    return float01_to_u8(pencil)


def original_edge_sketch_with_grid(img, grid_step=80, grid_color=200, **pencil_kwargs):
    """
    Replacement: uses pencil_readable_norm instead of Sobel/percentile edges.
    - img: PIL.Image
    - grid_step: spacing in pixels between grid lines
    - grid_color: 0..255 gray value for grid lines
    - **pencil_kwargs: forwarded to pencil_readable_norm (e.g., sketchiness01, stroke01, etc.)
    """
    # PIL -> OpenCV BGR
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # New sketch
    sketch_u8 = pencil_readable_norm(bgr, **pencil_kwargs)  # grayscale uint8

    # Overlay grid
    out = sketch_u8.copy()
    if grid_step and grid_step > 0:
        out[:, ::grid_step] = grid_color   # vertical lines
        out[::grid_step, :] = grid_color   # horizontal lines

    # Back to PIL
    return Image.fromarray(out, mode="L")

def add_grid_to_rgb(arr, grid_step=80, grid_color=200):
    """
    Overlay a grid onto an RGB uint8 image array, non-destructively.
    """
    out = arr.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("add_grid_to_rgb expects an HxWx3 RGB array.")
    h, w, _ = out.shape
    # Vertical lines
    for x in range(0, w, grid_step):
        out[:, x:x+1, :] = grid_color
    # Horizontal lines
    for y in range(0, h, grid_step):
        out[y:y+1, :, :] = grid_color
    return out

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

        ax.add_patch(Rectangle((0, row_idx), 1, 1, color=(target_color/255), ec="k", lw=0.2))
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

def main():
    parser = argparse.ArgumentParser(description="A4 PDF with classic, value5, or combined 9-step frames.")
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("--pdf", default="paint_by_numbers_guide.pdf", help="Output PDF path")
    parser.add_argument("--colors", type=int, default=25)
    parser.add_argument("--resize", type=int, nargs=2, default=[480, 480], metavar=("W", "H"))
    parser.add_argument("--palette", nargs="*", default=list(BASE_PALETTE.keys()))
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--max-parts", type=int, default=10)
    parser.add_argument("--mix-model", choices=["linear","lab","subtractive","km"], default="km")
    parser.add_argument("--frame-mode", choices=["classic","value5","both","combined"], default="combined",
                        help="Frame set: classic (4+complete), value5 (5), both (separate), or combined (interleaved 9-step)")
    parser.add_argument("--wrap", type=int, default=55)
    parser.add_argument("--grid-step", type=int, default=80)
    parser.add_argument("--edge-percentile", type=float, default=85.0)
    parser.add_argument("--hide-components", action="store_true")
    parser.add_argument("--per-color-frames", action="store_true",
                        help="If set, add a separate frame for each color (inserted before the completed page).")

    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    sketch_img = original_edge_sketch_with_grid(img, grid_step=args.grid_step)

    img_small = img.resize(tuple(args.resize), resample=Image.BILINEAR)
    data_small = np.array(img_small)
    Hs, Ws, _ = data_small.shape
    pixels_small = data_small.reshape((-1, 3))

    kmeans = KMeans(n_clusters=args.colors, random_state=42, n_init=5).fit(pixels_small)
    labels_small = kmeans.labels_.reshape(Hs, Ws).astype(np.uint8)
    centroids = kmeans.cluster_centers_.astype(float)
    target_palette = centroids.astype(np.uint8)

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

    approx_uint8 = np.clip(np.rint(np.array(approx_rgbs)), 0, 255).astype(np.uint8)

    seg_mixed_small = approx_uint8[labels_small]
    labels_orig = Image.fromarray(labels_small, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
    labels_orig = np.array(labels_orig, dtype=np.uint8)
    pbn_image = Image.fromarray(seg_mixed_small).resize((orig_w, orig_h), resample=Image.NEAREST)
    pbn_image = np.array(pbn_image, dtype=np.uint8)

    classic = group_classic(target_palette)
    value5 = group_value5(target_palette)

    # Orders
    classic_order = [
        ("Frame 1 – Shadows / Dark Blocks", classic["darks"]),
        ("Frame 2 – Mid-tone Masses",       classic["mids"]),
        ("Frame 3 – Neutrals / Background", classic["neutrals"]),
        ("Frame 4 – Highlights",            classic["highs"]),
        ("Frame 5 – Completed",             list(range(args.colors))),
    ]

    value5_order = [
        ("Value A – Deep Shadows (lowest ~10%)", value5["deep"]),
        ("Value B – Core Shadows (to ~25%)",     value5["core"]),
        ("Value C – Midtones (to ~70%)",         value5["mids"]),
        ("Value D – Half-Lights (to ~85%)",      value5["half"]),
        ("Value E – Highlights (top ~15%)",      value5["highs"]),
    ]

    def frames_from_order(order):
        frames = []
        for title, idxs in order:
            if len(idxs) == 0:
                continue
            mask = np.isin(labels_orig, np.array(idxs, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames.append((title, idxs, frame_img))
        return frames

    if args.frame_mode == "combined":
        # Build interleaved 9-step, subtracting already-painted indices to prevent duplicates
        painted = set()
        def remaining(idx_list):
            return [i for i in idx_list if i not in painted]

        sequence = [
            ("Step 1 – Deep Shadows",          value5["deep"]),
            ("Step 2 – Core Shadows",          value5["core"]),
            ("Step 3 – Shadows / Dark Blocks", classic["darks"]),     # remaining darks
            ("Step 4 – Value Midtones",        value5["mids"]),
            ("Step 5 – Mid-tone Masses",       classic["mids"]),      # remaining mids
            ("Step 6 – Neutrals / Background", classic["neutrals"]),
            ("Step 7 – Half-Lights",           value5["half"]),
            ("Step 8 – Highlights",            value5["highs"]),
            ("Step 9 – Highlight Accents",     classic["highs"]),
        ]

        frames_combined = []
        for title, idxs in sequence:
            rem = remaining(idxs)
            painted.update(rem)
            if not rem:
                continue
            mask = np.isin(labels_orig, np.array(rem, dtype=np.uint8))
            frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
            frames_combined.append((title, rem, frame_img))
        frames_to_emit = frames_combined
    elif args.frame_mode == "classic":
        frames_to_emit = frames_from_order(classic_order)
    elif args.frame_mode == "value5":
        frames_to_emit = frames_from_order(value5_order)
    else:  # both
        frames_to_emit = frames_from_order(classic_order) + frames_from_order(value5_order)

    tweaks = build_value_tweaks(target_palette, all_recipes)

    A4_LANDSCAPE = (11.69, 8.27)
    with PdfPages(args.pdf) as pdf:
        # Page 1: Overview
        fig = plt.figure(figsize=A4_LANDSCAPE)
        gs = GridSpec(2, 2, width_ratios=[1,1.6], figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[:,1])
        ax1.imshow(img)
        ax1.set_title("Original")
        ax1.axis("off")
        # Keep overview PBN without grid (acts as a clean reference)
        ax2.imshow(pbn_image)
        ax2.set_title(f"Paint by Numbers ({args.colors} colors) • model={args.mix_model} • max parts={args.max_parts}")
        ax2.axis("off")
        draw_color_key(ax3, target_palette, all_recipes, all_entries, BASE_PALETTE,
                       used_indices=list(range(args.colors)),
                       title=f"Color Key • All Clusters",
                       tweaks=tweaks, wrap_width=args.wrap,
                       show_components=not args.hide_components,
                       deltaEs=deltaEs)
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 2: Edge sketch (already has grid)
        fig = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig.add_subplot(111)
        ax.imshow(sketch_img, cmap='gray')
        ax.set_title(f"Original Edge Sketch + Grid (step={args.grid_step}px, percentile={args.edge_percentile:.0f})")
        ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Emit frames in chosen mode (now with grid applied)
        for title, idxs, frame in frames_to_emit:
            frame_with_grid = add_grid_to_rgb(frame, grid_step=args.grid_step, grid_color=200)
            fig = plt.figure(figsize=A4_LANDSCAPE)
            gs = GridSpec(1, 2, width_ratios=[1, 1.6], figure=fig)
            axL = fig.add_subplot(gs[0,0])
            axR = fig.add_subplot(gs[0,1])
            axL.imshow(frame_with_grid)
            axL.set_title(title + " + Grid")
            axL.axis("off")
            draw_color_key(axR, target_palette, all_recipes, all_entries, BASE_PALETTE,
                           used_indices=idxs,
                           title=f"Color Key • {title}",
                           tweaks=tweaks, wrap_width=args.wrap,
                           show_components=not args.hide_components,
                           deltaEs=deltaEs)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        # Optional: per-color frames (inserted just before the completed page)
        if args.per_color_frames:
            for i in range(args.colors):
                # Build a mask that reveals only this color's regions
                mask = (labels_orig == i)
                frame_img = np.where(mask[..., None], pbn_image, 255).astype(np.uint8)
                frame_with_grid = add_grid_to_rgb(frame_img, grid_step=args.grid_step, grid_color=200)

                # Page layout: left = per-color frame; right = color key for this color
                fig = plt.figure(figsize=A4_LANDSCAPE)
                gs = GridSpec(1, 2, width_ratios=[1, 1.6], figure=fig)
                axL = fig.add_subplot(gs[0, 0])
                axR = fig.add_subplot(gs[0, 1])

                axL.imshow(frame_with_grid)
                axL.set_title(f"Per-Color • #{i+1}")
                axL.axis("off")

                draw_color_key(
                    axR,
                    target_palette,
                    all_recipes,
                    all_entries,
                    BASE_PALETTE,
                    used_indices=[i],
                    title=f"Color Key • Color #{i+1}",
                    tweaks=tweaks,
                    wrap_width=args.wrap,
                    show_components=not args.hide_components,
                    deltaEs=deltaEs
                )

                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # Completed page (with grid applied)
        completed_with_grid = add_grid_to_rgb(pbn_image, grid_step=args.grid_step, grid_color=200)
        fig = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig.add_subplot(111)
        ax.imshow(completed_with_grid)
        ax.set_title("Completed — All Colors Applied + Grid")
        ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"✅ Saved A4 landscape PDF to {args.pdf} (frame-mode={args.frame_mode})")

if __name__ == "__main__":
    main()
