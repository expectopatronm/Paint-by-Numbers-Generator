from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np
import svgwrite
from skimage import measure
from skimage.morphology import skeletonize

# ==============================================================
# Centerline tracing (convert stencil to single-line SVG)
# ==============================================================
def centerline_gray_from_outline(gray: np.ndarray, args) -> np.ndarray:
    """Return a white-background raster matching the centerline SVG paths.

    Args:
        gray: Grayscale stencil/outline image to trace.
        args: Generator settings with centerline thresholding options.

    Returns:
        Grayscale uint8 image with black centerlines on white.
    """
    skel = _centerline_skeleton(gray, args)
    centerline_gray = np.full(skel.shape, 255, dtype=np.uint8)
    centerline_gray[skel > 0] = 0
    return centerline_gray


def _centerline_skeleton(gray: np.ndarray, args) -> np.ndarray:
    img = cv2.GaussianBlur(gray, (args.centerline_blur, args.centerline_blur), 0) if args.centerline_blur > 0 else gray

    if args.centerline_otsu:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        t = args.centerline_threshold if args.centerline_threshold is not None else 128
        _, bw = cv2.threshold(img, int(t), 255, cv2.THRESH_BINARY_INV)

    if args.centerline_dilate > 0:
        kernel = np.ones((2, 2), np.uint8)
        bw = cv2.dilate(bw, kernel, iterations=args.centerline_dilate)

    return skeletonize((bw > 0).astype(np.uint8)).astype(np.uint8)


def _gray_int_to_hex(c: int) -> str:
    """200 -> '#C8C8C8' etc."""
    c = int(np.clip(c, 0, 255))
    h = f"{c:02X}"
    return f"#{h}{h}{h}"


def _save_svg_atomic(dwg, output_path: str) -> str:
    """
    Save an svgwrite Drawing through a temporary file, then replace the target.
    This is more reliable on Windows than opening an existing large SVG directly.
    """
    out = os.path.abspath(os.fspath(output_path))
    out_dir = os.path.dirname(out) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".svg", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fileobj:
            dwg.write(fileobj)
        os.replace(tmp, out)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out


def _add_svg_grid(container, drawing, width, height, step, color):
    """Add a square grid with alternating cell diagonals to an SVG.

    Args:
        container: SVG drawing or group that receives the lines.
        drawing: SVG drawing used to construct line elements.
        width: Grid width in drawing units.
        height: Grid height in drawing units.
        step: Square-grid spacing in drawing units.
        color: SVG stroke color.
    """
    if step <= 0:
        return

    for x in range(0, width + 1, step):
        container.add(drawing.line(
            start=(x, 0), end=(x, height), stroke=color,
            stroke_width=0.5, opacity=0.7, class_="grid-square",
        ))
    for y in range(0, height + 1, step):
        container.add(drawing.line(
            start=(0, y), end=(width, y), stroke=color,
            stroke_width=0.5, opacity=0.7, class_="grid-square",
        ))

    for row, y0 in enumerate(range(0, height - step + 1, step)):
        for col, x0 in enumerate(range(0, width - step + 1, step)):
            x1, y1 = x0 + step, y0 + step
            start, end = (
                ((x0, y0), (x1, y1))
                if (row + col) % 2 == 0
                else ((x1, y0), (x0, y1))
            )
            container.add(drawing.line(
                start=start, end=end, stroke=color, stroke_width=0.5,
                opacity=0.45, class_="grid-diagonal",
            ))


def run_centerline_trace(args):
    """
    Generate a single-stroke centerline SVG from the final clean stencil outline,
    and overlay a grid in the SVG.

    Expects on `args`:
      - export_centerline_svg: bool
      - centerline_output: str
      - centerline_blur, centerline_threshold, centerline_otsu,
        centerline_dilate, centerline_simplify
      - outline_gray: np.ndarray (grayscale stencil to trace)
      - grid_step: int           # <-- used for SVG grid
      - (optional) grid_color: int (0..255)  # if not present, use 200
    """
    if not hasattr(args, "outline_gray") or args.outline_gray is None:
        print("No stencil available for centerline tracing — skipped.")
        return

    gray = args.outline_gray
    skel = _centerline_skeleton(gray, args)

    # Extract contours as polylines
    contours = measure.find_contours(skel, 0.5)
    h, w = skel.shape

    centerline_output = os.path.abspath(os.fspath(args.centerline_output))

    # Prepare SVG
    dwg = svgwrite.Drawing(centerline_output, size=(w, h))
    # Optional: white background rect, if you prefer explicit white:
    # dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="#FFFFFF"))

    # ---------------------------
    # Add GRID (draw first so it's behind the strokes)
    # ---------------------------
    grid_step = int(getattr(args, "grid_step", 250))
    grid_gray = int(getattr(args, "grid_color", 200))
    grid_hex = _gray_int_to_hex(grid_gray)
    _add_svg_grid(dwg, dwg, w, h, grid_step, grid_hex)

    # ---------------------------
    # Add CENTERLINES
    # ---------------------------
    for cnt in contours:
        pts = [(float(c[1]), float(c[0])) for c in cnt]
        if args.centerline_simplify > 0:
            epsilon = float(args.centerline_simplify)
            approx = cv2.approxPolyDP(np.array(pts, dtype=np.float32), epsilon, False)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        dwg.add(dwg.polyline(points=pts, fill="none", stroke="black", stroke_width=0.1))

    # Save SVG
    saved_centerline = _save_svg_atomic(dwg, centerline_output)
    print(f"Centerline SVG with grid saved: {saved_centerline} (blur={args.centerline_blur}, simplify={args.centerline_simplify}, grid_step={grid_step})")

    # ----------------------------------------------------
    # B) PAPER CANVAS SVG (portrait, mm) — scale+center the
    #    ORIGINAL pixel grid + centerlines together (no warp)
    # ----------------------------------------------------
    dims = getattr(args, "canvas_dimensions_mm", (240, 300))
    Wmm, Hmm = float(dims[0]), float(dims[1])

    canvas_out = getattr(args, "centerline_canvas_output", None) or (
            os.path.splitext(centerline_output)[0] + "_canvas.svg"
    )
    canvas_out = os.path.abspath(os.fspath(canvas_out))

    from svgwrite import Drawing
    dwg_mm = Drawing(canvas_out, size=(f"{Wmm}mm", f"{Hmm}mm"), viewBox=f"0 0 {Wmm} {Hmm}")

    # --- NEW: rotation (0 or 90 deg) on the canvas
    rot = int(getattr(args, "canvas_rotation_deg", 0))
    if rot not in (0, 90):
        rot = 0  # clamp to safe values

    # --- Margins on the CANVAS (both ends of each axis)
    long_margin = float(getattr(args, "canvas_long_margin_mm", 5.0))
    long_margin = max(0.0, long_margin)
    short_margin = float(getattr(args, "canvas_short_margin_mm", 0.0))
    short_margin = max(0.0, short_margin)
    width_is_long = Wmm >= Hmm

    # Artwork size in pixels BEFORE rotation: (w × h)
    # Effective bounding box AFTER rotation (in pixels)
    W_art_px = w if rot == 0 else h
    H_art_px = h if rot == 0 else w

    # Available drawing area in mm. Keeping a short-side margin prevents the
    # driver from clipping a 300 mm-wide portrait canvas after auto-rotation.
    if width_is_long:
        avail_W = max(0.0, Wmm - 2.0 * long_margin)
        avail_H = max(0.0, Hmm - 2.0 * short_margin)
        offset_x_mm = long_margin
        offset_y_mm = short_margin
    else:
        avail_W = max(0.0, Wmm - 2.0 * short_margin)
        avail_H = max(0.0, Hmm - 2.0 * long_margin)
        offset_x_mm = short_margin
        offset_y_mm = long_margin

    # Uniform scale to fit the (possibly rotated) artwork into available area
    s = min(
        avail_W / float(W_art_px) if W_art_px > 0 else 1.0,
        avail_H / float(H_art_px) if H_art_px > 0 else 1.0
    )

    # Center INSIDE the available area
    used_W = s * W_art_px
    used_H = s * H_art_px
    tx_inner = (avail_W - used_W) / 2.0
    ty_inner = (avail_H - used_H) / 2.0

    # Final translation (mm)
    tx = offset_x_mm + tx_inner
    ty = offset_y_mm + ty_inner

    # Group transform:
    # - For 0°: translate -> scale
    # - For 90°: translate -> rotate(90) -> scale -> pre-translate(0, -h)
    #   (Rightmost applies first; pre-translate keeps the rotated bbox in +X/+Y)
    if rot == 0:
        root = dwg_mm.g(transform=f"translate({tx},{ty}) scale({s})")
    else:  # 90 degrees clockwise
        root = dwg_mm.g(transform=f"translate({tx},{ty}) rotate(90) scale({s}) translate(0, {-h})")

    # ---- redraw the ORIGINAL pixel-space grid so it scales (and rotates) together
    grid_step_px = int(getattr(args, "grid_step", 250))
    grid_gray = int(getattr(args, "grid_color", 200))
    grid_hex = _gray_int_to_hex(grid_gray)
    _add_svg_grid(root, dwg_mm, w, h, grid_step_px, grid_hex)

    # ---- centerlines (pixel coords) go into the same group so they share transforms
    for cnt in contours:
        pts = [(float(c[1]), float(c[0])) for c in cnt]
        if args.centerline_simplify > 0:
            epsilon = float(args.centerline_simplify)
            approx = cv2.approxPolyDP(np.array(pts, dtype=np.float32), epsilon, False)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        root.add(dwg_mm.polyline(points=pts, fill="none", stroke="black", stroke_width=0.1))

    dwg_mm.add(root)
    saved_canvas = _save_svg_atomic(dwg_mm, canvas_out)
    print(f"Centerline canvas SVG saved: {saved_canvas} (paper {Wmm}×{Hmm} mm, s={s:.4f}, rot={rot}°)")

    # Optional: vpype optimization for the canvas file
    try:
        subprocess.run(
            ["vpype", "read", canvas_out, "linemerge", "linesimplify", "-t", "0.3", "reloop", "write", canvas_out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("vpype optimization applied (canvas SVG).")
    except Exception:
        print("vpype not available — saved raw canvas SVG instead.")




def _auto_grid_step(img_width: int, min_cols: int) -> int:
    """
    Choose a pixel step so that floor(img_width / step) >= min_cols.
    Using step = floor(img_width / min_cols) guarantees >= min_cols columns.
    """
    step = max(1, img_width // max(1, int(min_cols)))
    return int(step)
