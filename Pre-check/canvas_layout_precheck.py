#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pbn.config import DEFAULT_CONFIG  # noqa: E402


@dataclass(frozen=True)
class Placement:
    canvas_w_mm: float
    canvas_h_mm: float
    image_w_px: int
    image_h_px: int
    rotation_deg: int
    scale_mm_per_px: float
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    left_margin_mm: float
    right_margin_mm: float
    top_margin_mm: float
    bottom_margin_mm: float
    configured_long_margin_mm: float


def fmt_mm(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text} mm"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def compute_placement(image_size: tuple[int, int], cfg: dict) -> Placement:
    image_w_px, image_h_px = image_size
    canvas_w_mm, canvas_h_mm = (float(v) for v in cfg["canvas_dimensions_mm"])

    rotation_deg = int(cfg.get("canvas_rotation_deg", 0))
    if rotation_deg not in (0, 90):
        rotation_deg = 0

    long_margin_mm = max(0.0, float(cfg.get("canvas_long_margin_mm", 5.0)))
    width_is_long = canvas_w_mm >= canvas_h_mm

    art_w_px = image_w_px if rotation_deg == 0 else image_h_px
    art_h_px = image_h_px if rotation_deg == 0 else image_w_px

    # Keep this in lockstep with pbn.svg_trace.run_centerline_trace:
    # only the canvas's longest side gets the configured margin before fitting.
    if width_is_long:
        available_w_mm = max(0.0, canvas_w_mm - 2.0 * long_margin_mm)
        available_h_mm = canvas_h_mm
        offset_x_mm = long_margin_mm
        offset_y_mm = 0.0
    else:
        available_w_mm = canvas_w_mm
        available_h_mm = max(0.0, canvas_h_mm - 2.0 * long_margin_mm)
        offset_x_mm = 0.0
        offset_y_mm = long_margin_mm

    scale = min(
        available_w_mm / float(art_w_px) if art_w_px > 0 else 1.0,
        available_h_mm / float(art_h_px) if art_h_px > 0 else 1.0,
    )
    used_w_mm = scale * art_w_px
    used_h_mm = scale * art_h_px

    x_mm = offset_x_mm + (available_w_mm - used_w_mm) / 2.0
    y_mm = offset_y_mm + (available_h_mm - used_h_mm) / 2.0

    return Placement(
        canvas_w_mm=canvas_w_mm,
        canvas_h_mm=canvas_h_mm,
        image_w_px=image_w_px,
        image_h_px=image_h_px,
        rotation_deg=rotation_deg,
        scale_mm_per_px=scale,
        x_mm=x_mm,
        y_mm=y_mm,
        w_mm=used_w_mm,
        h_mm=used_h_mm,
        left_margin_mm=x_mm,
        right_margin_mm=canvas_w_mm - x_mm - used_w_mm,
        top_margin_mm=y_mm,
        bottom_margin_mm=canvas_h_mm - y_mm - used_h_mm,
        configured_long_margin_mm=long_margin_mm,
    )


def mm_to_px(value_mm: float, px_per_mm: float) -> int:
    return int(round(value_mm * px_per_mm))


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    anchor: str = "mm",
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def add_dimension_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    label_offset: tuple[int, int] = (0, 0),
) -> None:
    draw.line([start, end], fill=fill, width=2)
    arrow = 7
    if start[1] == end[1]:
        draw.line([(start[0], start[1]), (start[0] + arrow, start[1] - arrow)], fill=fill, width=2)
        draw.line([(start[0], start[1]), (start[0] + arrow, start[1] + arrow)], fill=fill, width=2)
        draw.line([(end[0], end[1]), (end[0] - arrow, end[1] - arrow)], fill=fill, width=2)
        draw.line([(end[0], end[1]), (end[0] - arrow, end[1] + arrow)], fill=fill, width=2)
    else:
        draw.line([(start[0], start[1]), (start[0] - arrow, start[1] + arrow)], fill=fill, width=2)
        draw.line([(start[0], start[1]), (start[0] + arrow, start[1] + arrow)], fill=fill, width=2)
        draw.line([(end[0], end[1]), (end[0] - arrow, end[1] - arrow)], fill=fill, width=2)
        draw.line([(end[0], end[1]), (end[0] + arrow, end[1] - arrow)], fill=fill, width=2)

    mid_x = (start[0] + end[0]) / 2 + label_offset[0]
    mid_y = (start[1] + end[1]) / 2 + label_offset[1]
    bbox = draw.textbbox((mid_x, mid_y), label, font=font, anchor="mm")
    pad = 4
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=4,
        fill=(255, 255, 255),
        outline=fill,
        width=1,
    )
    draw_centered_text(draw, (mid_x, mid_y), label, font, fill)


def render_preview(
    image: Image.Image,
    placement: Placement,
    output_path: Path,
    image_path: Path,
    px_per_mm: float,
) -> None:
    title_font = load_font(24, bold=True)
    label_font = load_font(16)
    small_font = load_font(13)

    pad_left = 130
    pad_right = 280
    pad_top = 135
    pad_bottom = 120

    canvas_w_px = mm_to_px(placement.canvas_w_mm, px_per_mm)
    canvas_h_px = mm_to_px(placement.canvas_h_mm, px_per_mm)
    out_w = pad_left + canvas_w_px + pad_right
    out_h = pad_top + canvas_h_px + pad_bottom

    preview = Image.new("RGB", (out_w, out_h), (238, 240, 243))
    draw = ImageDraw.Draw(preview)

    canvas_x = pad_left
    canvas_y = pad_top
    canvas_rect = (canvas_x, canvas_y, canvas_x + canvas_w_px, canvas_y + canvas_h_px)
    draw.rectangle(canvas_rect, fill=(255, 255, 255), outline=(20, 24, 31), width=3)

    image_for_canvas = image.convert("RGB")
    if placement.rotation_deg == 90:
        image_for_canvas = image_for_canvas.rotate(-90, expand=True)

    placed_w_px = mm_to_px(placement.w_mm, px_per_mm)
    placed_h_px = mm_to_px(placement.h_mm, px_per_mm)
    image_for_canvas.thumbnail((placed_w_px, placed_h_px), Image.Resampling.LANCZOS)
    image_for_canvas = image_for_canvas.resize((placed_w_px, placed_h_px), Image.Resampling.LANCZOS)

    image_x = canvas_x + mm_to_px(placement.x_mm, px_per_mm)
    image_y = canvas_y + mm_to_px(placement.y_mm, px_per_mm)
    preview.paste(image_for_canvas, (image_x, image_y))
    draw.rectangle(
        (image_x, image_y, image_x + placed_w_px, image_y + placed_h_px),
        outline=(32, 103, 198),
        width=3,
    )

    draw.text((canvas_x, 28), "Canvas layout pre-check", font=title_font, fill=(20, 24, 31))
    subtitle = (
        f"Canvas {fmt_mm(placement.canvas_w_mm)} x {fmt_mm(placement.canvas_h_mm)}; "
        f"image {placement.image_w_px} x {placement.image_h_px}px; "
        f"rotation {placement.rotation_deg} deg"
    )
    draw.text((canvas_x, 60), subtitle, font=label_font, fill=(55, 65, 81))

    add_dimension_arrow(
        draw,
        (canvas_x, canvas_y - 34),
        (canvas_x + canvas_w_px, canvas_y - 34),
        fmt_mm(placement.canvas_w_mm),
        label_font,
        (20, 24, 31),
        (0, -18),
    )
    add_dimension_arrow(
        draw,
        (canvas_x - 34, canvas_y),
        (canvas_x - 34, canvas_y + canvas_h_px),
        fmt_mm(placement.canvas_h_mm),
        label_font,
        (20, 24, 31),
        (-42, 0),
    )

    blue = (32, 103, 198)
    if placement.left_margin_mm > 0.01:
        add_dimension_arrow(
            draw,
            (canvas_x, image_y + placed_h_px // 2),
            (image_x, image_y + placed_h_px // 2),
            fmt_mm(placement.left_margin_mm),
            small_font,
            blue,
            (0, -18),
        )
    if placement.right_margin_mm > 0.01:
        add_dimension_arrow(
            draw,
            (image_x + placed_w_px, image_y + placed_h_px // 2),
            (canvas_x + canvas_w_px, image_y + placed_h_px // 2),
            fmt_mm(placement.right_margin_mm),
            small_font,
            blue,
            (0, -18),
        )
    if placement.top_margin_mm > 0.01:
        add_dimension_arrow(
            draw,
            (image_x + placed_w_px // 2, canvas_y),
            (image_x + placed_w_px // 2, image_y),
            fmt_mm(placement.top_margin_mm),
            small_font,
            blue,
            (58, 0),
        )
    if placement.bottom_margin_mm > 0.01:
        add_dimension_arrow(
            draw,
            (image_x + placed_w_px // 2, image_y + placed_h_px),
            (image_x + placed_w_px // 2, canvas_y + canvas_h_px),
            fmt_mm(placement.bottom_margin_mm),
            small_font,
            blue,
            (68, 0),
        )

    info_x = canvas_x + canvas_w_px + 28
    info_y = canvas_y
    lines = [
        "Placement",
        f"Artwork size: {fmt_mm(placement.w_mm)} x {fmt_mm(placement.h_mm)}",
        f"Scale: {placement.scale_mm_per_px:.5f} mm/px",
        f"Left: {fmt_mm(placement.left_margin_mm)}",
        f"Right: {fmt_mm(placement.right_margin_mm)}",
        f"Top: {fmt_mm(placement.top_margin_mm)}",
        f"Bottom: {fmt_mm(placement.bottom_margin_mm)}",
        "",
        "Config rule",
        f"Long-side margin: {fmt_mm(placement.configured_long_margin_mm)}",
        "Only the longest canvas side receives",
        "this configured margin before fitting.",
    ]
    y = info_y
    for index, line in enumerate(lines):
        font = label_font if index in (0, 8) else small_font
        fill = (20, 24, 31) if index in (0, 8) else (55, 65, 81)
        draw.text((info_x, y), line, font=font, fill=fill)
        y += 26 if index in (0, 8) else 21

    footer = f"Generated from {image_path.name} using pbn.config.DEFAULT_CONFIG"
    draw.text((canvas_x, out_h - 44), footer, font=small_font, fill=(75, 85, 99))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)


def write_summary(path: Path, image_path: Path, placement: Placement) -> None:
    text = "\n".join(
        [
            "Canvas layout pre-check",
            f"Input image: {image_path}",
            f"Canvas: {fmt_mm(placement.canvas_w_mm)} x {fmt_mm(placement.canvas_h_mm)}",
            f"Image pixels: {placement.image_w_px} x {placement.image_h_px}",
            f"Rotation: {placement.rotation_deg} deg",
            f"Placed artwork: {fmt_mm(placement.w_mm)} x {fmt_mm(placement.h_mm)}",
            f"Scale: {placement.scale_mm_per_px:.6f} mm/px",
            f"Configured long-side margin: {fmt_mm(placement.configured_long_margin_mm)}",
            f"Actual left margin: {fmt_mm(placement.left_margin_mm)}",
            f"Actual right margin: {fmt_mm(placement.right_margin_mm)}",
            f"Actual top margin: {fmt_mm(placement.top_margin_mm)}",
            f"Actual bottom margin: {fmt_mm(placement.bottom_margin_mm)}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a labeled preview of how the configured image fits on the configured canvas."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Image path. Defaults to pbn.config.DEFAULT_CONFIG['input'].",
    )
    parser.add_argument(
        "--output",
        default=str(Path("Pre-check") / "canvas_layout_precheck.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--summary",
        default=str(Path("Pre-check") / "canvas_layout_precheck.txt"),
        help="Text summary output path.",
    )
    parser.add_argument(
        "--px-per-mm",
        type=float,
        default=4.0,
        help="Preview rendering scale. This does not affect physical layout math.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)

    image_path = Path(args.input or cfg["input"])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    placement = compute_placement(image.size, cfg)
    render_preview(image, placement, output_path.resolve(), image_path, max(1.0, float(args.px_per_mm)))
    write_summary(summary_path.resolve(), image_path, placement)

    print(f"Saved preview: {output_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")
    print(
        "Margins: "
        f"left {fmt_mm(placement.left_margin_mm)}, "
        f"right {fmt_mm(placement.right_margin_mm)}, "
        f"top {fmt_mm(placement.top_margin_mm)}, "
        f"bottom {fmt_mm(placement.bottom_margin_mm)}"
    )


if __name__ == "__main__":
    main()
