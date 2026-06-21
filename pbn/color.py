from __future__ import annotations

from typing import Sequence, Tuple

import colorsys
import numpy as np
from skimage.color import deltaE_ciede2000 as _deltaE_ciede2000

try:
    from colour.difference import delta_E as _colour_delta_E
except Exception:
    _colour_delta_E = None

DELTA_E_METHOD = "colour_ciede2000"

# ---------------------------
# Color space & conversion helpers
# ---------------------------
def srgb_to_linear_arr(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert sRGB (0..1) to linear light (0..1), vectorized."""
    rgb_arr = np.clip(rgb_arr, 0.0, 1.0)
    return np.where(rgb_arr <= 0.04045,
                    rgb_arr / 12.92,
                    ((rgb_arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_arr(lin: np.ndarray) -> np.ndarray:
    """Convert linear light (0..1) to sRGB (0..1), vectorized."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308,
                    12.92 * lin,
                    1.055 * np.power(lin, 1 / 2.4) - 0.055)


def srgb8_to_xyz(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 (0..255) to XYZ (D65)."""
    lin = srgb_to_linear_arr(rgb_u8.astype(np.float32) / 255.0)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151520, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    return M @ lin


def xyz_to_srgb8(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ (D65) to sRGB uint8 (0..255)."""
    M = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32)
    lin = M @ xyz
    srgb = np.clip(linear_to_srgb_arr(lin), 0.0, 1.0)
    return srgb * 255.0


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to CIELAB (L*, a*, b*)."""
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t):
        return np.where(t > (6 / 29) ** 3,
                        np.cbrt(t),
                        (1 / 3) * (29 / 6) ** 2 * t + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b], dtype=np.float32)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to XYZ."""
    L, a, b = lab
    Yn = 1.0;
    Xn = 0.95047;
    Zn = 1.08883
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t):
        return np.where(t > 6 / 29,
                        t ** 3,
                        3 * (6 / 29) ** 2 * (t - 4 / 29))

    x = Xn * finv(fx)
    y = Yn * finv(fy)
    z = Zn * finv(fz)
    return np.array([x, y, z], dtype=np.float32)


def rgb8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 to CIELAB."""
    return xyz_to_lab(srgb8_to_xyz(rgb_u8))


def lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to sRGB uint8."""
    return xyz_to_srgb8(lab_to_xyz(lab))


def relative_luminance(rgb_u8: Sequence[int]) -> float:
    """Compute perceptual relative luminance (Y) from sRGB uint8."""
    lin = srgb_to_linear_arr(np.array(rgb_u8, dtype=np.float32) / 255.0)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def Lstar_from_rgb(rgb_u8: Sequence[int]) -> float:
    """Extract L* from a color in sRGB uint8."""
    return float(np.clip(rgb8_to_lab(np.array(rgb_u8, dtype=np.float32))[0], 0, 100))


def deltaE_lab(rgb1_u8: Sequence[int], rgb2_u8: Sequence[int]) -> float:
    """Compute perceptual Delta E using the configured color-science backend."""
    lab1 = rgb8_to_lab(np.array(rgb1_u8, dtype=np.float32))
    lab2 = rgb8_to_lab(np.array(rgb2_u8, dtype=np.float32))
    method = str(globals().get("DELTA_E_METHOD", "colour_ciede2000")).lower()

    if method in ("cie76", "deltae76", "lab", "euclidean"):
        return float(np.linalg.norm(lab1 - lab2))

    if method in ("colour", "colour_ciede2000", "colour-science", "colour-science_ciede2000"):
        if _colour_delta_E is not None:
            try:
                return float(_colour_delta_E(lab1, lab2, method="CIE 2000"))
            except Exception:
                pass

    try:
        return float(_deltaE_ciede2000(lab1[np.newaxis, :], lab2[np.newaxis, :])[0])
    except Exception:
        return float(np.linalg.norm(lab1 - lab2))


def rgb_to_hsv(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """Convert sRGB uint8 to (h, s, v) in [deg, 0..1, 0..1]."""
    rf, gf, bf = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    return h * 360.0, s, v
