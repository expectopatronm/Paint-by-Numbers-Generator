from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as _tv_transforms
from transformers import AutoModelForImageSegmentation as _HFSegModel, AutoImageProcessor as _HFProcessor

def rmbg_alpha_matte(input_path: str,
                     *,
                     model_dir: str = "rmbg",
                     device: str | None = None,
                     target_size: tuple[int,int] = (1024,1024)) -> np.ndarray:
    """
    Returns alpha matte in [0..1] as float32, same W×H as the input image.
    Uses BRIA RMBG-2.0 weights from `model_dir` (Hugging Face layout).
    """
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.dirname(os.path.realpath(__file__)))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    orig_w, orig_h = img.size

    # Load processor & model (trust_remote_code per model card)
    proc = _HFProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = _HFSegModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()

    # Basic resize+normalize; model cards often expect ~1024 square
    tfm = _tv_transforms.Compose([
        _tv_transforms.Resize(target_size),
        _tv_transforms.ToTensor(),
        _tv_transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
    ])
    inp = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
        # BRIA RMBG returns last tensor as 1×1×H×W logits
        logits = out[-1]
        matte_small = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

    matte_u8 = (np.clip(matte_small, 0, 1) * 255).astype(np.uint8)
    matte_u8 = np.array(Image.fromarray(matte_u8).resize((orig_w, orig_h), Image.BILINEAR))
    return matte_u8.astype(np.float32) / 255.0


def _latest_image_in_tree(root_dir: str, *, after_ts: float) -> str | None:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
    candidates = []
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            path = os.path.join(base, name)
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime >= after_ts:
                candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _maybe_upscale_with_supir(input_path: str,
                              *,
                              enable: bool,
                              ok_min: int,
                              repo_dir: str,
                              python_bin: str,
                              model_dir: str,
                              supir_sign: str = "Q",
                              upscale: int = 2,
                              min_size: int = 1024,
                              edm_steps: int = 50,
                              s_noise: float = 1.02,
                              s_cfg: float = 4.0,
                              spt_linear_cfg: float = 1.0,
                              s_stage2: float = 1.0,
                              color_fix_type: str = "Wavelet",
                              a_prompt: str = "",
                              n_prompt: str = "",
                              ae_dtype: str = "bf16",
                              diff_dtype: str = "fp16",
                              no_llava: bool = True,
                              use_tile_vae: bool = True,
                              loading_half_params: bool = False,
                              extra_args: Sequence[str] | None = None) -> str:
    """
    If enabled and the image is below ok_min, run a local SUPIR checkout.
    SUPIR is non-commercial-only upstream; this wrapper assumes local personal use.
    """
    if not enable:
        return input_path

    try:
        with Image.open(input_path) as im:
            w, h = im.size
    except Exception as e:
        raise RuntimeError(f"Could not open input for SUPIR pre-check: {e}") from e

    longest = max(w, h)
    if longest >= ok_min:
        print(f"SUPIR upscale check: longest={longest}px >= {ok_min}px -> no upscale.")
        return input_path

    repo_dir = os.path.abspath(repo_dir)
    test_py = os.path.join(repo_dir, "test.py")
    if not os.path.exists(test_py):
        raise FileNotFoundError(f"SUPIR repo not found at {repo_dir}. Expected {test_py}.")

    python_bin = str(python_bin or "python")
    if python_bin.lower() != "python" and not os.path.isabs(python_bin):
        python_bin = os.path.abspath(python_bin)
    model_dir = os.path.abspath(model_dir)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    scale = max(1, int(upscale))
    root = os.path.splitext(os.path.abspath(input_path))[0]
    final_path = f"{root}.supirx{scale}.png"

    with tempfile.TemporaryDirectory(prefix="pbn_supir_") as tmp:
        in_dir = os.path.join(tmp, "input")
        out_dir = os.path.join(tmp, "output")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        staged_input = os.path.join(in_dir, os.path.basename(input_path))
        shutil.copy2(input_path, staged_input)

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", repo_dir)
        env.setdefault("SUPIR_MODEL_DIR", model_dir)

        started = time.time()
        cmd = [
            python_bin,
            test_py,
            "--img_dir", in_dir,
            "--save_dir", out_dir,
            "--upscale", str(scale),
            "--SUPIR_sign", str(supir_sign).upper(),
            "--min_size", str(int(min_size)),
            "--edm_steps", str(int(edm_steps)),
            "--s_noise", str(float(s_noise)),
            "--s_cfg", str(float(s_cfg)),
            "--spt_linear_CFG", str(float(spt_linear_cfg)),
            "--s_stage2", str(float(s_stage2)),
            "--color_fix_type", str(color_fix_type),
            "--a_prompt", str(a_prompt),
            "--n_prompt", str(n_prompt),
            "--ae_dtype", str(ae_dtype),
            "--diff_dtype", str(diff_dtype),
        ]
        if no_llava:
            cmd.append("--no_llava")
        if use_tile_vae:
            cmd.append("--use_tile_vae")
        if loading_half_params:
            cmd.append("--loading_half_params")
        if extra_args:
            cmd.extend(str(x) for x in extra_args)

        print(f"Running SUPIR: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
        except Exception as e:
            raise RuntimeError(f"SUPIR failed and no original-image fallback is allowed: {e}") from e

        produced = _latest_image_in_tree(out_dir, after_ts=started)
        if not produced:
            raise RuntimeError("SUPIR finished but no output image was found; no original-image fallback is allowed.")

        try:
            Image.open(produced).save(final_path)
            with Image.open(final_path) as up:
                uw, uh = up.size
            print(f"SUPIR upscaled/restored to: {uw}x{uh} -> {final_path}")
            return final_path
        except Exception as e:
            raise RuntimeError(f"Could not preserve SUPIR output; no original-image fallback is allowed: {e}") from e


def _maybe_upscale_input(input_path: str, args) -> str:
    enable = bool(getattr(args, "enable_upscale", True))
    ok_min = int(getattr(args, "upscale_ok_min_long", 3000))

    if not enable:
        return input_path

    with Image.open(input_path) as im:
        longest = max(im.size)
    upscale_cfg = getattr(args, "supir_upscale", "auto")
    if str(upscale_cfg).lower() == "auto":
        scale = 1
        for candidate in tuple(getattr(args, "supir_upscale_choices", (1, 2, 3, 4))):
            if longest * int(candidate) >= ok_min:
                scale = int(candidate)
                break
        else:
            scale = max(int(x) for x in tuple(getattr(args, "supir_upscale_choices", (1, 2, 3, 4))))
    else:
        scale = int(upscale_cfg)

    prioritizing = str(getattr(args, "supir_prioritizing", "Quality")).lower()
    texture = float(getattr(args, "supir_texture_richness", 1.0))
    creativity = float(getattr(args, "supir_creativity", 0.0))

    if "quality" in prioritizing:
        default_s_stage2 = 0.93
        default_s_cfg = 6.0
        default_spt_linear_cfg = 3.0
        default_s_noise = 1.02
    else:
        default_s_stage2 = 1.0
        default_s_cfg = 4.0
        default_spt_linear_cfg = 1.0
        default_s_noise = 1.01

    # Match the website intent: texture richness raises perceptual restoration,
    # creativity raises prompt influence. A blank description plus creativity=0
    # keeps the result faithful for painting references.
    default_s_stage2 = float(np.clip(default_s_stage2 - 0.04 * max(0.0, texture - 1.0), 0.75, 1.0))
    default_s_cfg = float(np.clip(default_s_cfg + creativity * 2.0, 1.0, 10.0))

    result = _maybe_upscale_with_supir(
        input_path,
        enable=enable,
        ok_min=ok_min,
        repo_dir=str(getattr(args, "supir_repo_dir", "SUPIR")),
        python_bin=str(getattr(args, "supir_python", os.path.join("venv", "Scripts", "python.exe"))),
        model_dir=str(getattr(args, "supir_model_dir", "supir_models")),
        supir_sign=str(getattr(args, "supir_sign", "Q" if "quality" in prioritizing else "F")),
        upscale=scale,
        min_size=int(getattr(args, "supir_min_size", 1024)),
        edm_steps=int(getattr(args, "supir_edm_steps", 50)),
        s_noise=float(getattr(args, "supir_s_noise", default_s_noise)),
        s_cfg=float(getattr(args, "supir_s_cfg", default_s_cfg)),
        spt_linear_cfg=float(getattr(args, "supir_spt_linear_cfg", default_spt_linear_cfg)),
        s_stage2=float(getattr(args, "supir_s_stage2", default_s_stage2)),
        color_fix_type=str(getattr(args, "supir_color_fix_type", "Wavelet")),
        a_prompt=str(getattr(args, "supir_a_prompt", getattr(args, "supir_image_description", ""))),
        n_prompt=str(getattr(args, "supir_n_prompt", "")),
        ae_dtype=str(getattr(args, "supir_ae_dtype", "bf16")),
        diff_dtype=str(getattr(args, "supir_diff_dtype", "fp16")),
        no_llava=bool(getattr(args, "supir_no_llava", True)),
        use_tile_vae=bool(getattr(args, "supir_use_tile_vae", True)),
        loading_half_params=bool(getattr(args, "supir_loading_half_params", False)),
        extra_args=list(getattr(args, "supir_extra_args", []) or []),
    )
    if result == input_path and longest < ok_min:
        raise RuntimeError("SUPIR did not produce an upscaled image; no original-image fallback is allowed.")
    return result

