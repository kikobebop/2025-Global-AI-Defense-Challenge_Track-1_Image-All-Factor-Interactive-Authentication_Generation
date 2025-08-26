# ==========================================================
# 视觉文本编辑（VTTIE）任务 - FLUX-Kontext 编辑后端
# 功能（中文）：
#   1. 加载 FLUX-Kontext 编辑模型
#   2. 将原图等比缩放并填充至 1024×1024（保持纵横比）
#   3. 执行基于指令的编辑推理，并在保存前恢复为原始分辨率
#   4. 支持多候选生成、随机种子控制、增强模板与负向提示
#
# VTTIE (Visual Text-in-Image Editing) — FLUX-Kontext Edit Backend
# Features (English):
#   1. Load the FLUX-Kontext editing pipeline
#   2. Letterbox to 1024×1024 while preserving aspect ratio
#   3. Run instruction-based edits and restore to original resolution
#   4. Support multiple candidates, reproducible seeds, augmentation & negatives
# ==========================================================

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageOps
import torch
from diffusers import FluxKontextPipeline


def load_flux_edit(model_path: str, use_bf16: bool = True):
    """
    Load FLUX-Kontext edit pipeline with appropriate dtype/device and CUDA optimizations.

    Args:
        model_path (str): Local path or hub id of the model.
        use_bf16 (bool): Use bfloat16 precision when True; else float16.

    Returns:
        FluxKontextPipeline: Initialized edit pipeline placed on CUDA if available.
    """
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Disable VAE slicing/tiling if exposed for more consistent outputs
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.disable_slicing()
            pipe.vae.disable_tiling()
    except Exception:
        pass
    # Enable TF32 & cuDNN autotune on CUDA for better throughput
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return pipe


def _seeds_to_list(seeds: Union[int, List[int], None]) -> List[int]:
    """Normalize a single int or a list to a list of ints."""
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


def _letterbox_1024(im: Image.Image):
    """
    Resize with preserved aspect ratio and pad to a 1024×1024 square (gray background).

    Returns:
        im_sq (Image): Letterboxed square image.
        crop_box (tuple): (left, top, right, bottom) content box in the square.
        orig_wh (tuple): Original (width, height).
    """
    im = im.convert("RGB")
    w0, h0 = im.width, im.height
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"Invalid image size: {w0}x{h0}")
    S = 1024
    r = min(S / w0, S / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    pad_left = (S - new_w) // 2
    pad_right = S - new_w - pad_left
    pad_top = (S - new_h) // 2
    pad_bot = S - new_h - pad_top
    im_sq = ImageOps.expand(
        im_resized,
        border=(pad_left, pad_top, pad_right, pad_bot),
        fill=(127, 127, 127),
    )
    crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
    return im_sq, crop_box, (w0, h0)


def _unletterbox(out_sq: Image.Image, crop_box, orig_wh):
    """
    Crop the padded square back to content and resize to original resolution.

    Args:
        out_sq (Image): Output square image from the model.
        crop_box (tuple): The content region box inside the square.
        orig_wh (tuple): Original (width, height).

    Returns:
        Image: Final image in original resolution.
    """
    out_cropped = out_sq.crop(crop_box)
    return out_cropped.resize(orig_wh, Image.LANCZOS)


def _maybe_augment(instr: str, augment: Optional[Dict[str, Any]], fallback_negative: str):
    """
    Apply augmentation presets if enabled.

    Logic:
        - Only English presets are used here.
        - When augment.enable = True:
            * Append presets[mode].suffix_en to the instruction if provided
            * Override negative prompt with presets[mode].negative_en if provided
        - Otherwise, use fallback_negative.

    Returns:
        pos (str): Final positive prompt.
        neg (str): Final negative prompt.
    """
    s = str(instr).strip()
    if not augment or not augment.get("enable", False):
        return s, fallback_negative
    mode = str(augment.get("mode", "strict")).lower()
    presets = augment.get("presets", {})
    mode_cfg = presets.get(mode, {})
    suffix_en = mode_cfg.get("suffix_en", "")
    neg_en = mode_cfg.get("negative_en", "")
    pos = s + (suffix_en if suffix_en else "")
    neg = neg_en if neg_en else fallback_negative
    return pos, neg


def gen_flux_edit(
    pipe,
    items: List[Tuple[int, str, str]],  # [(index, instruction, ori_image_path)]
    out_dir: str,
    *,
    steps: int,
    guidance_scale: float,
    seeds: Union[int, List[int]],
    num_candidates: int = 1,
    augment: Optional[Dict[str, Any]] = None,
    negative_prompt: str = "",
) -> List[Dict[str, Any]]:
    """
    Batch-run FLUX-Kontext text-in-image edits.

    Args:
        pipe (FluxKontextPipeline): Loaded FLUX edit pipeline.
        items (list): Each item is (index, instruction, original_image_path).
        out_dir (str): Output directory for edited images.
        steps (int): Inference steps.
        guidance_scale (float): Guidance scale for editing strength.
        seeds (int/list): Reproducible seeds (single or list).
        num_candidates (int): Number of candidates per item.
        augment (dict): Augmentation presets (optional).
        negative_prompt (str): Fallback negative prompt.

    Returns:
        list: A list of metadata dicts per generated candidate.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []

    seeds_list = _seeds_to_list(seeds)
    base_seed = seeds_list[0] if seeds_list else 0

    def eff_seed(i: int) -> int:
        return seeds_list[i] if i < len(seeds_list) else (int(base_seed) + int(i))

    for idx, instr, img_path in items:
        im = Image.open(img_path).convert("RGB")
        im_sq, crop_box, orig_wh = _letterbox_1024(im)
        pos, neg = _maybe_augment(instr, augment, negative_prompt)

        gens = [torch.Generator(device="cpu").manual_seed(eff_seed(i))
                for i in range(num_candidates)]
        try:
            result = pipe(
                image=im_sq,
                prompt=pos,
                negative_prompt=neg if neg else None,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                generator=gens if num_candidates > 1 else gens[0],
                num_images_per_prompt=int(num_candidates),
            )
            images = result.images
            for i, img_out in enumerate(images):
                final_img = _unletterbox(img_out, crop_box, orig_wh)
                cid = f"{i:02d}"
                out_path = os.path.join(out_dir, f"{idx}_{cid}.jpg")
                final_img.save(out_path, quality=100)
                out.append({
                    "index": int(idx),
                    "candidate_id": cid,
                    "src_image": img_path,
                    "path": out_path,
                    "meta": {
                        "steps": int(steps),
                        "guidance_scale": float(guidance_scale),
                        "seed_effective": eff_seed(i),
                        "num_candidates": int(num_candidates),
                        "augmented": bool(augment.get("enable", False)) if augment else False,
                    }
                })
        except Exception as e:
            out.append({
                "index": int(idx),
                "candidate_id": "NA",
                "src_image": img_path,
                "path": "",
                "error": str(e),
                "meta": {
                    "steps": int(steps),
                    "guidance_scale": float(guidance_scale),
                    "seed_effective": None,
                    "num_candidates": int(num_candidates),
                    "augmented": bool(augment.get("enable", False)) if augment else False,
                }
            })
    return out
