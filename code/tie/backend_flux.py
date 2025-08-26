# -*- coding: utf-8 -*-
"""
说明
自然场景图像编辑（TIE）— FLUX 后端模块
- 作用：加载 FLUX.1-Kontext-dev 编辑模型，对输入图像按指令进行编辑并批量导出候选结果。
- 特性：
  1) 加载并初始化 FLUX-Kontext Pipeline（可选 bfloat16）
  2) 输入图像按比例缩放并补边为 1024×1024（letterbox），保持画幅不变
  3) 执行编辑推理并保存结果（支持多候选）
  4) 可选 Prompt 增强：按预设（lite/balanced/strict）附加正向后缀与负向约束
- 约定：
  * 随机种子可为单个或列表；若候选数大于种子数，则按首个种子递增补齐。
  * 输出命名：{index}_{kk}.jpg，kk 从 00 递增。

Overview
FLUX-based Image Editing Backend for TIE (Text-Instructed Editing)
- Purpose: Load FLUX.1-Kontext-dev edit model and run instruction-based edits, exporting multiple candidates.
- Features:
  1) Initialize FluxKontextPipeline (optional bfloat16)
  2) Letterbox inputs to 1024×1024 while preserving aspect ratio
  3) Perform editing inference and save results (multi-candidate)
  4) Optional prompt augmentation presets (lite/balanced/strict) for positive suffix & negative constraints
- Conventions:
  * Seeds: single int or list; if fewer than candidates, extend by incrementing from the first seed.
  * Output naming: {index}_{kk}.jpg with kk starting from 00.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageOps
import torch
from diffusers import FluxKontextPipeline


def load_flux_edit(model_path: str, use_bf16: bool = True):
    """
    Load FLUX-Kontext editing pipeline.

    Args:
        model_path: Local path or hub id of the model.
        use_bf16: Use bfloat16 for inference (recommended on modern GPUs).

    Returns:
        A FluxKontextPipeline placed on an appropriate device.
    """
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Keep VAE behavior consistent (no slicing/tiling)
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.disable_slicing()
            pipe.vae.disable_tiling()
    except Exception:
        pass

    # Enable GPU perf knobs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return pipe


def _is_english(s: str) -> bool:
    """Return True if the string contains any English letters."""
    s = str(s)
    return any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in s)


def _maybe_augment(instruction: str, augment: Optional[Dict[str, str]], fallback_negative: str):
    """
    Build positive/negative prompts with optional augmentation presets.

    Rules:
      - If augmentation is disabled, return the raw instruction and fallback_negative.
      - If enabled:
          * Append the preset suffix_en to the positive prompt.
          * If the preset provides negative_en, it overrides fallback_negative.

    Args:
        instruction: Raw editing instruction.
        augment: Dict with keys: enable, mode, presets[mode]{suffix_en, negative_en}.
        fallback_negative: Default negative prompt.

    Returns:
        (pos, neg): Final positive and negative prompts.
    """
    s = str(instruction).strip()
    if not augment or not augment.get("enable", False):
        return s, fallback_negative

    mode = str(augment.get("mode", "balanced")).lower()
    presets = augment.get("presets", {})
    mode_cfg = presets.get(mode, {})

    suffix_en = mode_cfg.get("suffix_en", "")
    neg_en = mode_cfg.get("negative_en", "")

    pos = s + (suffix_en if suffix_en else "")
    neg = neg_en if neg_en else fallback_negative
    return pos, neg


def _seeds_to_list(seeds: Union[int, List[int], None]) -> List[int]:
    """Normalize `seeds` to a list[int]."""
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


def _letterbox_1024(im: Image.Image):
    """
    Letterbox the input image to 1024x1024 while preserving aspect ratio.

    Returns:
        im_sq: Letterboxed square image (1024x1024)
        crop_box: Box to crop the content back from the square image
        orig_wh: Original (width, height)
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
        fill=(127, 127, 127)
    )
    crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
    return im_sq, crop_box, (w0, h0)


def _unletterbox(out_sq: Image.Image, crop_box, orig_wh):
    """Undo letterbox: crop and resize back to the original size."""
    out_cropped = out_sq.crop(crop_box)
    return out_cropped.resize(orig_wh, Image.LANCZOS)


def gen_flux_edit(
    pipe,
    items: List[Tuple[int, str, str]],  # [(index, instruction, ori_image_path)]
    out_dir: str,
    *,
    steps: int,
    guidance_scale: float,
    seeds: Union[int, List[int]],
    num_candidates: int = 1,
    augment: Optional[Dict[str, str]] = None,
    negative_prompt: str = "",  # kept for parity with config; not passed to pipeline by design
) -> List[Dict[str, Any]]:
    """
    Run FLUX editing on a batch of items.

    Args:
        pipe: Loaded FluxKontextPipeline.
        items: List of (index, instruction, source_image_path).
        out_dir: Output directory for edited images.
        steps: Number of diffusion steps.
        guidance_scale: CFG-like guidance.
        seeds: Single int or list of ints; extended if needed.
        num_candidates: Number of images per instruction.
        augment: Optional augmentation config.
        negative_prompt: Default negative prompt (kept for interface compatibility).

    Returns:
        A list of dicts with metadata for each generated image (or error entries).
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
            # NOTE: Intentionally not passing `negative_prompt` to match original behavior.
            result = pipe(
                image=im_sq,
                prompt=pos,
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
