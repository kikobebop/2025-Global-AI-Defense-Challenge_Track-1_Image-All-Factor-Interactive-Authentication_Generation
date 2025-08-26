# -*- coding: utf-8 -*-
"""
说明
自然场景图像编辑（TIE）— Qwen-Image-Edit 后端模块
- 作用：
  1) 加载 Qwen-Image-Edit 模型
  2) 支持中英文 Prompt 的增强模式（Prompt Augmentation）
  3) 使用 Letterbox 将输入图像适配至 1024×1024，同时保持原始纵横比
  4) 执行编辑推理并批量输出候选结果
- 约定：
  * 随机种子可为单个值或列表；当候选数多于种子数时，会按首个种子递增补齐。
  * 输出文件命名格式：`{index}_{kk}.jpg`，kk 从 00 递增。

Overview
Qwen-Image-Edit Backend for TIE (Text-Instructed Editing)
- Features:
  1) Load Qwen-Image-Edit model
  2) Optional prompt augmentation for Chinese/English instructions
  3) Letterbox input images to 1024×1024 while preserving aspect ratio
  4) Perform editing inference and export multi-candidate results
- Conventions:
  * Seeds: Single int or list; if candidates exceed seeds, extend by incrementing the first seed.
  * Output naming: `{index}_{kk}.jpg`, with kk starting from 00.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageOps
import torch
from diffusers.pipelines.qwenimage import QwenImageEditPipeline


def load_qwen_edit(model_path: str, use_bf16: bool = True):
    """
    Load Qwen-Image-Edit model.

    Args:
        model_path: Local path or hub id of the model.
        use_bf16: Whether to use bfloat16 precision (recommended True).

    Returns:
        A QwenImageEditPipeline on the appropriate device.
    """
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Disable slicing/tiling to keep VAE consistent
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.disable_slicing()
            pipe.vae.disable_tiling()
    except Exception:
        pass

    # GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return pipe


def _is_english(s: str) -> bool:
    """Return True if the string contains English letters."""
    s = str(s)
    return any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in s)


def _contains_chinese(s: str) -> bool:
    """Return True if the string contains Chinese characters."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in str(s))


def _maybe_augment(instruction: str, augment: Optional[Dict[str, str]], fallback_negative: str):
    """
    Build positive/negative prompts with optional augmentation.

    Rules:
        - If augmentation is disabled: return raw instruction and fallback negative prompt.
        - If enabled and instruction is Chinese:
            * Prepend prefix_cn to the instruction
            * Use negative_cn if provided; otherwise fallback to default negative
        - If instruction is English: no augmentation.

    Args:
        instruction: Raw editing instruction.
        augment: Config dict with optional "prefix_cn" and "negative_cn".
        fallback_negative: Default negative prompt.

    Returns:
        pos: Positive prompt (possibly augmented)
        neg: Negative prompt
    """
    s = str(instruction).strip()
    if not augment or not augment.get("enable", False):
        return s, fallback_negative

    if _contains_chinese(s):
        prefix = augment.get("prefix_cn", "").strip()
        neg_cn = augment.get("negative_cn", "").strip()
        pos = (prefix + "\n" + s) if prefix else s
        neg = neg_cn if neg_cn else fallback_negative
        return pos, neg
    else:
        return s, fallback_negative


def _seeds_to_list(seeds: Union[int, List[int], None]) -> List[int]:
    """Normalize seeds to a list of integers."""
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


def _letterbox_1024(im: Image.Image):
    """
    Letterbox image to 1024×1024, preserving aspect ratio.

    Returns:
        im_sq: Letterboxed square image.
        crop_box: Coordinates for cropping back to original ratio.
        orig_wh: Original (width, height).
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
    """Crop and resize back to original aspect ratio."""
    out_cropped = out_sq.crop(crop_box)
    return out_cropped.resize(orig_wh, Image.LANCZOS)


def gen_qwen_edit(
    pipe,
    items: List[Tuple[int, str, str]],  # [(index, instruction, ori_image_path)]
    out_dir: str,
    *,
    steps: int,
    true_cfg: float,
    seeds: Union[int, List[int]],
    negative_prompt: str = "",
    num_candidates: int = 1,
    augment: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform Qwen-Image-Edit inference for natural scene editing.

    Args:
        pipe: Loaded QwenImageEditPipeline.
        items: List of tuples (index, instruction, original_image_path).
        out_dir: Directory to save edited images.
        steps: Number of inference steps.
        true_cfg: True CFG scale for matching prompt fidelity.
        seeds: Single int or list; extended if fewer than candidates.
        negative_prompt: Default negative prompt.
        num_candidates: Number of images per instruction.
        augment: Optional augmentation config.

    Returns:
        List of dicts with metadata for each generated image (or error info).
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
        pos, neg_final = _maybe_augment(instr, augment, negative_prompt)

        gens = [torch.Generator(device="cpu").manual_seed(eff_seed(i))
                for i in range(num_candidates)]
        try:
            # Perform inference
            result = pipe(
                image=im_sq,
                prompt=pos,
                generator=gens if num_candidates > 1 else gens[0],
                true_cfg_scale=float(true_cfg),
                negative_prompt=neg_final,
                num_inference_steps=int(steps),
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
                        "true_cfg": float(true_cfg),
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
                    "true_cfg": float(true_cfg),
                    "seed_effective": None,
                    "num_candidates": int(num_candidates),
                    "augmented": bool(augment.get("enable", False)) if augment else False,
                }
            })
    return out
