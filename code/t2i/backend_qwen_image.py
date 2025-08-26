# -*- coding: utf-8 -*-
# ==========================================================
# 概览
# Qwen-Image 文生图后端
# - 负责模型加载与推理；支持多候选生成、固定/补齐随机种子；
# - 支持按中英文自动选择 Prompt 增强模版（正向/负向）。
#
# Overview
# Qwen-Image text-to-image backend
# - Loads the model and runs inference; supports multi-candidate generation,
#   fixed/auto-filled seeds for reproducibility;
# - Auto-picks prompt augmentation templates (positive/negative) for EN/ZH.
# ==========================================================

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
from diffusers import DiffusionPipeline


# ------------------------------
# Model loading
# ------------------------------
def load_qwen_image(model_path: str, use_bf16: bool = True):
    """
    Load Qwen-Image pipeline.

    Args:
        model_path: Path to the model (HF or local).
        use_bf16: Use bfloat16 precision if True; otherwise fp16.

    Returns:
        A DiffusionPipeline instance moved to CUDA.
    """
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to("cuda")
    try:
        # Disable VAE slicing/tiling for more consistent outputs.
        pipe.vae.disable_slicing()
        pipe.vae.disable_tiling()
    except Exception:
        pass

    # Enable TF32 to speed up inference on Ampere+ GPUs.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return pipe


# ------------------------------
# Utilities
# ------------------------------
def _is_english(s: str) -> bool:
    """Return True if the string contains any ASCII letters."""
    s = str(s)
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)


def _maybe_augment(raw_prompt: str, augment: Optional[Dict[str, str]]):
    """
    Optionally attach augmentation templates (positive & negative).

    Args:
        raw_prompt: Original prompt text.
        augment: Dict with keys {enable, suffix_en, suffix_cn, negative_en, negative_cn}.

    Returns:
        (pos_prompt, neg_prompt): Augmented positive & negative prompts.
    """
    if not augment or not augment.get("enable", False):
        return str(raw_prompt).strip(), ""

    s = str(raw_prompt).strip()
    if _is_english(s):
        pos = s + augment.get("suffix_en", "")
        neg = augment.get("negative_en", "")
    else:
        pos = s + augment.get("suffix_cn", "")
        neg = augment.get("negative_cn", "")
    return pos, neg


def _seeds_to_list(seeds: Union[int, List[int], None]) -> List[int]:
    """
    Normalize seeds into a list of ints.

    Args:
        seeds: Single int, list of ints, or None.

    Returns:
        List[int]
    """
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


# ------------------------------
# Main inference
# ------------------------------
def gen_qwen_image(
    pipe,
    items: List[Tuple[int, str]],   # [(index, prompt), ...]
    out_dir: str,
    *,
    width: int,
    height: int,
    steps: int,
    true_cfg: float,
    seeds: Union[int, List[int]],
    negative_prompt: str = "",
    num_candidates: int = 1,
    augment: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate images with Qwen-Image.

    Seeding policy:
        - If `seeds` is a single int:
            * N=1: use it as-is
            * N>1: auto-fill as [seed, seed+1, seed+2, ...]
        - If `seeds` is a list:
            * Use the first N elements
            * If not enough, keep incrementing from the first seed

    Filename:
        {index}_{kk}.jpg, where kk starts from 00.

    Args:
        pipe: Qwen-Image diffusion pipeline (CUDA).
        items: List of (index, prompt) pairs.
        out_dir: Output directory.
        width, height: Image size.
        steps: Sampling steps.
        true_cfg: "True CFG" scale for Qwen-Image.
        seeds: Seed(s) for reproducibility.
        negative_prompt: Global negative prompt (used only if augment is disabled).
        num_candidates: Number of images per item.
        augment: Optional dict for prompt augmentation.

    Returns:
        List of metadata dicts for each generated image.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out: List[Dict[str, Any]] = []

    seeds_list = _seeds_to_list(seeds)
    base_seed = seeds_list[0] if seeds_list else 0

    def _effective_seed(i: int) -> int:
        """Return the effective seed for the i-th candidate."""
        if i < len(seeds_list):
            return seeds_list[i]
        return int(base_seed) + int(i)

    for idx, raw_prompt in items:
        # Build positive/negative prompts
        if augment and augment.get("enable", False):
            pos_prompt, neg_prompt_auto = _maybe_augment(raw_prompt, augment)
            neg_prompt_final = neg_prompt_auto
        else:
            pos_prompt, neg_prompt_final = str(raw_prompt).strip(), negative_prompt

        # Prepare per-candidate generators
        gens = [torch.Generator(device="cuda").manual_seed(_effective_seed(i))
                for i in range(num_candidates)]

        try:
            # Run the pipeline
            result = pipe(
                prompt=pos_prompt,
                negative_prompt=neg_prompt_final,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                true_cfg_scale=float(true_cfg),
                num_images_per_prompt=int(num_candidates),
                generator=gens,
            )

            images = result.images
            for i, img in enumerate(images):
                cid = f"{i:02d}"
                out_path = os.path.join(out_dir, f"{idx}_{cid}.jpg")
                img.save(out_path, quality=100)

                eff_seed = _effective_seed(i)
                out.append({
                    "index": int(idx),
                    "candidate_id": cid,
                    "path": out_path,
                    "meta": {
                        "width": int(width),
                        "height": int(height),
                        "steps": int(steps),
                        "true_cfg": float(true_cfg),
                        "seed_effective": eff_seed,
                        "num_candidates": int(num_candidates),
                        "augmented": bool(augment.get("enable", False)) if augment else False,
                    }
                })

        except Exception as e:
            # Record error metadata on failure
            out.append({
                "index": int(idx),
                "candidate_id": "NA",
                "path": "",
                "error": str(e),
                "meta": {
                    "width": int(width),
                    "height": int(height),
                    "steps": int(steps),
                    "true_cfg": float(true_cfg),
                    "seed_effective": None,
                    "num_candidates": int(num_candidates),
                    "augmented": bool(augment.get("enable", False)) if augment else False,
                }
            })

    return out
