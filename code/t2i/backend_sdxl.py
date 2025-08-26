# -*- coding: utf-8 -*-
# ==========================================================
# 概览
# SDXL 文生图后端：支持 Base-only 与 Base+Refiner 两阶段推理，
# 多候选生成、固定/补齐随机种子、按中英自动选择增强模版。
#
# Overview
# SDXL text-to-image backend: supports Base-only and Base+Refiner
# two-stage inference, multi-candidate generation, reproducible seeds,
# and language-aware prompt augmentation.
# ==========================================================

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
from diffusers import DiffusionPipeline


# ------------------------------
# Model loading
# ------------------------------
def load_sdxl(
    base_path: str,
    *,
    refiner_path: Optional[str] = None,
    use_bf16: bool = True,
    use_refiner: bool = True,
):
    """
    Load SDXL pipelines (Base-only or Base+Refiner).

    Args:
        base_path: Path to SDXL Base.
        refiner_path: Path to SDXL Refiner (optional).
        use_bf16: Use bfloat16 precision if True; otherwise fp16.
        use_refiner: Whether to load the Refiner.

    Returns:
        (base, refiner): Base pipeline and optional Refiner (or None).
    """
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load Base
    base = DiffusionPipeline.from_pretrained(
        base_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    try:
        base.vae.disable_slicing()
        base.vae.disable_tiling()
    except Exception:
        pass

    # Optionally load Refiner sharing VAE/text_encoder_2 with Base
    refiner = None
    if use_refiner and refiner_path:
        refiner = DiffusionPipeline.from_pretrained(
            refiner_path,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        try:
            refiner.vae.disable_slicing()
            refiner.vae.disable_tiling()
        except Exception:
            pass

    # Enable TF32 to speed up inference on Ampere+ GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return base, refiner


# ------------------------------
# Utilities
# ------------------------------
def _is_english(s: str) -> bool:
    """Return True if the string contains any ASCII letters."""
    s = str(s)
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)


def _maybe_augment(raw_prompt: str, augment: Optional[Dict[str, str]]):
    """
    Optionally attach augmentation templates.

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
    """Normalize seeds into a list of ints."""
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


# ------------------------------
# Main inference
# ------------------------------
def gen_sdxl(
    base,
    refiner,
    items: List[Tuple[int, str]],
    out_dir: str,
    *,
    width: int,
    height: int,
    total_steps: int,
    guidance_scale: float,
    high_noise_frac: float,
    seeds: Union[int, List[int]],
    use_refiner: bool,
    num_candidates: int = 1,
    augment: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate images with SDXL.

    Features:
        - Base-only or Base+Refiner two-stage flow.
        - Multi-candidate generation per prompt.
        - Seed policy matches the Qwen-Image backend:
          single seed auto-increments; list is truncated/auto-extended.

    Args:
        base: SDXL Base pipeline.
        refiner: SDXL Refiner pipeline (or None).
        items: [(index, prompt), ...].
        out_dir: Output directory.
        width, height: Output size.
        total_steps: Total sampling steps.
        guidance_scale: CFG scale.
        high_noise_frac: Split point between Base (denoising_end) and Refiner (denoising_start).
        seeds: Single int or list for reproducibility.
        use_refiner: Whether to run the two-stage flow.
        num_candidates: Images per prompt.
        augment: Optional prompt augmentation config.

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
        pos_prompt, neg_prompt = _maybe_augment(raw_prompt, augment)

        # Use CPU generators for determinism across stages; pipelines accept them.
        gens = [torch.Generator(device="cpu").manual_seed(_effective_seed(i))
                for i in range(num_candidates)]

        try:
            # ---------- Base-only ----------
            if not use_refiner or refiner is None:
                result = base(
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=int(total_steps),
                    guidance_scale=float(guidance_scale),
                    generator=gens,
                    num_images_per_prompt=int(num_candidates),
                    width=int(width),
                    height=int(height),
                )
                images = result.images

            # ---------- Base + Refiner ----------
            else:
                prompt_list = [pos_prompt] * int(num_candidates)
                neg_list = [neg_prompt] * int(num_candidates)

                # Stage 1: Base -> latents up to high_noise_frac
                latents = base(
                    prompt=prompt_list,
                    negative_prompt=neg_list,
                    num_inference_steps=int(total_steps),
                    guidance_scale=float(guidance_scale),
                    denoising_end=float(high_noise_frac),
                    output_type="latent",
                    generator=gens,
                    num_images_per_prompt=1,
                    width=int(width),
                    height=int(height),
                ).images

                # Stage 2: Refiner -> final images from high_noise_frac
                images = refiner(
                    prompt=prompt_list,
                    negative_prompt=neg_list,
                    num_inference_steps=int(total_steps),
                    denoising_start=float(high_noise_frac),
                    image=latents,
                ).images

            # Save results
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
                        "total_steps": int(total_steps),
                        "guidance_scale": float(guidance_scale),
                        "high_noise_frac": float(high_noise_frac),
                        "use_refiner": bool(use_refiner and refiner is not None),
                        "seed_effective": eff_seed,
                        "num_candidates": int(num_candidates),
                        "augmented": bool(augment.get("enable", False)) if augment else False,
                    }
                })

        except Exception as e:
            # Record error metadata
            out.append({
                "index": int(idx),
                "candidate_id": "NA",
                "path": "",
                "error": str(e),
                "meta": {
                    "width": int(width),
                    "height": int(height),
                    "total_steps": int(total_steps),
                    "guidance_scale": float(guidance_scale),
                    "high_noise_frac": float(high_noise_frac),
                    "use_refiner": bool(use_refiner and refiner is not None),
                    "seed_effective": None,
                    "num_candidates": int(num_candidates),
                    "augmented": bool(augment.get("enable", False)) if augment else False,
                }
            })

    return out
