# ==========================================================
# 视觉文字编辑（VTTIE）任务 - Qwen-Image 编辑后端
# 功能（中文）：
#   1. 加载 Qwen-Image 编辑模型
#   2. 根据文字编辑指令生成严格受限的“仅改字”增强提示
#   3. 支持多候选生成、随机种子控制、增强模板和负向提示
#   4. 自动保持原图比例和尺寸恢复
#
# VTTIE (Visual Text-in-Image Editing) — Qwen-Image Edit Backend
# Features (English):
#   1. Load the Qwen-Image edit pipeline
#   2. Wrap prompts to strictly constrain edits to text-only regions
#   3. Support multiple candidates, reproducible seeds, augmentation & negatives
#   4. Preserve original aspect ratio (letterbox) and restore resolution
# ==========================================================

import os, re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageOps
import torch
from diffusers.pipelines.qwenimage import QwenImageEditPipeline


def load_qwen_edit(model_path: str, use_bf16: bool = True):
    """Load Qwen-Image edit pipeline with proper dtype and device, enable TF32 on CUDA."""
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Disable VAE slicing/tiling for consistency (if available)
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.disable_slicing()
            pipe.vae.disable_tiling()
    except Exception:
        pass
    # Throughput optimization on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return pipe


# ----------------------------
# Helpers (language detection)
# ----------------------------
def _has_cn(s: str) -> bool:
    """Return True if string contains any Chinese characters."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in str(s))

def _norm_quotes(s: str) -> str:
    """Normalize curly quotes to straight quotes."""
    return s.replace("“", '"').replace("”", '"').replace("‘","'").replace("’","'")

def _first_quoted(s: str) -> str:
    """Extract the first quoted substring as anchor text ("" or '')."""
    s = _norm_quotes(s)
    m = re.search(r'"([^"]+)"', s)
    if m:
        return m.group(1).strip()
    m = re.search(r"'([^']+)'", s)
    return m.group(1).strip() if m else ""


# ----------------------------
# Task type detection
# ----------------------------
def _detect_task_kind(prompt: str) -> str:
    """
    Detect edit kind from instruction:
      - replace: replace existing text
      - remove : remove text
      - add    : add new text
      - generic: fallback
    """
    p_raw = _norm_quotes(prompt)
    p = p_raw.lower()
    has_remove = any(k in p for k in ["remove","delete","erase"]) or any(k in p_raw for k in ["删除","去除","去掉","移除","清除","擦除"])
    has_add    = any(k in p for k in ["add","insert","write"]) or any(k in p_raw for k in ["加入","增加","添加","加上","写上","插入","新增"])
    has_replace= any(k in p for k in ["replace","change"]) or any(k in p_raw for k in ["替换","改为","改成","换成","更改","改掉"])
    if has_remove and has_add:
        return "replace"
    if has_replace:
        return "replace"
    if has_remove:
        return "remove"
    if has_add:
        return "add"
    return "generic"


# ----------------------------
# Prompt wrapper for VTTIE
# ----------------------------
def _wrap_prompt_for_vttie(prompt: str) -> str:
    """
    Build a strict text-only edit instruction prefix (+anchor note if present),
    in Chinese or English depending on the input language.
    """
    kind = _detect_task_kind(prompt)
    anchor = _first_quoted(prompt)
    cn = _has_cn(prompt)

    prefix_cn = (
        "只对图中【文字】进行编辑；不得改变任何非文字内容（人物、背景、物体、光照、颜色、风格保持不变）。"
        "保持全部未涉及文字的字体/字号/描边/排版/透视一致。"
    )
    prefix_en = (
        "Edit TEXT only; do not change any non-text content (keep people, background, objects, lighting, colors, style unchanged). "
        "Keep all untouched text exactly the same (font/size/stroke/layout/perspective)."
    )

    if kind == "replace":
        spec_cn = "仅将指定文字替换为目标文字；除指定文字外，其他文字一律保持不变。"
        spec_en = "Only replace the specified text with the target text; leave all other text untouched."
    elif kind == "remove":
        spec_cn = "仅移除指定文字，并自然复原其下方的原始背景纹理与光照；不新增任何文字。"
        spec_en = "Only remove the specified text, and naturally restore the original background texture/lighting underneath; do not add any new text."
    elif kind == "add":
        spec_cn = "仅添加所需文字，字体/字号/颜色/描边/透视与周围内容一致；不得修改现有文字与背景。"
        spec_en = "Only add the required text; match surrounding font/size/color/stroke/perspective; do not modify existing text or background."
    else:
        spec_cn = "严格限制在文字编辑范围内。"
        spec_en = "Strictly restrict modifications to text only."

    anchor_cn = f"（若出现“{anchor}”字样，仅对其生效。）" if anchor else ""
    anchor_en = f"(If the token “{anchor}” appears, apply changes only to it.)" if anchor else ""

    prefix = f"{prefix_cn}\n{spec_cn}{anchor_cn}\n" if cn \
             else f"{prefix_en}\n{spec_en} {anchor_en}\n"
    return prefix + prompt


# ----------------------------
# Augmentation & seeds
# ----------------------------
def _maybe_augment(instr: str, augment: Optional[Dict[str, Any]], fallback_negative: str):
    """Apply text-only prefix and choose negative prompt if augmentation enabled."""
    s = str(instr).strip()
    if not augment or not augment.get("enable", False):
        return s, fallback_negative
    neg = str(augment.get("negative_common", "")).strip() or fallback_negative
    pos = _wrap_prompt_for_vttie(s)
    return pos, neg

def _seeds_to_list(seeds: Union[int, List[int], None]) -> List[int]:
    """Normalize seeds to a list."""
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    return [int(x) for x in seeds]


# ----------------------------
# Letterbox utilities
# ----------------------------
def _letterbox_1024(im: Image.Image):
    """Resize with aspect ratio and pad to 1024×1024 (gray background)."""
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
    """Crop padded area and resize back to original resolution."""
    out_cropped = out_sq.crop(crop_box)
    return out_cropped.resize(orig_wh, Image.LANCZOS)


# ----------------------------
# Main batch edit
# ----------------------------
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
    augment: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run batch VTTIE with Qwen-Image edit pipeline.
    Returns a list of metadata dicts for each generated image/candidate.
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
            # Model inference (supports multiple candidates)
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
            # Record error metadata for this item
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
