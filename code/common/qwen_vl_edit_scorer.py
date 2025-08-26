# -*- coding: utf-8 -*-
"""
自然场景图像编辑任务（TIE）与视觉文字编辑任务（VTTIE）自动化评分工具
支持基于 Qwen2.5-VL 模型的自动评分，输出语义一致性（SC）、感知质量（PQ）以及综合评分（VIE）。
可用于批量化分析生成或编辑结果质量。

Automated scoring utility for:
- TIE (Text-Instructed Image Editing)
- VTTIE (Vision-Text-To-Image Editing)

Features:
- Uses Qwen2.5-VL model for semantic consistency (SC), perceptual quality (PQ), and combined score (VIE)
- Produces structured results for downstream analysis and quality control
"""

import json
import re
import torch
from typing import Tuple, Dict, Any
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------
def load_qwen_vl_model(model_id: str, torch_dtype: str = "auto", device_map: str = "auto"):
    """
    Load Qwen2.5-VL processor and model.

    Args:
        model_id: Local path or model hub id.
        torch_dtype: Precision (default: "auto").
        device_map: Device mapping (default: "auto").

    Returns:
        (processor, model)
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map
    ).eval()
    return processor, model

# ------------------------------------------------------------------------
# Scoring prompt templates (Chinese content for better instruction-following)
# ------------------------------------------------------------------------
SYSTEM_MSG = (
    "你是一名严谨的图像编辑评测员。只输出一个JSON对象，禁止输出列表([])、代码块(```)、注释或任何多余文字。\n"
    "任务：基于“原图 + 编辑后图 + 提示词（编辑指令）”，给出两个分数：\n"
    " - sc（0-10）：编辑语义一致性——编辑后图是否正确实现提示词要求，同时保留原图应保留的主体/布局/身份/关键元素；"
    "   对于 vttie（视觉文字编辑），还需严格检查文字内容是否与提示词一致（文字内容、位置、样式）；\n"
    " - pq（0-10）：编辑后图的感知质量——清晰度、边缘/融合自然度、光影/色彩一致性、是否有伪影或错位；人物需检查脸/手/肢体比例等；\n"
    "扣分参考：\n"
    " - 关键编辑未生效或与指令冲突 → sc ≤ 3；\n"
    " - vttie 中文字内容/位置/样式错误或拼写错误 → sc ≤ 4；\n"
    " - 融合不自然、明显接缝/多余物体、几何/光照严重异常 → pq ≤ 4；\n"
    "刻度参考：9–10 接近完美；7–8 轻微问题；4–6 多处可见问题；0–3 严重不符或质量极差。\n"
    "最后需给出简短中文理由，并注明扣分点（括号内注明针对 sc 或 pq）。\n"
    "输出格式（严格为一个JSON对象，键名固定且小写）："
    "{\"sc\": <0-10>, \"pq\": <0-10>, \"notes\": \"一句中文理由（简短）\"}"
)

USER_TEMPLATE = (
    "这是图像编辑评测：\n"
    "原图（第一张）：用于对照应保留的主体/布局/身份/场景。\n"
    "编辑后图（第二张）：应实现以下编辑指令。\n\n"
    "编辑提示词（指令）：{prompt}\n\n"
    "请仅返回一个JSON对象，键名固定：sc, pq, notes。不要输出任何多余内容。"
)

# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------
def _clamp_0_10(x):
    """Clamp numeric value to [0, 10]; return None on invalid input."""
    try:
        v = float(x)
    except Exception:
        return None
    return max(0.0, min(10.0, v))

def _parse_scores_from_text(out_text: str) -> Dict[str, Any]:
    """
    Parse model output into a dict with keys: sc, pq, notes.

    Strategy:
      1) Try to parse JSON (including fenced blocks and curly-brace spans)
      2) Fallback to regex extraction for sc/pq if JSON parsing fails
    """
    candidates = []

    # Extract fenced code blocks
    if "```" in out_text:
        parts = out_text.split("```")
        for i in range(1, len(parts), 2):
            candidates.append(parts[i].strip())

    # Extract substring bounded by the outermost curly braces
    l, r = out_text.find("{"), out_text.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidates.append(out_text[l:r+1].strip())

    candidates.append(out_text.strip())

    # Attempt JSON decoding
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                obj = obj[0]
            if isinstance(obj, dict):
                sc = _clamp_0_10(obj.get("sc"))
                pq = _clamp_0_10(obj.get("pq"))
                notes = obj.get("notes", obj.get("explain", ""))
                if sc is not None and pq is not None:
                    return {"sc": sc, "pq": pq, "notes": str(notes)}
        except Exception:
            pass

    # Regex fallback
    m_sc = re.search(r'"?sc"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', out_text, re.I)
    m_pq = re.search(r'"?pq"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', out_text, re.I)
    sc = _clamp_0_10(m_sc.group(1)) if m_sc else 0.0
    pq = _clamp_0_10(m_pq.group(1)) if m_pq else 0.0
    return {"sc": sc, "pq": pq, "notes": "PARSE_FALLBACK"}

# ------------------------------------------------------------------------
# Main scoring function
# ------------------------------------------------------------------------
def score_one_edit(
    ori_image_path: str,
    edited_image_path: str,
    prompt_text: str,
    processor,
    model,
    max_new_tokens: int = 256,
    do_sample: bool = False
) -> Tuple[float, float, float, str]:
    """
    Score one edited image given original image, edited image, and prompt.

    Returns:
        (sc, pq, vie, notes)
          - sc: semantic consistency score
          - pq: perceptual quality score
          - vie: combined score = sqrt(sc * pq)
          - notes: short explanation (Chinese)
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{ori_image_path}"},
            {"type": "image", "image": f"file://{edited_image_path}"},
            {"type": "text",  "text": USER_TEMPLATE.format(prompt=prompt_text)},
        ]},
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Inference
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

    # Decode output text (strip prompt tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Parse to scores
    data = _parse_scores_from_text(out_text)
    sc = _clamp_0_10(data.get("sc")) or 0.0
    pq = _clamp_0_10(data.get("pq")) or 0.0
    vie = (sc * pq) ** 0.5
    return round(sc, 2), round(pq, 2), round(vie, 4), str(data.get("notes", ""))
