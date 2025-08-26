# -*- coding: utf-8 -*-
# ==========================================================
# 概览
# Qwen-VL 文生图自动评分工具：
# - 评估生成图像与提示词的语义一致性 (sc) 与感知质量 (pq)
# - 仅依赖 Qwen2.5-VL 模型；输出 (sc, pq, vie, notes)
#
# Overview
# Qwen-VL auto-scoring utility for T2I results:
# - Scores semantic consistency (sc) and perceptual quality (pq)
# - Uses a Qwen2.5-VL model; returns (sc, pq, vie, notes)
# ==========================================================

import json
import re
import torch
from typing import Tuple, Dict, Any
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ------------------------------
# System / instruction message
# ------------------------------
SYSTEM_MSG = (
    "你是一名严谨的图像评测员。只输出一个JSON对象，禁止输出列表([])、代码块(```)、注释或任何多余文字。\n"
    "任务：Text-to-Image 评测，给出两个分数：\n"
    " - sc（0-10）：语义一致性——严格衡量图像与提示词的匹配度，包括主体、属性、数量、动作/姿态、场景、风格等。\n"
    "   * 缺少关键主体或属性 → sc ≤ 3；\n"
    "   * 姿态/动作与提示词矛盾（如要求站立却坐着） → sc ≤ 5；\n"
    "   * 缺少明显细节或风格不匹配（如要求数字化风格却输出写实风格） → 显著扣分。\n"
    " - pq（0-10）：感知质量——衡量画面的清晰度、构图、光影、真实感。\n"
    "   * 人物场景需重点检查：人脸、眼睛、手部、四肢比例是否合理。\n"
    "   * 出现人脸畸形、手指异常、多肢体 → pq ≤ 4。\n"
    "   * 非人物场景则重点关注伪影、锐度、噪点、色彩自然度及几何关系。\n"
    "刻度参考：\n"
    " - 9–10：完全符合提示词，画面自然真实，无明显缺陷。\n"
    " - 7–8：基本符合，仅有轻微偏差（如姿态/细节略有差异或轻微噪点）。\n"
    " - 4–6：多处不符或质量中等（缺少关键细节，人物有明显瑕疵但可辨认）。\n"
    " - 0–3：严重不符或严重缺陷（主体缺失，姿态完全错误，严重畸形）。\n"
    "输出格式（必须严格为一个JSON对象，键名固定且小写）："
    "{\"sc\": <0-10>, \"pq\": <0-10>, \"notes\": \"一句中文理由\"}"
)

# ------------------------------
# Model loading
# ------------------------------
def load_qwen_vl_model(model_id: str, torch_dtype: str = "auto", device_map: str = "auto"):
    """
    Load Qwen2.5-VL processor and model.

    Args:
        model_id: local path or hub id of the model
        torch_dtype: dtype argument for model loading (str or torch dtype as string)
        device_map: device mapping policy ("auto" to place on available GPU/CPU)

    Returns:
        (processor, model)
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map
    ).eval()
    return processor, model

# ------------------------------
# Helpers
# ------------------------------
def _clamp_0_10(x):
    """Clamp a value into [0, 10]; returns None for invalid input."""
    try:
        v = float(x)
    except Exception:
        return None
    return max(0.0, min(10.0, v))

def _parse_scores_from_text(out_text: str) -> Dict[str, Any]:
    """
    Parse model output into scores.
    Priority: JSON parse → regex fallback.

    Returns:
        dict with keys: sc, pq, notes
    """
    candidates = []

    # Try extracting content inside triple backticks
    if "```" in out_text:
        parts = out_text.split("```")
        for i in range(1, len(parts), 2):
            candidates.append(parts[i].strip())

    # Try extracting the largest {...} JSON segment
    l, r = out_text.find("{"), out_text.rfind("}")
    if l != -1 and r != -1 and r > l:
        candidates.append(out_text[l:r + 1].strip())

    # Also consider the entire text as a candidate
    candidates.append(out_text.strip())

    # Attempt JSON parsing
    for c in candidates:
        try:
            obj = json.loads(c)
            # Handle list-wrapped dicts
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

# Prompt template for the user turn
USER_TEMPLATE = (
    "原始Prompt：{prompt}\n\n"
    "请仅返回一个JSON对象，键名固定：sc, pq, notes。不要输出任何多余内容。"
)

# ------------------------------
# Main scoring function
# ------------------------------
def score_one_image(
    image_path: str,
    prompt_text: str,
    processor,
    model,
    max_new_tokens: int = 256,
    do_sample: bool = False
) -> Tuple[float, float, float, str]:
    """
    Score a generated image with Qwen2.5-VL.

    Args:
        image_path: path to the generated image
        prompt_text: original text prompt used for generation
        processor: loaded AutoProcessor
        model: loaded Qwen2.5-VL model
        max_new_tokens: decoding length cap
        do_sample: enable sampling for generation (False ensures determinism)

    Returns:
        (sc, pq, vie, notes)
        - sc: semantic consistency [0,10]
        - pq: perceptual quality [0,10]
        - vie: combined score = sqrt(sc * pq)
        - notes: short Chinese rationale
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": USER_TEMPLATE.format(prompt=prompt_text)},
        ]},
    ]

    # Build inputs for the VL model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

    # Strip the prompt part, keep only newly generated tokens
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse scores
    data = _parse_scores_from_text(out_text)
    sc = _clamp_0_10(data.get("sc")) or 0.0
    pq = _clamp_0_10(data.get("pq")) or 0.0
    vie = (sc * pq) ** 0.5

    return round(sc, 2), round(pq, 2), round(vie, 4), str(data.get("notes", ""))
