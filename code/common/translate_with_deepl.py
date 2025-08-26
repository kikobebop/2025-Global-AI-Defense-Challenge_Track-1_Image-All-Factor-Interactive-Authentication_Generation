# tools/translate_prompts_deepl.py
# -*- coding: utf-8 -*-
"""
简介
----------------------------------------------------------------
功能：批量将任务表 (data/task.csv) 中的中文 prompt 翻译为英文，并输出：
  1) 翻译后的任务表 data/task_en.csv（用于后续流程直接替换 prompt 列）
  2) 原始与英文 prompt 的对应表 data/prompt_map.csv
  3) 翻译缓存 data/translate_cache.json（避免重复调用 API）

特点：
  - 仅对包含中文字符的 prompt 调用 DeepL，其余原样保留
  - 带断点缓存与指数退避重试，尽量稳健
  - 列名假定为：index, task_type, prompt, ori_image, target_image

准备：
  - 需要先设置环境变量：export DEEPL_API_KEY="你的_deepl_api_key"

用法：
  python tools/translate_prompts_deepl.py
  （可按需修改脚本顶部的路径常量）

Overview
----------------------------------------------------------------
Purpose: Translate Chinese prompts in `data/task.csv` to English in batch,
and produce:
  1) `data/task_en.csv` – same schema as input with `prompt` replaced by English
  2) `data/prompt_map.csv` – mapping table: original prompt → English
  3) `data/translate_cache.json` – local cache to avoid redundant API calls

Highlights:
  - Only translate rows whose prompt contains Chinese characters
  - Resilient with local cache and exponential backoff retries
  - Assumes columns: index, task_type, prompt, ori_image, target_image

Prerequisite:
  - Set environment variable: export DEEPL_API_KEY="YOUR_DEEPL_API_KEY"

Usage:
  python tools/translate_prompts_deepl.py
  (Adjust the path constants at the top if needed.)
"""

import os
import re
import time
import json
from typing import List, Dict
from pathlib import Path

import pandas as pd
import deepl

# ========= Paths (adjust as needed) =========
TASKS_CSV    = "./data/task.csv"                 # Input tasks CSV (Chinese prompts)
OUT_TASK_EN  = "./data/task_en.csv"              # Output tasks CSV (prompts translated to English)
OUT_MAP      = "./data/prompt_map.csv"           # Mapping: original prompt → English
CACHE_PATH   = "./data/translate_cache.json"     # Local translation cache

# ========= DeepL settings =========
# Make sure DEEPL_API_KEY is set in the environment, e.g.:
#   export DEEPL_API_KEY="your_deepl_api_key"
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY")
TARGET_LANG   = "EN-US"   # or "EN-GB"
BATCH_SIZE    = 50        # batch size per API call
SLEEP_SEC     = 0.5       # pause between batches to avoid rate limits
MAX_RETRY     = 5         # max retries per batch on transient errors

# ========= Required columns =========
COLS = ["index", "task_type", "prompt", "ori_image", "target_image"]
PROMPT_COL = "prompt"

# ========= Simple Chinese detector =========
zh_re = re.compile(r"[\u4e00-\u9fff]")  # CJK Unified Ideographs


def has_zh(s: str) -> bool:
    """Return True if the string contains any Chinese character."""
    return isinstance(s, str) and bool(zh_re.search(s))


def load_cache(path: str) -> Dict[str, str]:
    """
    Load translation cache from JSON file.
    Returns an empty dict if the cache is missing or corrupted.
    """
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, str], path: str):
    """Persist translation cache to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_batch(translator: deepl.Translator, batch: List[str]) -> List[str]:
    """
    Translate a batch of texts via DeepL with retry and backoff.

    Args:
        translator: deepl.Translator instance
        batch: list of strings to translate

    Returns:
        list of translated strings in the same order
    """
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = translator.translate_text(batch, target_lang=TARGET_LANG)
            if isinstance(res, list):
                return [r.text for r in res]
            else:
                return [res.text]
        except deepl.exceptions.QuotaExceededException as e:
            # Hard stop: quota exhausted
            raise RuntimeError(f"DeepL quota exceeded: {e}")
        except deepl.exceptions.TooManyRequestsException:
            # Backoff and retry
            time.sleep(min(2 ** attempt, 10))
        except Exception:
            # Backoff and retry for other transient errors
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("DeepL translation failed after retries")


def main():
    # Ensure API key
    assert DEEPL_API_KEY, "Environment variable DEEPL_API_KEY is not set. Please `export DEEPL_API_KEY=...` first."

    # Load tasks CSV
    df = pd.read_csv(TASKS_CSV)
    missing = [c for c in COLS if c not in df.columns]
    assert not missing, f"CSV is missing columns: {missing}; expected columns: {COLS}"

    # Normalize prompt column to string
    df[PROMPT_COL] = df[PROMPT_COL].fillna("").astype(str)
    prompts = df[PROMPT_COL].tolist()

    # Find rows that actually need translation (contain Chinese)
    need_idx = [i for i, t in enumerate(prompts) if has_zh(t)]

    # If nothing to translate, copy input to output and create an empty mapping
    if not need_idx:
        Path(OUT_TASK_EN).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_TASK_EN, index=False)
        pd.DataFrame(columns=["index", "task_type", "prompt", "prompt_en"]).to_csv(OUT_MAP, index=False)
        print(f"[OK] No Chinese prompts. Saved as-is -> {OUT_TASK_EN}; empty map -> {OUT_MAP}")
        return

    # Init translator and cache
    translator = deepl.Translator(DEEPL_API_KEY)
    cache = load_cache(CACHE_PATH)

    outputs = prompts[:]   # translated prompts will be written back here
    map_rows = []          # rows for mapping CSV

    # Use cache first
    pending_pairs = []
    for i in need_idx:
        zh = prompts[i]
        if zh in cache:
            en = cache[zh]
            outputs[i] = en
            map_rows.append({
                "index": df.at[i, "index"],
                "task_type": df.at[i, "task_type"],
                "prompt": zh,
                "prompt_en": en
            })
        else:
            pending_pairs.append((i, zh))

    # Translate the rest in batches
    for s in range(0, len(pending_pairs), BATCH_SIZE):
        batch = pending_pairs[s:s + BATCH_SIZE]
        batch_texts = [t for _, t in batch]
        en_list = translate_batch(translator, batch_texts)

        for (i, zh), en in zip(batch, en_list):
            outputs[i] = en
            cache[zh] = en
            map_rows.append({
                "index": df.at[i, "index"],
                "task_type": df.at[i, "task_type"],
                "prompt": zh,
                "prompt_en": en
            })

        # Lightweight throttling and periodic cache flush
        time.sleep(SLEEP_SEC)
        save_cache(cache, CACHE_PATH)

    # Write translated prompts back
    df[PROMPT_COL] = outputs

    # Save outputs
    Path(OUT_TASK_EN).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TASK_EN, index=False)

    map_df = pd.DataFrame(map_rows, columns=["index", "task_type", "prompt", "prompt_en"])
    map_df.to_csv(OUT_MAP, index=False)

    print(f"[OK] saved:\n  task_en -> {OUT_TASK_EN}\n  prompt_map -> {OUT_MAP}\n  cache -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
