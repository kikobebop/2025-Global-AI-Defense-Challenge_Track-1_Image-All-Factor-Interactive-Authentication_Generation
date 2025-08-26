# -*- coding: utf-8 -*-
# ==========================================================
# 概述
# T2I 结果批量超分（目录版）
# - 从 YAML 的 postprocess.esrgan.input.dir 读取待处理目录
# - 仅处理文件名形如 index_*.jpg/png/... 的图片
# - 先按 scale 用 RealESRGAN 超分
# - 若 respect_prompt=true 且对应 prompt 含 4k/8k，再将长边对齐到 3840/7680
# - 结果保存在同一目录，文件名追加后缀：_sr2x/_sr4x/_sr2x_4k 等
#
# Overview
# Batch super-resolution for T2I results
# - Read input directory from YAML: postprocess.esrgan.input.dir
# - Only process image files with names like: index_*.jpg/png/...
# - First apply RealESRGAN with the given `scale`
# - If respect_prompt=true and the prompt contains 4k/8k, align the long edge
#   to 3840/7680 after SR
# - Save outputs in-place with suffixes: _sr2x/_sr4x/_sr2x_4k, etc.
# ==========================================================

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import yaml
import unicodedata

# Make package imports work both as script and module
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent  # .../code
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from common.backend_superres import ESRGANUpsampler, resize_to_long_edge

# Supported image extensions
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

# Robust 4K/8K detectors (friendly to mixed/Chinese text, avoid \b)
PAT_4K = re.compile(r"(?<![0-9A-Za-z])4\s*[kK](?![0-9A-Za-z])", re.IGNORECASE)
PAT_8K = re.compile(r"(?<![0-9A-Za-z])8\s*[kK](?![0-9A-Za-z])", re.IGNORECASE)


def _norm(s: str) -> str:
    """Normalize to NFKC to handle full-width characters like '４Ｋ'."""
    return unicodedata.normalize("NFKC", s or "")


def load_yaml(p: str | Path) -> dict:
    """Load YAML config."""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image_bgr(path: Path) -> np.ndarray:
    """Read image as BGR (H, W, 3), uint8, respecting EXIF orientation."""
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return np.array(img)[:, :, ::-1].copy()


def parse_prompt_target(prompt: str, long4k: int, long8k: int) -> Optional[Tuple[str, int]]:
    """
    Inspect prompt and decide target long edge:
      - returns ("8k", long8k) if prompt contains 8k
      - returns ("4k", long4k) if prompt contains 4k
      - else returns None
    """
    p = _norm(prompt or "")
    if PAT_8K.search(p):
        return "8k", long8k
    if PAT_4K.search(p):
        return "4k", long4k
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to configs/t2i.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    esr = (cfg.get("postprocess", {}) or {}).get("esrgan", {}) or {}
    if not esr.get("enable", False):
        print("[INFO] postprocess.esrgan.enable = false. Skip SR.")
        return

    # --- Core options ---
    scale = float(esr.get("scale", 2))
    respect_prompt = bool(esr.get("respect_prompt", False))
    long4k = int(esr.get("target_long_4k", 3840))
    long8k = int(esr.get("target_long_8k", 7680))

    weights = esr.get("weights")
    if not weights:
        raise ValueError("Please set postprocess.esrgan.weights to RealESRGAN_x4plus .pth")

    tile = int(esr.get("tile", 0))
    tile_pad = int(esr.get("tile_pad", 10))
    use_half = esr.get("use_half", None)

    # --- Input directory ---
    in_cfg = esr.get("input", {})
    in_dir = in_cfg.get("dir", "")
    if not in_dir:
        raise ValueError("Please specify postprocess.esrgan.input.dir as the folder to process.")
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    # Build mapping: index -> prompt (only if respect_prompt)
    prompt_map = {}
    if respect_prompt:
        task_csv = Path(cfg["data"]["task_csv"])
        if not task_csv.exists():
            raise FileNotFoundError(f"task_csv not found: {task_csv}")
        df_task = pd.read_csv(task_csv)
        if not {"index", "prompt"}.issubset(df_task.columns):
            raise ValueError("task.csv must contain columns: index, prompt")
        for r in df_task.itertuples():
            idx = str(int(getattr(r, "index")))
            prompt_map[idx] = str(getattr(r, "prompt") or "")

    # Init ESRGAN backend
    upsampler = ESRGANUpsampler(
        weights_path=weights,
        tile=tile,
        tile_pad=tile_pad,
        use_half=use_half
    )

    # Scan files named like "1234_*.jpg/png/..."
    print(f"[INFO] Scanning directory: {in_dir}")
    files = [
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS and re.match(r"^\d+_", p.name)
    ]
    if not files:
        print("[WARN] No files like 'index_*.jpg/png' found.")
        return

    ok = fail = 0
    for img_path in tqdm(files, desc="SR", ncols=100):
        try:
            m = re.match(r"^(\d+)_", img_path.name)
            if not m:
                continue
            idx = m.group(1)

            img0 = load_image_bgr(img_path)

            # Step 1: RealESRGAN upscaling with `scale`
            out = upsampler.enhance(img0, outscale=scale)
            if abs(scale - round(scale)) < 1e-6:
                tag = f"sr{int(round(scale))}x"
            else:
                tag = f"sr{scale}x"

            # Step 2: Optional 4K/8K long-edge alignment based on prompt
            if respect_prompt:
                prompt = prompt_map.get(idx, "")
                tgt = parse_prompt_target(prompt, long4k, long8k)
                if tgt is not None:
                    label, tgt_long = tgt
                    out = resize_to_long_edge(out, tgt_long)
                    tag = f"{tag}_{label}"

            # Save in-place with suffix
            dst = img_path.with_name(f"{img_path.stem}_{tag}{img_path.suffix.lower()}")
            if dst.suffix in (".jpg", ".jpeg"):
                cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(dst), out)

            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERR] {img_path.name}: {e}")

    print(f"[DONE] ok={ok}, fail={fail}, dir={in_dir}")


if __name__ == "__main__":
    main()
