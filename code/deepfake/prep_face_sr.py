#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================================
# Deepfake 预处理：GFPGAN 人脸增强（尺寸不变）
# ----------------------------------------------------------
# 作用：
#   - 利用现有 data.* 配置读取 deepfake 任务（ori/target）
#   - 执行 GFPGAN 轻量清晰化（不更改图像尺寸）
#   - 输出到 preprocess.face_sr.output_dir，并写出新的任务表
#   - 与 pipeline/backend/postprocess 解耦，不污染现有结构
# ==========================================================

import os
import warnings
from pathlib import Path
from typing import Dict, Any

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ---------------- 配置加载 ----------------
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def expand_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()

# ---------------- GFPGAN ----------------
_restorer = None

def init_gfpgan(model_path: Path):
    global _restorer
    try:
        import torch
        from gfpgan import GFPGANer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,              # 保持原尺寸
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        print(f"[INFO] GFPGAN 初始化完成 | device={device} | weight={model_path}")
    except Exception as e:
        raise RuntimeError(f"GFPGAN 初始化失败：{e}")

# ---------------- I/O 与增强 ----------------
def safe_read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取失败：{path}")
    return img

def enhance_face(img_bgr, only_center: bool):
    try:
        _, _, restored_img = _restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=only_center,
            paste_back=True,
        )
        return restored_img if restored_img is not None else img_bgr
    except Exception as e:
        print(f"[WARN] GFPGAN 处理失败，使用原图。原因：{e}")
        return img_bgr

# ---------------- 主流程 ----------------
def run_face_sr(cfg: Dict[str, Any]):
    # 读取 preprocess.face_sr；若未配置或禁用则直接退出
    pre = cfg.get("preprocess", {}).get("face_sr")
    if not pre or not pre.get("enable", False):
        print("[INFO] 预处理 face_sr 未启用或未配置，跳过。")
        return

    data_cfg = cfg["data"]
    out_dir = expand_path(pre["output_dir"])

    gcfg = pre["gfpgan"]
    gfpgan_path   = expand_path(gcfg["model_path"])
    only_center   = bool(gcfg.get("only_center_face", False))
    jpeg_quality  = int(gcfg.get("jpeg_quality", 95))

    task_csv   = expand_path(data_cfg["task_csv"])
    image_root = expand_path(data_cfg["image_root"])

    new_task_csv = out_dir / "task_deepfake_sr.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    init_gfpgan(gfpgan_path)

    # 读取任务
    df = pd.read_csv(task_csv)
    if "filter_task_types" in data_cfg:
        df = df[df["task_type"].isin(data_cfg["filter_task_types"])]
    if "index_min" in data_cfg and "index_max" in data_cfg:
        df = df[(df["index"] >= data_cfg["index_min"]) & (df["index"] < data_cfg["index_max"])]

    need_cols = {"index", "task_type", "ori_image", "target_image"}
    if missing := need_cols - set(df.columns):
        raise ValueError(f"CSV 缺少列：{missing}")

    # 批量处理
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="GFPGAN(face) → face_sr", ncols=100):
        idx = str(row["index"])
        ori_path = image_root / Path(row["ori_image"])
        tgt_path = image_root / Path(row["target_image"])

        ori_img = safe_read(ori_path)
        tgt_img = safe_read(tgt_path)

        ori_sr = enhance_face(ori_img, only_center)
        tgt_sr = enhance_face(tgt_img, only_center)

        ori_name = f"{idx}_ori_sr.jpg"
        tgt_name = f"{idx}_target_sr.jpg"

        cv2.imwrite(str(out_dir / ori_name), ori_sr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        cv2.imwrite(str(out_dir / tgt_name), tgt_sr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        rows.append({
            "index": row["index"],
            "task_type": row["task_type"],
            "ori_image": ori_name,     # 注意：此处相对的是 face_sr 输出目录
            "target_image": tgt_name,
        })

    # 写新任务表
    pd.DataFrame(rows, columns=["index", "task_type", "ori_image", "target_image"]).to_csv(
        new_task_csv, index=False, encoding="utf-8-sig"
    )

    print("\n✅ 完成")
    print(f"- 新图片目录：{out_dir}")
    print(f"- 新任务表：{new_task_csv}")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    cfg = load_config("configs/deepfake.yaml")
    run_face_sr(cfg)
