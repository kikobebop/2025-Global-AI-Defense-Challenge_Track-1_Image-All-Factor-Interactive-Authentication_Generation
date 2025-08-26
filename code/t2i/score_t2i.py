# -*- coding: utf-8 -*-
# ==========================================================
# 概览
# 文生图（T2I）结果评分脚本：
# 1. 读取配置文件（configs/t2i.yaml）的 scoring 区块
# 2. 合并 candidates.csv 与 task.csv，根据 index 补充 prompt
# 3. 调用通用打分模型（Qwen2.5-VL）评估生成图片的语义一致性（SC）和感知质量（PQ）
# 4. 输出评分结果至 scores.csv
#
# Overview
# T2I result scoring script:
# 1. Reads `scoring` block in configs/t2i.yaml
# 2. Merges candidates.csv and task.csv to enrich prompts
# 3. Uses Qwen2.5-VL model to evaluate semantic consistency (SC) and perceptual quality (PQ)
# 4. Outputs results to scores.csv
# ==========================================================

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Ensure the script can run standalone or as part of the package
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent  # .../code
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please run `pip install pyyaml`.", file=sys.stderr)
    raise

# Import shared Qwen-VL scoring functions
from common.qwen_vl_scorer import load_qwen_vl_model, score_one_image


def load_cfg(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ---------------- Argument parsing ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c", "--config",
        required=True,
        help="Path to t2i.yaml (must include scoring section)"
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    scoring = cfg.get("scoring", {})
    if not scoring or not scoring.get("enable", False):
        print("[INFO] scoring.enable = false. Skipping scoring.")
        return

    # ---------------- Model config ----------------
    model_id = scoring.get("model_id")
    if not model_id:
        raise KeyError("Missing required field: scoring.model_id")

    max_new_tokens = int(scoring.get("max_new_tokens", 256))
    do_sample = bool(scoring.get("do_sample", False))

    # ---------------- Input paths ----------------
    input_cfg = scoring.get("input", {})
    run_dir = input_cfg.get("run_dir")  # Preferred: run folder with candidates.csv
    candidates_csv = input_cfg.get("candidates_csv")  # Alternative: direct CSV path

    if not run_dir and not candidates_csv:
        raise KeyError("Specify either run_dir or candidates_csv in scoring.input")

    if run_dir:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        if not candidates_csv:
            candidates_csv = run_dir / "candidates.csv"
        out_csv = scoring.get("out_csv", str(run_dir / "scores.csv"))
    else:
        candidates_csv = Path(candidates_csv)
        if not candidates_csv.exists():
            raise FileNotFoundError(f"candidates_csv does not exist: {candidates_csv}")
        out_csv = scoring.get("out_csv", str(candidates_csv.parent / "scores.csv"))

    # ---------------- Task CSV (for prompt lookup) ----------------
    task_csv = cfg.get("data", {}).get("task_csv") or scoring.get("task_csv")
    if not task_csv or not os.path.exists(task_csv):
        raise FileNotFoundError("Missing data.task_csv (or scoring.task_csv); cannot enrich prompts.")

    # ---------------- Load data ----------------
    df_cand = pd.read_csv(candidates_csv)
    if "image_path" not in df_cand.columns or "index" not in df_cand.columns:
        raise ValueError("candidates.csv must contain 'index' and 'image_path' columns")

    df_task = pd.read_csv(task_csv)[["index", "task_type", "prompt"]]
    df_merged = df_cand.merge(df_task, on="index", how="left")

    # Keep only rows with valid image paths
    df_merged = df_merged[df_merged["image_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))]
    if len(df_merged) == 0:
        print("[WARN] No valid images found in candidates.csv. Scoring skipped.")
        return

    # ---------------- Load scoring model ----------------
    processor, model = load_qwen_vl_model(model_id, torch_dtype="auto", device_map="auto")

    # ---------------- Prepare output file ----------------
    header = ["index", "task_type", "prompt", "result_image", "sc", "pq", "VIEScore", "notes"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=header).to_csv(out_csv, index=False)

    # ---------------- Score each image ----------------
    for r in df_merged.itertuples():
        idx = int(getattr(r, "index"))
        ttype = str(getattr(r, "task_type"))
        prm = str(getattr(r, "prompt"))
        img_path = str(getattr(r, "image_path"))

        try:
            sc, pq, vie, notes = score_one_image(
                img_path, prm, processor, model,
                max_new_tokens=max_new_tokens, do_sample=do_sample
            )
            row = [idx, ttype, prm, img_path, sc, pq, vie, notes]
        except Exception as e:
            # On failure, log the error for this image
            row = [idx, ttype, prm, img_path, None, None, None, f"ERROR: {e}"]

        pd.DataFrame([row], columns=header).to_csv(out_csv, mode="a", index=False, header=False)

    print(f"[OK] Scoring complete -> {out_csv}")


if __name__ == "__main__":
    main()
