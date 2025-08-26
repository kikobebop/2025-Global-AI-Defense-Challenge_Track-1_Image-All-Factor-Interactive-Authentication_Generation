# ==========================================================
# 视觉文字编辑（VTTIE）任务 - 自动化评分脚本
# 功能（中文）：
#   1) 读取 vttie.yaml 中 scoring 配置
#   2) 合并 candidates.csv 与 task.csv，补齐 prompt 和 task_type 信息
#   3) 使用 Qwen2.5-VL 模型对原图与编辑后图进行自动化评分
#   4) 输出 scores.csv 文件，包含语义一致性、感知质量和综合得分
#
# VTTIE (Visual Text-in-Image Editing) — Scoring Script
# Features (English):
#   1) Load scoring section from vttie.yaml
#   2) Merge candidates.csv with task.csv to supplement prompt and task_type
#   3) Evaluate original and edited images using Qwen2.5-VL model
#   4) Output scores.csv with semantic consistency, perceptual quality, and overall score
#
# 快速使用（Quick Start）：
#   中文：
#     1) 确保 vttie.yaml 配置了 scoring.input.run_dir 或 candidates_csv；
#     2) 运行：python code/vttie/score_vttie.py -c configs/vttie.yaml
#   English：
#     1) Ensure scoring.input.run_dir or candidates_csv is set in vttie.yaml;
#     2) Run: python code/vttie/score_vttie.py -c configs/vttie.yaml
# ==========================================================

import os, sys, argparse
from pathlib import Path
import pandas as pd

# Support both direct run and module import
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent  # .../code
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please install via `pip install pyyaml`", file=sys.stderr)
    raise

from common.qwen_vl_edit_scorer import load_qwen_vl_model, score_one_edit


def load_cfg(p: str):
    """Load YAML config as dict."""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to vttie.yaml (with scoring section)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    scoring = cfg.get("scoring", {})
    if not scoring or not scoring.get("enable", False):
        print("[INFO] scoring.enable=false, skip scoring.")
        return

    model_id = scoring.get("model_id")
    if not model_id:
        raise KeyError("Missing required: scoring.model_id")

    max_new_tokens = int(scoring.get("max_new_tokens", 256))
    do_sample = bool(scoring.get("do_sample", False))

    # -------------------------------
    # Locate input files
    # -------------------------------
    input_cfg = scoring.get("input", {})
    run_dir = input_cfg.get("run_dir")
    candidates_csv = input_cfg.get("candidates_csv")
    if not run_dir and not candidates_csv:
        raise KeyError("Provide either run_dir or candidates_csv in scoring.input.")

    if run_dir:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        if not candidates_csv:
            candidates_csv = run_dir / "candidates.csv"
        out_csv = scoring.get("out_csv", str(run_dir / "scores.csv"))
    else:
        candidates_csv = Path(candidates_csv)
        if not candidates_csv.exists():
            raise FileNotFoundError(f"candidates_csv not found: {candidates_csv}")
        out_csv = scoring.get("out_csv", str(candidates_csv.parent / "scores.csv"))

    # -------------------------------
    # Load task data and merge
    # -------------------------------
    task_csv = scoring.get("task_csv") or cfg.get("data", {}).get("task_csv")
    if not task_csv or not os.path.exists(task_csv):
        raise FileNotFoundError("Missing data.task_csv (or scoring.task_csv) for prompt/task_type merge.")

    df_cand = pd.read_csv(candidates_csv)
    required_cols = {"index", "image_path", "src_image"}
    if not required_cols.issubset(set(df_cand.columns)):
        raise ValueError(f"candidates.csv must include: {required_cols}")

    # Filter invalid paths
    df_cand = df_cand[df_cand["image_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))]
    df_cand = df_cand[df_cand["src_image"].map(lambda p: isinstance(p, str) and os.path.exists(p))]
    if len(df_cand) == 0:
        print("[WARN] No valid images in candidates, exiting.")
        return

    df_task = pd.read_csv(task_csv)[["index", "task_type", "prompt"]]
    df_merged = df_cand.merge(df_task, on="index", how="left")

    # -------------------------------
    # Load Qwen2.5-VL model
    # -------------------------------
    processor, model = load_qwen_vl_model(model_id, torch_dtype="auto", device_map="auto")

    # -------------------------------
    # Initialize output file
    # -------------------------------
    header = [
        "index", "task_type", "prompt",
        "ori_image", "result_dir", "result_image",
        "sc", "pq", "VIEScore", "notes"
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=header).to_csv(out_csv, index=False)

    # -------------------------------
    # Process rows and score
    # -------------------------------
    for r in df_merged.itertuples():
        idx = int(getattr(r, "index"))
        ttype = str(getattr(r, "task_type"))
        prm = str(getattr(r, "prompt"))
        edited_path = str(getattr(r, "image_path"))
        ori_path = str(getattr(r, "src_image"))
        rdir = str(Path(edited_path).parent)

        try:
            sc, pq, vie, notes = score_one_edit(
                ori_path, edited_path, prm,
                processor, model,
                max_new_tokens=max_new_tokens, do_sample=do_sample
            )
            row = [idx, ttype, prm, ori_path, rdir, edited_path, sc, pq, vie, notes]
        except Exception as e:
            row = [idx, ttype, prm, ori_path, rdir, edited_path, None, None, None, f"ERROR: {e}"]

        pd.DataFrame([row], columns=header).to_csv(out_csv, mode="a", index=False, header=False)

    print(f"[OK] Scoring completed -> {out_csv}")


if __name__ == "__main__":
    main()
