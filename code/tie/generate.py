# -*- coding: utf-8 -*-
"""
说明
自然场景图像编辑（TIE）任务 - 主生成脚本
功能：
  1) 解析 YAML 配置文件并加载任务 CSV；
  2) 支持 Qwen-Image-Edit 和 FLUX.1-Kontext-dev 两种编辑后端；
  3) 按索引批量执行编辑任务，输出候选图片及元数据；
  4) 自动生成 config_resolved.json 快照，方便复现与溯源。

约定：
  * seeds 可为单个 int 或 int 列表；候选数大于 seeds 数量时会自动补齐。
  * 输出目录内生成 candidates.csv 和 failures.csv，记录成功与失败的元数据。

Overview
TIE (Text-Instructed Editing) Main Generation Script
Features:
  1) Parse YAML configuration and load the task CSV;
  2) Support two editing backends: Qwen-Image-Edit and FLUX.1-Kontext-dev;
  3) Perform batch editing by index and export candidates and metadata;
  4) Save a `config_resolved.json` snapshot for reproducibility and traceability.

Conventions:
  * Seeds can be a single int or a list; if fewer than candidate count, seeds will be extended incrementally.
  * Outputs `candidates.csv` and `failures.csv` to log success and failure metadata.
"""

import os, sys, csv, argparse, json, time
from pathlib import Path
import pandas as pd

# Ensure script works in both direct and module execution
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception as e:
    print("Missing dependency: PyYAML. Install it with `pip install pyyaml`.", file=sys.stderr)
    raise

from backend_qwen_image_edit import load_qwen_edit, gen_qwen_edit
from backend_flux import load_flux_edit, gen_flux_edit


def load_cfg(path: str) -> dict:
    """Load YAML config and return a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    """Ensure directory exists."""
    Path(p).mkdir(parents=True, exist_ok=True)


def req(d: dict, key: str, typ):
    """
    Get and type-check a required key from a dict.
    """
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    try:
        return typ(d[key])
    except Exception:
        raise TypeError(f"Config key `{key}` must be {typ.__name__}, got: {d[key]!r}")


def dump_snapshot(run_dir: str, cfg_resolved: dict):
    """
    Save a snapshot of resolved config and CLI arguments for reproducibility.
    """
    snap = {
        "argv": sys.argv,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg_resolved
    }
    with open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)


def main():
    # ===== Parse CLI args =====
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="YAML config path (tie.yaml)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # ===== Basic configuration =====
    run_name = cfg.get("run", {}).get("name", "tie_run")
    data_cfg = cfg.get("data", {})
    data_csv = req(data_cfg, "task_csv", str)
    type_filter = data_cfg.get("filter_task_types", ["tie"])
    index_max = int(data_cfg.get("index_max", 10**9))

    out_root = req(cfg.get("output", {}), "dir", str)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{run_name}_{ts}")
    ensure_dir(out_dir)

    pipe_cfg = cfg.get("pipeline", {})
    backend = pipe_cfg.get("backend", "qwen_edit")
    dtype = str(pipe_cfg.get("dtype", "bf16")).lower()
    num_candidates = int(pipe_cfg.get("num_candidates", 1))
    if "seeds" not in pipe_cfg:
        raise KeyError("Missing config key: pipeline.seeds (int or [int,int,...])")
    seeds = pipe_cfg["seeds"]

    # ===== Load task CSV =====
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Task CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)
    need_cols = {"index", "task_type", "prompt", "ori_image"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"Task CSV must contain columns: {need_cols}")

    df_tie = df[df["task_type"].isin(type_filter)]
    df_tie = df_tie[df_tie["index"] < index_max][["index", "prompt", "ori_image"]]

    img_root = str(data_cfg.get("img_root", "")).strip()
    repo_root = PKG_ROOT.parent if PKG_ROOT.parent.exists() else Path.cwd()

    def resolve_img(p: str) -> str:
        """
        Resolve image path using these fallback strategies:
          1. Absolute path if exists;
          2. img_root path if provided;
          3. <repo_root>/data/imgs path;
          4. Relative path from current working directory.
        """
        p = str(p).strip()
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if img_root:
            cand = os.path.join(img_root, p)
            if os.path.exists(cand):
                return cand
        cand2 = os.path.join(str(repo_root), "data", "imgs", p)
        if os.path.exists(cand2):
            return cand2
        if os.path.exists(p):
            return p
        return ""

    df_tie["ori_image_resolved"] = df_tie["ori_image"].map(resolve_img)

    # Filter out invalid paths
    df_bad = df_tie[df_tie["ori_image_resolved"] == ""]
    if len(df_bad) > 0:
        print(f"[WARN] Missing source images: {len(df_bad)} (showing first 5)")
        print(df_bad.head(5).to_string(index=False))

    df_tie = df_tie[df_tie["ori_image_resolved"] != ""]
    items = [(int(r.index), str(r.prompt), str(r.ori_image_resolved)) for r in df_tie.itertuples()]
    if not items:
        print("[WARN] No valid samples found. Check your data paths.")
        return

    # ===== Run backend =====
    if backend == "qwen_edit":
        p = cfg.get("backend_params", {}).get("qwen_edit", {})
        model_path = req(p, "model_path", str)
        steps = req(p, "steps", int)
        true_cfg = req(p, "true_cfg", float)
        negative_prompt = p.get("negative_prompt", "")
        augment = p.get("augment", None)

        cfg_resolved = {
            "run": {"name": run_name, "ts": ts},
            "data": {"task_csv": data_csv, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "qwen_edit", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"qwen_edit": {
                "model_path": model_path, "steps": steps, "true_cfg": true_cfg,
                "negative_prompt": negative_prompt, "augment": augment
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        pipe = load_qwen_edit(model_path, use_bf16=(dtype == "bf16"))
        results = gen_qwen_edit(
            pipe, items, out_dir,
            steps=steps, true_cfg=true_cfg, seeds=seeds,
            negative_prompt=negative_prompt,
            num_candidates=num_candidates,
            augment=augment,
        )

    elif backend == "flux_edit":
        p = cfg.get("backend_params", {}).get("flux_edit", {})
        model_path = req(p, "model_path", str)
        steps = req(p, "steps", int)
        guidance_scale = req(p, "guidance_scale", float)
        augment = p.get("augment", None)
        negative_prompt = p.get("negative_prompt", "")

        cfg_resolved = {
            "run": {"name": run_name, "ts": ts},
            "data": {"task_csv": data_csv, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "flux_edit", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"flux_edit": {
                "model_path": model_path, "steps": steps,
                "guidance_scale": guidance_scale, "augment": augment
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        pipe = load_flux_edit(model_path, use_bf16=(dtype == "bf16"))
        results = gen_flux_edit(
            pipe, items, out_dir,
            steps=steps, guidance_scale=guidance_scale, seeds=seeds,
            num_candidates=num_candidates, augment=augment,
            negative_prompt=negative_prompt,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ===== Write results =====
    cand_csv = os.path.join(out_dir, "candidates.csv")
    fail_csv = os.path.join(out_dir, "failures.csv")

    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if backend == "qwen_edit":
            w.writerow([
                "index", "candidate_id", "backend", "image_path", "src_image",
                "seed_effective", "steps", "true_cfg", "augmented"
            ])
        else:
            w.writerow([
                "index", "candidate_id", "backend", "image_path", "src_image",
                "seed_effective", "steps", "guidance_scale", "augmented"
            ])
        for r in results:
            if "error" in r:
                continue
            m = r["meta"]
            if backend == "qwen_edit":
                w.writerow([
                    r["index"], r.get("candidate_id", "00"), backend, r["path"], r["src_image"],
                    m["seed_effective"], m["steps"], m["true_cfg"], int(m.get("augmented", False))
                ])
            else:
                w.writerow([
                    r["index"], r.get("candidate_id", "00"), backend, r["path"], r["src_image"],
                    m["seed_effective"], m["steps"], m["guidance_scale"], int(m.get("augmented", False))
                ])

    fails = [r for r in results if "error" in r]
    if fails:
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "candidate_id", "backend", "error", "src_image", "steps", "param"])
            for r in fails:
                m = r["meta"]
                param = m.get("true_cfg", m.get("guidance_scale", ""))
                w.writerow([
                    r["index"], r.get("candidate_id", "NA"), backend, r["error"],
                    r.get("src_image", ""), m.get("steps", None), param
                ])

    print(f"[DONE] images -> {out_dir}")
    print(f"[DONE] candidates.csv -> {cand_csv}")
    if fails:
        print(f"[WARN] failures: {len(fails)} -> {fail_csv}")


if __name__ == "__main__":
    main()
