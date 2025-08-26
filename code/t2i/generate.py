# -*- coding: utf-8 -*-
# ==========================================================
# 概览
# 文生图（T2I）主生成脚本，支持 Qwen-Image 与 SDXL 两种后端；
# 通过单一 YAML 配置驱动统一流程，输出图像与 candidates/failures CSV。
#
# Overview
# Text-to-Image (T2I) main runner supporting Qwen-Image and SDXL.
# Driven by one YAML config; saves images plus candidates/failures CSVs.
# ==========================================================

import os
import sys
import csv
import argparse
import json
import time
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML. Please run `pip install pyyaml`.", file=sys.stderr)
    raise

from backend_qwen_image import load_qwen_image, gen_qwen_image
from backend_sdxl import load_sdxl, gen_sdxl


# ----------------------------------------------------------
# Common aspect ratios -> default (width, height)
# (Used when width/height are not explicitly provided)
# ----------------------------------------------------------
AR_SIZES = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def load_cfg(path: str) -> dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    """Create directory if it does not exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def req(d: dict, key: str, typ):
    """Fetch a required config field and validate its type."""
    if key not in d:
        raise KeyError(f"Missing config field: {key}")
    try:
        return typ(d[key])
    except Exception:
        raise TypeError(f"`{key}` must be {typ.__name__}, got: {d[key]!r}")


def dump_snapshot(run_dir: str, cfg_resolved: dict):
    """Save a JSON snapshot of the current run (args + resolved config)."""
    snap = {
        "argv": sys.argv,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg_resolved,
    }
    with open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    # ---------------- Arg parsing ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # ---------------- Base config ----------------
    run_name = cfg.get("run", {}).get("name", "t2i_run")
    data_cfg = cfg.get("data", {})
    data_csv = req(data_cfg, "task_csv", str)
    type_filter = data_cfg.get("filter_task_types", ["t2i"])
    index_max = int(data_cfg.get("index_max", 10**9))

    out_cfg = cfg.get("output", {})
    out_root = req(out_cfg, "dir", str)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{run_name}_{ts}")
    ensure_dir(out_dir)

    pipe_cfg = cfg.get("pipeline", {})
    backend = pipe_cfg.get("backend", "qwen_image")
    dtype = str(pipe_cfg.get("dtype", "bf16")).lower()
    num_candidates = int(pipe_cfg.get("num_candidates", 1))

    # seeds are required for reproducibility
    if "seeds" not in pipe_cfg:
        raise KeyError("Missing config field: pipeline.seeds (int or [int, int, ...])")
    seeds = pipe_cfg["seeds"]

    # ---------------- Load data ----------------
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Task CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)
    df_t2i = df[df["task_type"].isin(type_filter)]
    df_t2i = df_t2i[df_t2i["index"] < index_max][["index", "prompt"]]
    items = [(int(r.index), str(r.prompt)) for r in df_t2i.itertuples()]

    # ---------------- Dispatch by backend ----------------
    if backend == "qwen_image":
        # Qwen-Image
        qwen_params = cfg.get("backend_params", {}).get("qwen_image", {})
        model_path = req(qwen_params, "model_path", str)

        # Resolve size
        width = qwen_params.get("width")
        height = qwen_params.get("height")
        if width is not None and height is not None:
            width, height = int(width), int(height)
            ar = f"{width}x{height}"
        else:
            ar = req(qwen_params, "aspect_ratio", str)
            if ar not in AR_SIZES:
                raise ValueError(f"Unknown aspect_ratio: {ar}. Choices: {', '.join(AR_SIZES.keys())}")
            width, height = AR_SIZES[ar]

        steps = req(qwen_params, "steps", int)
        true_cfg = req(qwen_params, "true_cfg", float)
        negative_prompt = qwen_params.get("negative_prompt", "")
        augment = qwen_params.get("augment", None)

        # Snapshot
        cfg_resolved = {
            "run": {"name": run_name, "ts": ts},
            "data": {"task_csv": data_csv, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "qwen_image", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"qwen_image": {
                "model_path": model_path, "aspect_ratio": ar,
                "width": width, "height": height, "steps": steps,
                "true_cfg": true_cfg, "negative_prompt": negative_prompt,
                "augment": augment,
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        if not os.path.exists(model_path):
            print(f"[WARN] Model path does not exist: {model_path}")
        use_bf16 = (dtype == "bf16")
        pipe = load_qwen_image(model_path, use_bf16=use_bf16)

        results = gen_qwen_image(
            pipe, items, out_dir,
            width=width, height=height, steps=steps, true_cfg=true_cfg,
            seeds=seeds,
            negative_prompt=negative_prompt,
            num_candidates=num_candidates,
            augment=augment,
        )

    elif backend == "sdxl":
        # SDXL
        sdxl_params = cfg.get("backend_params", {}).get("sdxl", {})
        base_path = req(sdxl_params, "model_path_base", str)
        refiner_path = sdxl_params.get("model_path_refiner")
        use_refiner = bool(sdxl_params.get("use_refiner", True))
        augment = sdxl_params.get("augment", None)

        # Resolve size
        width = sdxl_params.get("width")
        height = sdxl_params.get("height")
        if width is not None and height is not None:
            width, height = int(width), int(height)
            ar = f"{width}x{height}"
        else:
            ar = req(sdxl_params, "aspect_ratio", str)
            if ar not in AR_SIZES:
                raise ValueError(f"Unknown aspect_ratio: {ar}. Choices: {', '.join(AR_SIZES.keys())}")
            width, height = AR_SIZES[ar]

        total_steps = req(sdxl_params, "total_steps", int)
        guidance_scale = req(sdxl_params, "guidance_scale", float)
        high_noise_frac = req(sdxl_params, "high_noise_frac", float)

        # Snapshot
        cfg_resolved = {
            "run": {"name": run_name, "ts": ts},
            "data": {"task_csv": data_csv, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "sdxl", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"sdxl": {
                "model_path_base": base_path,
                "model_path_refiner": refiner_path,
                "use_refiner": use_refiner,
                "aspect_ratio": ar, "width": width, "height": height,
                "total_steps": total_steps, "guidance_scale": guidance_scale,
                "high_noise_frac": high_noise_frac,
                "augment": augment,
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        use_bf16 = (dtype == "bf16")
        base, refiner = load_sdxl(
            base_path,
            refiner_path=refiner_path,
            use_bf16=use_bf16,
            use_refiner=use_refiner,
        )

        results = gen_sdxl(
            base, refiner, items, out_dir,
            width=width, height=height,
            total_steps=total_steps, guidance_scale=guidance_scale,
            high_noise_frac=high_noise_frac,
            seeds=seeds,
            use_refiner=use_refiner,
            num_candidates=num_candidates,
            augment=augment,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # ---------------- Persist results ----------------
    cand_csv = os.path.join(out_dir, "candidates.csv")
    fail_csv = os.path.join(out_dir, "failures.csv")

    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if backend == "qwen_image":
            w.writerow([
                "index", "candidate_id", "backend", "image_path",
                "seed_effective", "width", "height", "steps", "true_cfg", "aspect_ratio", "augmented",
            ])
        else:
            w.writerow([
                "index", "candidate_id", "backend", "image_path",
                "seed_effective", "width", "height",
                "total_steps", "guidance_scale", "high_noise_frac", "use_refiner", "aspect_ratio", "augmented",
            ])

        for r in results:
            if "error" in r:
                continue
            m = r["meta"]
            if backend == "qwen_image":
                w.writerow([
                    r["index"], r.get("candidate_id", "00"), backend, r["path"],
                    m["seed_effective"], m["width"], m["height"], m["steps"], m["true_cfg"],
                    cfg_resolved["backend_params"]["qwen_image"]["aspect_ratio"],
                    int(m.get("augmented", False)),
                ])
            else:
                w.writerow([
                    r["index"], r.get("candidate_id", "00"), backend, r["path"],
                    m["seed_effective"], m["width"], m["height"],
                    m["total_steps"], m["guidance_scale"], m["high_noise_frac"], m["use_refiner"],
                    cfg_resolved["backend_params"]["sdxl"]["aspect_ratio"],
                    int(m.get("augmented", False)),
                ])

    # Write failures (if any)
    fails = [r for r in results if "error" in r]
    if fails:
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if backend == "qwen_image":
                w.writerow([
                    "index", "candidate_id", "backend", "error",
                    "width", "height", "steps", "true_cfg", "seed_effective", "augmented",
                ])
            else:
                w.writerow([
                    "index", "candidate_id", "backend", "error",
                    "width", "height", "total_steps", "guidance_scale", "high_noise_frac", "use_refiner",
                    "seed_effective", "augmented",
                ])
            for r in fails:
                m = r["meta"]
                if backend == "qwen_image":
                    w.writerow([
                        r["index"], r.get("candidate_id", "NA"), backend, r["error"],
                        m["width"], m["height"], m["steps"], m["true_cfg"],
                        m.get("seed_effective", ""), int(m.get("augmented", False)),
                    ])
                else:
                    w.writerow([
                        r["index"], r.get("candidate_id", "NA"), backend, r["error"],
                        m["width"], m["height"], m["total_steps"], m["guidance_scale"], m["high_noise_frac"],
                        m["use_refiner"], m.get("seed_effective", ""), int(m.get("augmented", False)),
                    ])

    print(f"[DONE] images -> {out_dir}")
    print(f"[DONE] candidates.csv -> {cand_csv}")
    if fails:
        print(f"[WARN] failures: {len(fails)} -> {fail_csv}")


# ----------------------------------------------------------
# Entry
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
