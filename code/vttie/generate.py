# ==========================================================
# 视觉文本编辑（VTTIE）任务 - 图像生成主脚本
# 功能（中文）：
#   1) 读取配置（configs/vttie.yaml），解析任务与模型参数
#   2) 从任务 CSV 加载原图与文字编辑指令
#   3) 调用后端（Qwen-Image-Edit / FLUX-Kontext-Edit）批量生成
#   4) 保存结果图、运行快照、candidates.csv 与 failures.csv，确保可复现
#
# VTTIE (Visual Text-in-Image Editing) — Generate Script
# Features (English):
#   1) Parse config (configs/vttie.yaml) for tasks and model params
#   2) Load source images and text-edit instructions from CSV
#   3) Run chosen backend (Qwen-Image-Edit / FLUX-Kontext-Edit) in batch
#   4) Save edited images, a config snapshot, candidates.csv and failures.csv for reproducibility
# ==========================================================

import os, sys, csv, argparse, json, time
from pathlib import Path
import pandas as pd

# Ensure both script run and module import work
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent  # .../code
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception as e:
    print("Missing dependency: PyYAML. Please `pip install pyyaml`.", file=sys.stderr)
    raise

from vttie_backend_qwen_edit import load_qwen_edit, gen_qwen_edit
from vttie_backend_flux import load_flux_edit, gen_flux_edit


def load_cfg(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    """Create directory if it does not exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def req(d: dict, key: str, typ):
    """Read a required key from a dict and cast to a type."""
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    try:
        return typ(d[key])
    except Exception:
        raise TypeError(f"Config `{key}` must be {typ.__name__}, got: {d[key]!r}")


def dump_snapshot(run_dir: str, cfg_resolved: dict):
    """Save a reproducible snapshot of runtime arguments and resolved config."""
    snap = {"argv": sys.argv, "time_start": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg_resolved}
    with open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)


def _resolve_img_path(p_raw: str, img_root: str):
    """
    Resolve original image path:
      1) absolute or existing relative path
      2) join with img_root if provided
      3) fallback to <repo_root>/data/imgs/<p_raw>
    """
    if not isinstance(p_raw, str) or not p_raw.strip():
        return None
    p = p_raw.strip()
    # Absolute or existing relative path
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    # Under img_root
    if img_root:
        j = os.path.join(img_root, p)
        if os.path.exists(j):
            return j
    # Fallback to repo data/imgs
    repo_root = PKG_ROOT.parent
    guess = os.path.join(repo_root, "data", "imgs", p)
    if os.path.exists(guess):
        return guess
    return None


def main():
    # -------- Parse CLI --------
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config (vttie.yaml)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # -------- Basics --------
    run_name = cfg.get("run", {}).get("name", "vttie_run")
    data_cfg = cfg.get("data", {})
    data_csv = req(data_cfg, "task_csv", str)
    img_root = data_cfg.get("img_root", "")
    type_filter = data_cfg.get("filter_task_types", ["vttie"])
    index_max = int(data_cfg.get("index_max", 10**9))

    out_root = req(cfg.get("output", {}), "dir", str)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{run_name}_{ts}")
    ensure_dir(out_dir)

    pipe_cfg = cfg.get("pipeline", {})
    backend = pipe_cfg.get("backend", "flux_edit")
    dtype = str(pipe_cfg.get("dtype", "bf16")).lower()
    num_candidates = int(pipe_cfg.get("num_candidates", 1))
    if "seeds" not in pipe_cfg:
        raise KeyError("Missing pipeline.seeds (int or [int,int,...])")
    seeds = pipe_cfg["seeds"]

    # -------- Load task rows --------
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"task_csv not found: {data_csv}")
    df = pd.read_csv(data_csv)
    need_cols = {"index", "task_type", "prompt", "ori_image"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"task_csv must have columns: {need_cols}")

    df_vttie = df[df["task_type"].isin(type_filter)]
    df_vttie = df_vttie[df_vttie["index"] < index_max][["index", "prompt", "ori_image"]]

    # -------- Resolve original image paths --------
    df_vttie["ori_image_full"] = df_vttie["ori_image"].map(lambda p: _resolve_img_path(p, img_root))
    miss = df_vttie[df_vttie["ori_image_full"].isna()]
    if len(miss) > 0:
        print(f"[WARN] Missing source images (showing head):")
        print(miss.head().to_string(index=False))

    df_vttie = df_vttie.dropna(subset=["ori_image_full"])
    items = [(int(r.index), str(r.prompt), str(r.ori_image_full)) for r in df_vttie.itertuples()]
    if not items:
        print("[WARN] No editable items. Check `ori_image` paths and `img_root`.")
        return

    # -------- Dispatch by backend --------
    if backend == "qwen_edit":
        p = cfg.get("backend_params", {}).get("qwen_edit", {})
        model_path = req(p, "model_path", str)
        steps = req(p, "steps", int)
        true_cfg = req(p, "true_cfg", float)
        negative_prompt = p.get("negative_prompt", "")
        augment = p.get("augment", None)

        cfg_resolved = {
            "run": {"name": run_name, "ts": ts},
            "data": {"task_csv": data_csv, "img_root": img_root, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "qwen_edit", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"qwen_edit": {
                "model_path": model_path, "steps": steps, "true_cfg": true_cfg,
                "negative_prompt": negative_prompt, "augment": augment
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        use_bf16 = (dtype == "bf16")
        pipe = load_qwen_edit(model_path, use_bf16=use_bf16)

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
            "data": {"task_csv": data_csv, "img_root": img_root, "filter_task_types": type_filter, "index_max": index_max},
            "output": {"dir": out_root, "run_dir": out_dir},
            "pipeline": {"backend": "flux_edit", "dtype": dtype, "num_candidates": num_candidates, "seeds": seeds},
            "backend_params": {"flux_edit": {
                "model_path": model_path, "steps": steps, "guidance_scale": guidance_scale,
                "augment": augment
            }},
        }
        dump_snapshot(out_dir, cfg_resolved)

        use_bf16 = (dtype == "bf16")
        pipe = load_flux_edit(model_path, use_bf16=use_bf16)

        results = gen_flux_edit(
            pipe, items, out_dir,
            steps=steps, guidance_scale=guidance_scale, seeds=seeds,
            num_candidates=num_candidates, augment=augment,
            negative_prompt=negative_prompt,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # -------- Write candidates.csv & failures.csv --------
    cand_csv = os.path.join(out_dir, "candidates.csv")
    fail_csv = os.path.join(out_dir, "failures.csv")

    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if backend == "qwen_edit":
            w.writerow(["index", "candidate_id", "backend", "image_path", "src_image",
                        "seed_effective", "steps", "true_cfg", "augmented"])
        else:
            w.writerow(["index", "candidate_id", "backend", "image_path", "src_image",
                        "seed_effective", "steps", "guidance_scale", "augmented"])

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
