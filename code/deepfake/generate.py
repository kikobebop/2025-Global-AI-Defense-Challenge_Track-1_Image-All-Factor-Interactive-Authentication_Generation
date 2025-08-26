"""
批量执行 InSwapper 换脸：
1. 从配置文件读取任务信息（原图 + 目标图）；
2. 调用 InsightFace InSwapper 模型进行人脸替换；
3. 输出换脸结果和 candidates.csv（包含每个结果的基础信息与可选的相似度得分）。

用法：
    python -m code.deepfake.generate -c configs/deepfake.yaml

说明：
- 适合批量换脸，支持 GPU/CPU，CPU 下也能较快处理。
- 运行结果目录结构：
    <output.dir>/<run.name>_<timestamp>/
        ├── candidates.csv   # 记录所有成功的换脸结果及路径
        ├── failures.csv     # 记录换脸失败的条目
        ├── <index>_inswapper.jpg  # 每个样本的换脸输出图像
        └── config_resolved.json   # 本次运行的配置快照，方便复现
"""
import os, sys, csv, time, json, argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# —— 模块路径兼容：支持直接运行或 `-m code.deepfake.generate` ——
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception as e:
    print("Missing PyYAML. `pip install pyyaml`.", file=sys.stderr)
    raise

from deepfake.backend_inswapper import (
    init_inswapper,
    face_swap_target_into_ori_inswapper,
    cosine_similarity_target_vs_generated
)

# ---------------- 工具函数 ----------------

def load_cfg(p: str):
    """加载 YAML 配置文件并返回字典。"""
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def req(d: dict, k: str, typ):
    """
    从配置字典 d 中提取 key 并检查类型。
    - d: 配置字典
    - k: 键名
    - typ: 期望类型
    """
    if k not in d:
        raise KeyError(f"Missing config key: {k}")
    try:
        return typ(d[k])
    except:
        raise TypeError(f"`{k}` must be {typ.__name__}, got {d[k]!r}")

def resolve_path(val: str, image_root: str) -> str:
    """
    解析图像路径：
    - 若为绝对路径或包含斜杠，则直接返回；
    - 否则拼接 image_root。
    """
    val = str(val).strip()
    if not val:
        return ""
    p = Path(val)
    if p.is_absolute() or "/" in val or "\\" in val:
        return str(p)
    return str(Path(image_root) / val)

def dump_snapshot(run_dir: str, cfg_resolved: dict):
    """
    将本次运行的解析配置保存为 JSON 文件，方便后续复现。
    保存路径：
        <run_dir>/config_resolved.json
    """
    snap = {
        "argv": sys.argv,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg_resolved
    }
    with open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

# ---------------- 主流程 ----------------

def main():
    # 解析命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="configs/deepfake.yaml")
    args = ap.parse_args()

    # 读取配置文件
    cfg = load_cfg(args.config)
    run_name = cfg.get("run", {}).get("name", "deepfake_run")
    out_root = req(cfg.get("output", {}), "dir", str)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"{run_name}_{ts}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- 数据准备 ----------------
    data_cfg = cfg.get("data", {})
    task_csv = req(data_cfg, "task_csv", str)
    filter_types = data_cfg.get("filter_task_types", ["deepfake"])
    image_root = data_cfg.get("image_root", "")
    index_min = int(data_cfg.get("index_min", -10**18))
    index_max = int(data_cfg.get("index_max", 10**18))

    # 读取任务 CSV
    df = pd.read_csv(task_csv)
    need_cols = {"index", "task_type", "ori_image", "target_image"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"task_csv 需要列：{need_cols}")

    # 筛选类型与索引范围
    df = df[df["task_type"].astype(str).str.lower().isin([t.lower() for t in filter_types])]
    df = df[(df["index"] >= index_min) & (df["index"] < index_max)][["index", "ori_image", "target_image"]]

    # 解析有效条目
    items = []
    for r in df.itertuples():
        idx = int(getattr(r, "index"))
        ori = resolve_path(getattr(r, "ori_image"), image_root) if image_root else str(getattr(r, "ori_image"))
        tgt = resolve_path(getattr(r, "target_image"), image_root) if image_root else str(getattr(r, "target_image"))
        if os.path.exists(ori) and os.path.exists(tgt):
            items.append((idx, ori, tgt))

    if not items:
        print("[WARN] 无可处理样本。")
        return

    # ---------------- 初始化 InSwapper 后端 ----------------
    bp = cfg.get("backend_params", {}).get("inswapper", {})
    app_name = bp.get("app_name", "buffalo_l")
    det_size = tuple(bp.get("det_size", [640, 640]))
    model_path = req(bp, "model_path", str)
    providers = bp.get("providers", "auto")

    init_inswapper(app_name=app_name, det_size=det_size, model_path=model_path, providers=providers)

    # ---------------- 相似度评分配置 ----------------
    scoring = cfg.get("scoring", {})
    score_enable = bool(scoring.get("enable", True))
    save_cos_in_candidates = bool(scoring.get("save_to_candidates", True))

    # 保存运行配置快照
    dump_snapshot(run_dir, {
        "run": {"name": run_name, "ts": ts},
        "data": {
            "task_csv": task_csv,
            "image_root": image_root,
            "index_min": index_min,
            "index_max": index_max,
            "filter": filter_types
        },
        "output": {"dir": out_root, "run_dir": run_dir},
        "backend_params": {
            "inswapper": {
                "app_name": app_name,
                "det_size": list(det_size),
                "model_path": model_path,
                "providers": providers
            }
        },
        "scoring": scoring
    })

    # ---------------- 输出文件初始化 ----------------
    cand_csv = os.path.join(run_dir, "candidates.csv")
    fail_csv = os.path.join(run_dir, "failures.csv")

    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 标准化列头
        w.writerow([
            "index", "candidate_id", "backend",
            "image_path", "ori_image", "target_image", "cos_sim"
        ])

    fails = []

    # ---------------- 批量换脸 ----------------
    for idx, ori, tgt in tqdm(items, desc="Deepfake InSwapper", unit="img"):
        try:
            # 执行换脸
            out = face_swap_target_into_ori_inswapper(tgt, ori)
            out_path = os.path.join(run_dir, f"{idx}_inswapper.jpg")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 保存换脸图像
            import cv2
            cv2.imwrite(out_path, out)

            # 计算相似度
            cos = None
            if score_enable and save_cos_in_candidates:
                cos = cosine_similarity_target_vs_generated(tgt, out_path)

            # 写入 candidates.csv
            with open(cand_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    idx, "00", "inswapper", out_path, ori, tgt,
                    (f"{cos:.6f}" if (cos is not None and cos == cos) else "")
                ])

        except Exception as e:
            # 记录失败条目
            fails.append({
                "index": idx,
                "ori": ori,
                "target": tgt,
                "error": str(e)
            })

    # ---------------- 保存失败信息 ----------------
    if fails:
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "backend", "error", "ori_image", "target_image"])
            for r in fails:
                w.writerow([r["index"], "inswapper", r["error"], r["ori"], r["target"]])

    # ---------------- 汇总输出 ----------------
    print(f"[DONE] images -> {run_dir}")
    print(f"[DONE] candidates.csv -> {cand_csv}")
    if fails:
        print(f"[WARN] failures: {len(fails)} -> {fail_csv}")

if __name__ == "__main__":
    main()
