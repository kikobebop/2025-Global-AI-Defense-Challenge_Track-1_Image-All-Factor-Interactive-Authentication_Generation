# -*- coding: utf-8 -*-
"""
批量后处理：对 InSwapper 的换脸结果进行 FLUX.1-Kontext-dev 轻度修补
- 输入优先从 run_dir/candidates.csv 读取（列：image_path），也支持按模式遍历目录
- 自动 letterbox 到 1024×1024，再还原到原尺寸，尽量“只修不改样”
- 输出保存到 run_dir/post_flux/ 或配置指定目录
用法：
  python -m code.deepfake.post_edit_flux -c configs/deepfake.yaml

设计要点：
1) 统一的 I/O 入口：
   - 推荐：读取某次换脸运行目录 <run_dir>/candidates.csv（来自 deepfake.generate 的产物）。
   - 兼容：按目录扫描（source_root + filename_contains），用于历史结果或临时数据的修补。
2) 尺寸处理：
   - FLUX Kontext 通常以 1024×1024 输入最稳，因此先 letterbox 到方形，再在导出时“无损还原”至原尺寸。
   - 填充色默认 (127,127,127)，避免对整体色调产生主观偏置。
3) 轻度修补策略：
   - prompt/negative_prompt 默认偏“只修不改样”，不主动改变身份、发型、背景、构图等。
   - steps/guidance 可在 YAML 中调整，越大越“听指令”，但越可能偏离原貌；建议 28~32 / 2.0~5.5。
"""

import os, sys, csv, json, time, argparse, traceback
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image, ImageOps

# —— 允许模块方式运行 ——
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import yaml
except Exception:
    print("Missing PyYAML. Please `pip install pyyaml`.", file=sys.stderr)
    raise

from diffusers import FluxKontextPipeline

# ============== 通用工具 ==============

def load_cfg(path: str) -> dict:
    """
    读取 YAML 配置文件。
    参数：
        path: 配置文件路径（通常为 configs/deepfake.yaml）
    返回：
        dict 类型的配置对象
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    """确保目录存在（若不存在则递归创建）。"""
    Path(p).mkdir(parents=True, exist_ok=True)

def dump_snapshot(run_dir: str, cfg_resolved: dict):
    """
    将本次后处理的解析配置写入 JSON 快照，便于复现与审计。
    文件名：post_flux_config_resolved.json（存放于输出目录）
    """
    snap = {"argv": sys.argv, "time_start": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg_resolved}
    with open(os.path.join(run_dir, "post_flux_config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

def _resolve_img_path(val: str, root: str) -> Optional[str]:
    """
    将候选图像路径解析为绝对可用路径：
      - 绝对路径或当前工作目录下存在的相对路径 → 原样返回
      - 若提供 root，则尝试 root/val
      - 均失败则返回 None
    """
    if not isinstance(val, str) or not val.strip():
        return None
    p = val.strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    if root:
        j = os.path.join(root, p)
        if os.path.exists(j):
            return j
    return None

# ============== 图像 I/O 与还原 ==============

def _letterbox_square(img_path: str, side: int = 1024, fill=(127,127,127)):
    """
    等比缩放 + 灰边补齐到 side×side（letterbox），返回：
      - 原图 PIL.Image（已转 RGB）
      - 方图 PIL.Image（side×side）
      - 元信息元组 meta=(w0, h0, new_w, new_h, pad_left, pad_top)，用于后续还原尺寸
    说明：
      - 使用 LANCZOS 插值在缩放阶段尽量保留细节
      - 灰色填边可降低对视觉色调的干扰
    """
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.width, im.height
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"Bad image size: {w0}x{h0}")
    r = min(side / w0, side / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    pad_left = (side - new_w) // 2
    pad_right = side - new_w - pad_left
    pad_top = (side - new_h) // 2
    pad_bot = side - new_h - pad_top
    im_sq = ImageOps.expand(im_resized, border=(pad_left, pad_top, pad_right, pad_bot), fill=fill)
    meta = (w0, h0, new_w, new_h, pad_left, pad_top)
    return im, im_sq, meta

def _unletterbox(out_sq: Image.Image, meta):
    """
    将 FLUX 输出的方图裁剪掉填边，并按原图尺寸拉回：
      - 利用 meta 定位有效内容区域（去除灰边）
      - 采用 LANCZOS 将裁剪后区域 resize 回原图尺寸
    """
    w0, h0, new_w, new_h, pad_left, pad_top = meta
    crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
    out_cropped = out_sq.crop(crop_box)
    return out_cropped.resize((w0, h0), Image.LANCZOS)

# ============== 主流程 ==============

def main():
    """
    主入口：
      1) 读取 deepfake.yaml 中的 postprocess.flux_edit 配置（模型、参数、输入与输出路径）。
      2) 解析输入来源（run_dir/candidates.csv 或 source_root 扫描）。
      3) 批量调用 FLUX.1-Kontext-dev 进行“轻度修补”，尽可能仅消除局部伪影与接缝。
      4) 输出保存到 out_dir（默认 <run_dir>/post_flux 或 <source_root>/post_flux）。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="configs/deepfake.yaml（含 postprocess.flux_edit）")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # —— 读取后处理配置 —— #
    pp = cfg.get("postprocess", {}).get("flux_edit", {})
    enable = bool(pp.get("enable", True))
    if not enable:
        print("[INFO] postprocess.flux_edit.enable = false，退出。")
        return

    model_path = pp.get("model_path")
    if not model_path:
        raise KeyError("postprocess.flux_edit.model_path 必填（FLUX.1-Kontext-dev 路径）")

    # 核心生成参数（可在 YAML 中调节）
    steps = int(pp.get("steps", 32))            # 迭代步数：越大越“听话”，但越慢，且更易改变原貌
    guidance = float(pp.get("guidance", 5.5))   # 引导强度：2.0~5.5 比较稳
    side = int(pp.get("side", 1024))            # 输入方边长（Kontext 常用 1024）
    seed = int(pp.get("seed", 124))             # 固定随机种子，保证可复现
    fill = tuple(pp.get("fill_rgb", [127,127,127]))  # letterbox 灰边
    skip_existing = bool(pp.get("skip_existing", True))  # 已存在同名输出时是否跳过

    # 默认“只修补、不重绘”的 prompt/negative
    prompt = str(pp.get("prompt", (
        "remove unnatural seams around the face if any, "
        "remove incomplete eyeglasses or artifacts around the eyes if any, "
        "match surrounding lighting and skin tone; preserve identity, background, hair and clothing"
    )))
    negative_prompt = str(pp.get("negative_prompt", (
        "makeup, beauty filter, face reshape, sharpen, oversharpened, "
        "plastic skin, waxy, color shift, background change, hairstyle change, text, watermark"
    )))

    # —— 输入来源：优先 run_dir（读取 candidates.csv）；否则走 source_root + pattern —— #
    input_cfg = pp.get("input", {})
    run_dir = input_cfg.get("run_dir", "")
    candidates_csv = input_cfg.get("candidates_csv", "")
    source_root = input_cfg.get("source_root", "")
    filename_contains = input_cfg.get("filename_contains", "_inswapper")  # 目录扫描时用于筛选文件名的关键字
    out_dir = pp.get("out_dir", "")  # 若空则默认写到 run_dir/post_flux

    # 设备与精度：A100/H100 时用 bfloat16；其他 GPU 用 fp16；CPU 用 fp32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    props = torch.cuda.get_device_properties(0) if device == "cuda" else None
    use_bf16 = (device == "cuda") and (props is not None) and (("A100" in props.name) or ("H100" in props.name))
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    print(f"[INFO] torch {torch.__version__}, device: {props.name if props else 'CPU'} | dtype: "
          f"{'bf16' if use_bf16 else ('fp16' if device=='cuda' else 'fp32')}")

    # 加载 FLUX 模型
    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    try:
        if hasattr(pipe, "vae"):
            # 关闭 VAE 的 slicing/tiling 以避免潜在的边界拼接痕迹
            pipe.vae.disable_slicing(); pipe.vae.disable_tiling()
    except Exception:
        pass
    if device == "cuda":
        # 合理打开 TF32 / benchmark，兼顾吞吐与稳定性
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 解析输入列表
    inputs: List[str] = []
    if run_dir or candidates_csv:
        # 情况 A：从候选表读取（推荐：run_dir 下的 candidates.csv）
        if not candidates_csv:
            candidates_csv = os.path.join(run_dir, "candidates.csv")
        if not os.path.exists(candidates_csv):
            raise FileNotFoundError(f"找不到 candidates.csv: {candidates_csv}")
        df = pd.read_csv(candidates_csv)
        if "image_path" not in df.columns:
            raise ValueError("candidates.csv 缺少列：image_path")
        # 仅保留真实存在的路径
        inputs = [p for p in df["image_path"].tolist() if isinstance(p, str) and os.path.exists(p)]
        # 默认输出目录：<run_dir>/post_flux
        if not out_dir:
            out_dir = os.path.join(run_dir, "post_flux")
    else:
        # 情况 B：目录扫描兜底（兼容历史 faceswap_result 结构）
        if not os.path.isdir(source_root):
            raise FileNotFoundError(f"source_root 不存在：{source_root}")
        for dirpath, _, filenames in os.walk(source_root):
            for fn in filenames:
                if filename_contains and (filename_contains not in fn):
                    continue
                if fn.lower().endswith(".jpg") or fn.lower().endswith(".png"):
                    inputs.append(os.path.join(dirpath, fn))
        if not out_dir:
            # 若未指定，默认在 source_root 平级创建一个目录
            out_dir = os.path.join(source_root, "post_flux")

    inputs.sort()
    ensure_dir(out_dir)

    # 保存快照到输出目录（不是 run_dir），避免污染原始换脸结果
    dump_snapshot(out_dir, {
        "model_path": model_path,
        "run_dir": run_dir,
        "candidates_csv": candidates_csv if candidates_csv else "",
        "source_root": source_root if source_root else "",
        "filename_contains": filename_contains,
        "out_dir": out_dir,
        "params": {
            "steps": steps, "guidance": guidance, "side": side,
            "seed": seed, "fill_rgb": list(fill), "skip_existing": skip_existing
        }
    })

    if not inputs:
        print("[WARN] 没有可处理的图像。")
        return

    print(f"[INFO] 待修补图像数：{len(inputs)}")

    # 固定生成器（CPU 端）以保证跨设备可复现
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # —— 主处理循环 —— #
    for in_path in tqdm(inputs, ncols=100, desc="post-edit (FLUX)"):
        # 输出文件名：沿用原名，写到 out_dir
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(out_dir, f"{base}.jpg")
        if skip_existing and os.path.exists(out_path):
            # 已存在时跳过，便于断点续跑
            continue
        try:
            # 1) letterbox 到方图
            _, im_sq, meta = _letterbox_square(in_path, side=side, fill=fill)
            # 2) 调用 FLUX 生成修补结果（轻度修补）
            with torch.inference_mode():
                out_sq = pipe(
                    image=im_sq,
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    guidance_scale=float(guidance),
                    num_inference_steps=int(steps),
                    generator=g,
                ).images[0]
            # 3) 还原回原图分辨率
            out_final = _unletterbox(out_sq, meta)
            out_final.save(out_path, quality=100)
        except Exception:
            # 单张失败不影响整体流程；输出堆栈方便定位问题图像
            print(f"[ERROR] {in_path}")
            traceback.print_exc()
            continue

    print(f"[DONE] saved to {out_dir}")

if __name__ == "__main__":
    main()
