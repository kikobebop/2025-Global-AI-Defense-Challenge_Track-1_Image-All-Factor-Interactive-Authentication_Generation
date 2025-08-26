#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deepfake 后端模块（InsightFace InSwapper, CUDA12 兼容）
-------------------------------------------------------
功能：
1. 优先使用 CUDAExecutionProvider，如失败则自动回退 CPU 并打印原因
2. 自动补齐 CUDA/cuDNN 等动态库路径，适配 pip 安装的 nvidia-* 库
3. 使用极小 ONNX 模型进行 CUDA EP 自检，避免无声降级到 CPU
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ==========================================================
# 1. 补齐 CUDA 相关路径
# ==========================================================
def _ensure_cuda_paths() -> None:
    """
    将常见 CUDA/cuDNN 库路径加入 LD_LIBRARY_PATH，确保运行时能加载到 GPU 库
    """
    import site
    import glob

    extra: List[str] = []

    # 1) 集群 module 提供的 CUDA 路径
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        extra += [
            os.path.join(cuda_home, "lib64"),
            os.path.join(cuda_home, "lib"),
            os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
            os.path.join(cuda_home, "extras", "CUPTI", "lib64"),
            os.path.join(cuda_home, "nvvm", "lib64"),
        ]

    # 2) pip 安装的 nvidia-* 库路径（nvidia-cudnn-cu12、nvidia-cublas-cu12 等）
    for base in (site.getsitepackages() + [site.getusersitepackages()]):
        for sub in ["cublas", "cudnn", "curand", "cuda_runtime", "cusparse", "cufft", "cudnn_cu12", "cublas_cu12"]:
            extra += glob.glob(os.path.join(base, "nvidia", sub, "lib"))
            extra += glob.glob(os.path.join(base, "nvidia", sub, "lib", "**"), recursive=True)

    # 3) 系统常见路径兜底
    extra += ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"]

    # 合并去重并写回 LD_LIBRARY_PATH
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for p in extra:
        if p and os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

_ensure_cuda_paths()

# ==========================================================
# 2. 导入依赖模块
# ==========================================================
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    _ORT_AVAILABLE = False

import insightface
from insightface.app import FaceAnalysis

# ==========================================================
# 3. 全局变量
# ==========================================================
_APP: Optional[FaceAnalysis] = None   # 人脸检测/识别器
_SWAPPER = None                      # InSwapper 模型
_PROVIDERS: List[str] = []           # 实际使用的推理后端
_CTX_ID: int = -1                    # GPU=0, CPU=-1

# ==========================================================
# 4. CUDA 自检
# ==========================================================
def _cuda_smoketest() -> Tuple[bool, str]:
    """
    通过加载一个极小的 ONNX 模型测试 CUDA 是否可用
    返回 (是否可用, 信息字符串)
    """
    if not _ORT_AVAILABLE:
        return False, "onnxruntime 未加载"

    try:
        import tempfile, onnx
        from onnx import helper, TensorProto

        # 构造极小模型：输入输出相同（Identity）
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        g = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "g", [X], [Y])
        m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 13)])
        m.ir_version = 9

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(m, f.name)
            sess = ort.InferenceSession(
                f.name, ort.SessionOptions(),
                providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            )
            used = sess.get_providers()

        if "CUDAExecutionProvider" in used:
            return True, f"CUDA 可用: {used}"
        return False, f"CUDA 未启用: {used}"
    except Exception as e:
        return False, f"CUDA 初始化失败: {e}"

# ==========================================================
# 5. 选择推理后端
# ==========================================================
def _select_providers(cfg_providers) -> List[str]:
    """
    根据配置选择推理后端：
    - "auto": 优先 CUDA，不可用则回退 CPU
    - list: 维持用户顺序，但确保 CUDA 优先
    """
    def _cuda_first(lst: List[str]) -> List[str]:
        lst = list(dict.fromkeys(lst))
        if "CUDAExecutionProvider" in lst:
            lst.remove("CUDAExecutionProvider")
            lst.insert(0, "CUDAExecutionProvider")
        return lst

    if cfg_providers == "auto":
        ok, msg = _cuda_smoketest()
        print(f"[CUDA 自检] {msg}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if ok else ["CPUExecutionProvider"]

    if isinstance(cfg_providers, list) and cfg_providers:
        return _cuda_first(cfg_providers)

    return ["CPUExecutionProvider"]

# ==========================================================
# 6. 模型初始化
# ==========================================================
def init_inswapper(app_name: str, det_size: Tuple[int, int], model_path: str, providers="auto") -> None:
    """
    初始化 InSwapper 和人脸分析器（仅首次调用时加载）
    """
    global _APP, _SWAPPER, _PROVIDERS, _CTX_ID
    if _APP is not None and _SWAPPER is not None:
        return

    _PROVIDERS = _select_providers(providers)
    print(f"[INFO] 选择的推理后端: {_PROVIDERS}")
    _CTX_ID = 0 if (_PROVIDERS and _PROVIDERS[0].startswith("CUDA")) else -1

    # 初始化人脸分析器
    _APP = FaceAnalysis(name=app_name, providers=_PROVIDERS)
    _APP.prepare(ctx_id=_CTX_ID, det_size=tuple(det_size))

    # 确定 InSwapper 模型路径
    mp = os.path.expanduser(model_path)
    if os.path.isdir(mp):
        cand1 = os.path.join(mp, "models", "inswapper_128.onnx")
        cand2 = os.path.join(mp, "inswapper_128.onnx")
        mp = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else mp)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"InSwapper 模型不存在: {mp}")

    # 加载 InSwapper 模型
    _SWAPPER = insightface.model_zoo.get_model(mp, providers=_PROVIDERS)
    if _SWAPPER is None:
        raise RuntimeError(f"加载 InSwapper 模型失败: {mp}")

    # 打印实际使用的后端，方便确认是否启用了 GPU
    try:
        if _ORT_AVAILABLE:
            print("[INFO] ORT 可用后端:", ort.get_available_providers())
        if hasattr(_APP, "det_model") and hasattr(_APP.det_model, "session"):
            print("[INFO] 人脸检测:", _APP.det_model.session.get_providers())
        if hasattr(_APP, "rec_model") and getattr(_APP, "rec_model", None) and hasattr(_APP.rec_model, "session"):
            print("[INFO] 人脸识别:", _APP.rec_model.session.get_providers())
        if hasattr(_SWAPPER, "session"):
            print("[INFO] InSwapper:", _SWAPPER.session.get_providers())
    except Exception as e:
        print("[WARN] 无法获取 session 信息:", e)

# ==========================================================
# 7. 工具函数
# ==========================================================
def _largest_face(faces):
    """返回面积最大的人脸（多脸时默认选主脸）"""
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

def _get_normed_embedding_from_img(img: np.ndarray):
    """提取最大人脸的归一化特征向量"""
    faces = _APP.get(img)
    if not faces:
        return None
    face = _largest_face(faces)
    emb = getattr(face, "normed_embedding", None)
    return emb.astype(np.float32) if emb is not None else None

def cosine_similarity_target_vs_generated(target_img_path: str, generated_img_or_path) -> float:
    """计算生成结果与目标身份图像的余弦相似度"""
    gen_img = cv2.imread(generated_img_or_path) if isinstance(generated_img_or_path, str) else generated_img_or_path
    tgt_img = cv2.imread(target_img_path)
    if tgt_img is None or gen_img is None:
        return float("nan")
    a = _get_normed_embedding_from_img(tgt_img)
    b = _get_normed_embedding_from_img(gen_img)
    if a is None or b is None:
        return float("nan")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def face_swap_target_into_ori_inswapper(target_img_path: str, ori_img_path: str) -> np.ndarray:
    """执行换脸，将 target 图像的人脸替换到 ori 图像中"""
    src = cv2.imread(target_img_path)
    dst = cv2.imread(ori_img_path)
    if src is None or dst is None:
        raise FileNotFoundError("读取图片失败")
    src_faces = _APP.get(src)
    dst_faces = _APP.get(dst)
    if len(src_faces) == 0 or len(dst_faces) == 0:
        raise ValueError("未检测到人脸")
    src_face = _largest_face(src_faces)
    dst_face = _largest_face(dst_faces)
    return _SWAPPER.get(dst, dst_face, src_face, paste_back=True)
