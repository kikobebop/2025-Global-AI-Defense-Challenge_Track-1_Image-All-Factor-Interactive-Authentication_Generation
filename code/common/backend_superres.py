# code/common/backend_superres.py
# -*- coding: utf-8 -*-
"""
通用超分辨率后端 (RealESRGAN_x4plus)
--------------------------------------------------
概览:
- 封装 RealESRGAN 的核心推理功能
- 输入 BGR 图像，输出增强后的 BGR 图像
- 支持非整数放大倍数（内部始终使用 4x 权重进行重建）

Overview:
- Wrapper for RealESRGAN upscaling (x4 model)
- Input: BGR uint8 image → Output: BGR uint8 image
- Supports non-integer scaling (internally always reconstructs with x4 weights)
--------------------------------------------------
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class ESRGANUpsampler:
    """
    RealESRGAN upscaling utility class.

    Args:
        weights_path (str | Path): Path to the RealESRGAN_x4plus .pth weights
        tile (int): Tile size for tiled inference (default=0, disabled)
        tile_pad (int): Padding around each tile to reduce seams
        use_half (bool): Use half precision if supported (default: True on CUDA)
        device (str): Explicit device to run on ("cuda", "mps", "cpu")
    """

    def __init__(
        self,
        weights_path: str | Path,
        tile: int = 0,
        tile_pad: int = 10,
        use_half: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if use_half is None:
            use_half = (device == "cuda")

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

        self._upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights_path),
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=use_half,
            device=device
        )

    def enhance(self, img_bgr: np.ndarray, outscale: float = 2.0) -> np.ndarray:
        """
        Perform one-step RealESRGAN enhancement.

        Args:
            img_bgr (np.ndarray): Input BGR image (uint8)
            outscale (float): Desired scale factor (can be non-integer)

        Returns:
            np.ndarray: Upscaled BGR image (uint8)
        """
        out, _ = self._upsampler.enhance(img_bgr, outscale=float(outscale))
        return out


def resize_to_long_edge(img_bgr: np.ndarray, target_long: int) -> np.ndarray:
    """
    Resize an image to match the target length of its longer edge, preserving aspect ratio.

    Args:
        img_bgr (np.ndarray): Input BGR image
        target_long (int): Desired long edge size

    Returns:
        np.ndarray: Resized BGR image
    """
    h, w = img_bgr.shape[:2]
    long0 = max(w, h)

    if long0 == 0 or long0 == target_long:
        return img_bgr

    if w >= h:
        new_w = target_long
        new_h = int(round(h * (target_long / w)))
    else:
        new_h = target_long
        new_w = int(round(w * (target_long / h)))

    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
