"""End-to-end apply: optional pair-based color transfer + global + face."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.face import apply_face_recipe
from app.lut import apply_histogram_match_with_strength
from app.recipe import Recipe, apply_global_recipe


def apply_full_pipeline(
    new_bgr: np.ndarray,
    recipe: Recipe,
    ref_original_bgr: Optional[np.ndarray],
    ref_edited_bgr: Optional[np.ndarray],
) -> np.ndarray:
    out = new_bgr
    cg = recipe.color_grading
    if (
        cg.use_lut
        and cg.histogram_match_strength > 0
        and ref_original_bgr is not None
        and ref_edited_bgr is not None
    ):
        out = apply_histogram_match_with_strength(
            out,
            ref_original_bgr,
            ref_edited_bgr,
            cg.histogram_match_strength,
        )
    out = apply_global_recipe(out, recipe.global_)
    f = recipe.face
    out = apply_face_recipe(out, f.skin_smooth_strength, f.skin_warmth, f.lip_saturation)
    return out


def decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    return img


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("编码失败")
    return buf.tobytes()
