"""Face detection (OpenCV Haar) and lightweight skin-region adjustments (MVP)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

_cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def detect_face_bbox(
    bgr: np.ndarray, min_confidence: float = 0.5
) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=max(3, int(5 - min_confidence * 5)),
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return None
    areas = [w * h for (_, _, w, h) in faces]
    i = int(np.argmax(areas))
    x, y, w, h = faces[i]
    return int(x), int(y), int(x + w), int(y + h)


def estimate_face_stats(bgr: np.ndarray) -> Optional[Dict[str, float]]:
    bbox = detect_face_bbox(bgr)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    return {
        "mean_l": float(np.mean(l)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_saturation": float(np.mean(s)),
        "bbox_rel": (
            x1 / bgr.shape[1],
            y1 / bgr.shape[0],
            (x2 - x1) / bgr.shape[1],
            (y2 - y1) / bgr.shape[0],
        ),
    }


def diff_face_recipe(orig_bgr: np.ndarray, edited_bgr: np.ndarray) -> Dict[str, Any]:
    o = estimate_face_stats(orig_bgr)
    e = estimate_face_stats(edited_bgr)
    if o is None or e is None:
        return {
            "skin_smooth_strength": 0.0,
            "skin_warmth": 0.0,
            "lip_saturation": 0.0,
            "face_slim_approx": 0.0,
            "notes": "未检测到稳定人脸区域，人脸相关配方为默认值。",
        }
    warmth = np.clip((e["mean_a"] - o["mean_a"]) / 25.0 + (e["mean_b"] - o["mean_b"]) / 40.0, -1.0, 1.0)
    lip_sat = np.clip((e["mean_saturation"] - o["mean_saturation"]) / 80.0, -0.5, 0.5)
    smooth_guess = np.clip((o["mean_l"] - e["mean_l"]) / 80.0 + 0.15, 0.0, 0.8)
    return {
        "skin_smooth_strength": float(smooth_guess * 0.5),
        "skin_warmth": float(warmth),
        "lip_saturation": float(lip_sat),
        "face_slim_approx": 0.0,
        "notes": "由面部 Lab/HSV 差异估计的粗略美颜相关参数。",
    }


def _bilateral_smooth(bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.01:
        return bgr
    d = max(3, int(5 + strength * 10))
    sigma = 20 + strength * 40
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)


def apply_face_recipe(bgr: np.ndarray, skin_smooth: float, skin_warmth: float, lip_sat: float) -> np.ndarray:
    bbox = detect_face_bbox(bgr)
    if bbox is None:
        return bgr
    x1, y1, x2, y2 = bbox
    face = bgr[y1:y2, x1:x2].copy()
    face = _bilateral_smooth(face, skin_smooth)
    if abs(skin_warmth) > 0.02 or abs(lip_sat) > 0.02:
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        a = np.clip(a + skin_warmth * 6.0, 0, 255)
        b = np.clip(b + skin_warmth * 4.0, 0, 255)
        lab = cv2.merge([l, a, b])
        face = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        fh = face.shape[0]
        lip_y1 = int(fh * 0.55)
        s[lip_y1:, :] = np.clip(s[lip_y1:, :] * (1.0 + lip_sat * 2.0), 0, 255)
        hsv_merged = cv2.merge([h, s, v])
        face = cv2.cvtColor(hsv_merged.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = bgr.copy()
    out[y1:y2, x1:x2] = face
    return out
