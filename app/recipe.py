"""Structured recipe schema and OpenCV-based application."""

from __future__ import annotations

from typing import Any, Callable, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class GlobalRecipe(BaseModel):
    exposure: float = Field(0.0, ge=-1.5, le=1.5, description="Roughly -1..1, brighten/darken")
    contrast: float = Field(1.0, ge=0.3, le=2.5)
    saturation: float = Field(1.0, ge=0.0, le=2.5)
    temperature: float = Field(
        0.0, ge=-1.0, le=1.0, description="Negative = warmer, positive = cooler (Lab a/b shift)"
    )
    highlights: float = Field(0.0, ge=-1.0, le=1.0)
    shadows: float = Field(0.0, ge=-1.0, le=1.0)
    grain: float = Field(0.0, ge=0.0, le=1.0)
    vignette: float = Field(0.0, ge=0.0, le=1.0)


class ColorGradingRecipe(BaseModel):
    use_lut: bool = False
    lut_id: Optional[str] = None
    histogram_match_strength: float = Field(0.0, ge=0.0, le=1.0)


class FaceRecipe(BaseModel):
    skin_smooth_strength: float = Field(0.0, ge=0.0, le=1.0)
    skin_warmth: float = Field(0.0, ge=-1.0, le=1.0)
    lip_saturation: float = Field(0.0, ge=-0.5, le=0.5)
    face_slim_approx: float = Field(0.0, ge=0.0, le=1.0)
    notes: str = ""


class GeometryRecipe(BaseModel):
    rotation_deg: float = 0.0
    crop_center_x: float = 0.5
    crop_center_y: float = 0.5
    crop_scale: float = 1.0


class Recipe(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    version: int = 1
    global_: GlobalRecipe = Field(alias="global", default_factory=GlobalRecipe)
    color_grading: ColorGradingRecipe = Field(default_factory=ColorGradingRecipe)
    face: FaceRecipe = Field(default_factory=FaceRecipe)
    geometry: GeometryRecipe = Field(default_factory=GeometryRecipe)

    def model_dump_for_json(self) -> dict[str, Any]:
        d = self.model_dump(by_alias=True)
        return d


def _adjust_exposure_contrast_lab(img_bgr: np.ndarray, exposure: float, contrast: float) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    # exposure: add to L channel (0-100 scale in OpenCV LAB)
    l = l + exposure * 25.0
    # contrast around mid gray ~ 128 for 8-bit L stored 0-255 in cv2? Actually L is 0-255 in 8bit
    mid = 128.0
    l = (l - mid) * contrast + mid
    l = np.clip(l, 0, 255)
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _adjust_saturation(img_bgr: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation, 0, 255)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _adjust_temperature(img_bgr: np.ndarray, temperature: float) -> np.ndarray:
    """temperature: negative = warmer (more a+), positive = cooler (more b-)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a = a - temperature * 8.0
    b = b + temperature * 5.0
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _shadows_highlights(img_bgr: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, bch = cv2.split(lab)
    dark = np.clip((128.0 - l) / 128.0, 0, 1)
    bright = np.clip((l - 128.0) / 127.0, 0, 1)
    l = l + shadows * 20.0 * dark + highlights * 20.0 * bright
    l = np.clip(l, 0, 255)
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), bch.astype(np.uint8)])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _grain(img_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.001:
        return img_bgr
    noise = np.random.randn(*img_bgr.shape).astype(np.float32) * strength * 15
    out = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _vignette(img_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.001:
        return img_bgr
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / (np.sqrt(cx**2 + cy**2) + 1e-6)
    mask = 1.0 - np.clip(r * strength * 1.2, 0, 1) * strength
    mask = mask[..., np.newaxis]
    return np.clip(img_bgr.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def apply_global_recipe(img_bgr: np.ndarray, g: GlobalRecipe) -> np.ndarray:
    out = img_bgr.copy()
    out = _adjust_exposure_contrast_lab(out, g.exposure, g.contrast)
    out = _adjust_saturation(out, g.saturation)
    out = _adjust_temperature(out, g.temperature)
    out = _shadows_highlights(out, g.shadows, g.highlights)
    out = _grain(out, g.grain)
    out = _vignette(out, g.vignette)
    return out


def apply_recipe(
    img_bgr: np.ndarray,
    recipe: Recipe,
    lut_apply_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    out = apply_global_recipe(img_bgr, recipe.global_)
    if recipe.color_grading.use_lut and lut_apply_fn is not None:
        out = lut_apply_fn(out)
    return out
