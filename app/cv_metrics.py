"""Estimate global recipe deltas from two aligned BGR images."""

from __future__ import annotations

import cv2
import numpy as np

from app.recipe import GlobalRecipe, Recipe


def _resize_pair(
    a: np.ndarray, b: np.ndarray, max_side: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    h, w = a.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
    return a, b


def estimate_global_recipe_from_pair(original_bgr: np.ndarray, edited_bgr: np.ndarray) -> GlobalRecipe:
    """Heuristic mapping from Lab/HSV statistics to GlobalRecipe fields."""
    o, e = _resize_pair(original_bgr.copy(), edited_bgr.copy())
    if o.shape != e.shape:
        e = cv2.resize(e, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)

    o_lab = cv2.cvtColor(o, cv2.COLOR_BGR2LAB).astype(np.float32)
    e_lab = cv2.cvtColor(e, cv2.COLOR_BGR2LAB).astype(np.float32)
    ol, oa, ob = cv2.split(o_lab)
    el, ea, eb = cv2.split(e_lab)

    mean_l_o, mean_l_e = float(np.mean(ol)), float(np.mean(el))
    std_l_o, std_l_e = float(np.std(ol)) + 1e-6, float(np.std(el)) + 1e-6

    exposure = np.clip((mean_l_e - mean_l_o) / 40.0, -1.2, 1.2)
    contrast = np.clip(std_l_e / std_l_o, 0.4, 2.2)

    o_hsv = cv2.cvtColor(o, cv2.COLOR_BGR2HSV).astype(np.float32)
    e_hsv = cv2.cvtColor(e, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, osat, _ = cv2.split(o_hsv)
    _, esat, _ = cv2.split(e_hsv)
    sat_ratio = (float(np.mean(esat)) + 1e-3) / (float(np.mean(osat)) + 1e-3)
    saturation = np.clip(sat_ratio, 0.5, 2.0)

    # temperature: warmer = lower a in edited vs original? Actually warm adds red/yellow -> Lab a+, b+
    da = float(np.mean(ea) - np.mean(oa))
    db = float(np.mean(eb) - np.mean(ob))
    temperature = np.clip(-(da * 0.02 + db * 0.01), -1.0, 1.0)

    # crude highlights/shadows from tails
    shadow_lift = float(np.percentile(el, 15) - np.percentile(ol, 15)) / 40.0
    highlight_adj = float(np.percentile(el, 85) - np.percentile(ol, 85)) / 40.0
    shadows = np.clip(shadow_lift, -1.0, 1.0)
    highlights = np.clip(highlight_adj, -1.0, 1.0)

    return GlobalRecipe(
        exposure=float(exposure),
        contrast=float(contrast),
        saturation=float(saturation),
        temperature=float(temperature),
        highlights=float(highlights * 0.5),
        shadows=float(shadows * 0.5),
        grain=0.0,
        vignette=0.0,
    )


def recipe_from_pair(original_bgr: np.ndarray, edited_bgr: np.ndarray) -> Recipe:
    g = estimate_global_recipe_from_pair(original_bgr, edited_bgr)
    return Recipe(global_=g)
