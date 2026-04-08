"""Histogram matching and optional 3D LUT-style color transfer between image pairs."""

from __future__ import annotations

import cv2
import numpy as np


def _match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    s_flat = source.ravel()
    r_flat = reference.ravel()
    s_vals, s_counts = np.unique(s_flat, return_counts=True)
    r_vals, r_counts = np.unique(r_flat, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.float64) / s_flat.size
    r_cdf = np.cumsum(r_counts).astype(np.float64) / r_flat.size
    interp_values = np.interp(s_vals, r_vals, r_vals)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[s_vals.astype(int)] = np.clip(interp_values, 0, 255).astype(np.uint8)
    return lookup[source]


def histogram_match_bgr(source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """Match each BGR channel histogram of source to reference."""
    if source_bgr.shape != reference_bgr.shape:
        reference_bgr = cv2.resize(
            reference_bgr, (source_bgr.shape[1], source_bgr.shape[0]), interpolation=cv2.INTER_AREA
        )
    out = np.empty_like(source_bgr)
    for c in range(3):
        out[:, :, c] = _match_channel(source_bgr[:, :, c], reference_bgr[:, :, c])
    return out


def build_color_transfer_pair(
    original_bgr: np.ndarray, edited_bgr: np.ndarray, max_side: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Resize pair for stable histogram matching."""
    o, e = original_bgr.copy(), edited_bgr.copy()
    h, w = o.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        o = cv2.resize(o, (nw, nh), interpolation=cv2.INTER_AREA)
        e = cv2.resize(e, (nw, nh), interpolation=cv2.INTER_AREA)
    if o.shape != e.shape:
        e = cv2.resize(e, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
    return o, e


def apply_histogram_match_with_strength(
    target_bgr: np.ndarray, ref_original_bgr: np.ndarray, ref_edited_bgr: np.ndarray, strength: float
) -> np.ndarray:
    """
    Apply the color change from (ref_original -> ref_edited) onto target.
    strength 0..1 blends between target and fully matched result.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return target_bgr

    ro, re = build_color_transfer_pair(ref_original_bgr, ref_edited_bgr)
    # Transfer: match target's histogram to edited's distribution relative to original
    # Simple approach: hm = hist_match(target, re) blended; or match(target, re) after aligning means
    matched = histogram_match_bgr(target_bgr, re)
    # Also pull toward "delta" style: difference between re and ro applied in Lab
    lab_t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ro = cv2.cvtColor(ro, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_re = cv2.cvtColor(re, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab_re - lab_ro
    # Resize delta to target if needed
    if delta.shape[:2] != lab_t.shape[:2]:
        delta = cv2.resize(delta, (lab_t.shape[1], lab_t.shape[0]), interpolation=cv2.INTER_LINEAR)
    lab_adjusted = lab_t + delta * strength
    lab_adjusted[..., 0] = np.clip(lab_adjusted[..., 0], 0, 255)
    lab_adjusted[..., 1] = np.clip(lab_adjusted[..., 1], 0, 255)
    lab_adjusted[..., 2] = np.clip(lab_adjusted[..., 2], 0, 255)
    delta_style = cv2.cvtColor(lab_adjusted.astype(np.uint8), cv2.COLOR_LAB2BGR)

    blended = cv2.addWeighted(matched, strength, target_bgr, 1.0 - strength, 0)
    blended2 = cv2.addWeighted(delta_style, strength, target_bgr, 1.0 - strength, 0)
    return cv2.addWeighted(blended, 0.5, blended2, 0.5, 0)
