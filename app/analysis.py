"""Combine CV metrics, face hints, and optional VLM for natural-language summary + recipe."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.cv_metrics import estimate_global_recipe_from_pair, recipe_from_pair
from app.face import diff_face_recipe
from app.recipe import ColorGradingRecipe, FaceRecipe, Recipe


def _bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()


def _cv_summary_lines(
    original_bgr: np.ndarray, edited_bgr: np.ndarray, g: Any
) -> list[str]:
    o_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    e_lab = cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dl = float(np.mean(cv2.split(e_lab)[0]) - np.mean(cv2.split(o_lab)[0]))
    lines = [
        f"整体亮度变化约 {dl:+.1f}（Lab L 均值差，正值偏亮）。",
        f"估计曝光调整系数 exposure≈{g.exposure:.2f}，对比度 contrast≈{g.contrast:.2f}。",
        f"饱和度比例约 {g.saturation:.2f}，色温倾向 temperature≈{g.temperature:.2f}（负偏暖、正偏冷）。",
    ]
    return lines


async def analyze_pair(
    original_bgr: np.ndarray, edited_bgr: np.ndarray, use_vlm: bool = True
) -> dict[str, Any]:
    base = recipe_from_pair(original_bgr, edited_bgr)
    g = estimate_global_recipe_from_pair(original_bgr, edited_bgr)
    base.global_ = g

    fd = diff_face_recipe(original_bgr, edited_bgr)
    base.face = FaceRecipe(
        skin_smooth_strength=fd["skin_smooth_strength"],
        skin_warmth=fd["skin_warmth"],
        lip_saturation=fd["lip_saturation"],
        face_slim_approx=fd["face_slim_approx"],
        notes=str(fd.get("notes", "")),
    )
    base.color_grading = ColorGradingRecipe(
        use_lut=True,
        lut_id="pair_histogram",
        histogram_match_strength=0.7,
    )

    cv_lines = _cv_summary_lines(original_bgr, edited_bgr, g)
    description = "【自动分析】\n" + "\n".join(cv_lines)
    if base.face.notes:
        description += "\n" + base.face.notes

    recipe_obj: Recipe = base
    if use_vlm and os.environ.get("OPENAI_API_KEY"):
        try:
            vlm_text, vlm_recipe_patch = await _call_openai_vision(
                original_bgr, edited_bgr, recipe_obj.model_dump_for_json()
            )
            description = vlm_text + "\n\n" + description
            if vlm_recipe_patch:
                recipe_obj = merge_recipe_patch(recipe_obj, vlm_recipe_patch)
        except Exception as exc:  # noqa: BLE001
            description += f"\n\n（VLM 增强失败，已回退纯 CV：{exc}）"

    return {
        "description": description,
        "recipe": recipe_obj.model_dump_for_json(),
    }


async def _call_openai_vision(
    original_bgr: np.ndarray, edited_bgr: np.ndarray, recipe_hint: dict[str, Any]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    o_b64 = base64.standard_b64encode(_bgr_to_png_bytes(original_bgr)).decode("ascii")
    e_b64 = base64.standard_b64encode(_bgr_to_png_bytes(edited_bgr)).decode("ascii")

    system = (
        "你是图像修图分析助手。用户给出「原图」和「修后图」。"
        "你必须只输出一个 JSON 对象，包含："
        ' "summary": 中文详细分析（全局曝光/对比/饱和度/色温/滤镜感；'
        "若有人像则写肤色、肤质、妆容、发型；构图与裁剪），"
        "避免身体羞辱或医疗诊断式表述；"
        ' "recipe_patch": 可选微调，结构与 {"global":{"exposure":0,"contrast":1,...}} 一致，'
        "字段均为数字；若无法估计则 recipe_patch 为 null。"
    )
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "以下为原图（第一张）与修后图（第二张）。当前 CV 估计配方如下："},
        {"type": "text", "text": json.dumps(recipe_hint, ensure_ascii=False)[:8000]},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{o_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{e_b64}"}},
    ]

    resp = await client.chat.completions.create(
        model=os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    text = data.get("summary") or data.get("description") or data.get("text") or ""
    if not text:
        text = str(data)

    patch = data.get("recipe_patch")
    if patch is None and data.get("global"):
        patch = {"global": data.get("global")}

    out_patch: Optional[Dict[str, Any]] = None
    if isinstance(patch, dict):
        out_patch = patch

    return text, out_patch


def merge_recipe_patch(base: Recipe, patch: Optional[Dict[str, Any]]) -> Recipe:
    if not patch:
        return base
    d = base.model_dump(by_alias=True)
    if "global" in patch and isinstance(patch["global"], dict):
        g = {**d["global"], **patch["global"]}
        d["global"] = g
    if "face" in patch and isinstance(patch["face"], dict):
        d["face"] = {**d["face"], **patch["face"]}
    if "color_grading" in patch and isinstance(patch["color_grading"], dict):
        d["color_grading"] = {**d["color_grading"], **patch["color_grading"]}
    return Recipe.model_validate(d)
