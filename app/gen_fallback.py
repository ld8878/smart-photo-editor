"""Optional generative-style edit via OpenAI Images API when configured."""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image

from app.recipe import Recipe


def _bgr_to_square_png(bgr: np.ndarray, size: int = 1024) -> bytes:
    h, w = bgr.shape[:2]
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    padded = cv2.copyMakeBorder(
        bgr, pad_y, side - h - pad_y, pad_x, side - w - pad_x, cv2.BORDER_REFLECT_101
    )
    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", resized)
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()


def _png_bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码生成图像")
    return img


def _crop_square_back(generated_bgr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    side = max(th, tw)
    pad_x = (side - tw) // 2
    pad_y = (side - th) // 2
    gen_h, gen_w = generated_bgr.shape[:2]
    g = cv2.resize(generated_bgr, (side, side), interpolation=cv2.INTER_AREA)
    return g[pad_y : pad_y + th, pad_x : pad_x + tw]


async def apply_generative_fallback(
    new_bgr: np.ndarray,
    description: str,
    recipe: Union[Recipe, Dict[str, Any]],
) -> tuple[np.ndarray, str]:
    """
    若配置了 OPENAI_API_KEY，调用 `images.edit`（DALL·E 2）做整图编辑。
    否则返回原图与说明。
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return new_bgr, "未设置 OPENAI_API_KEY，已跳过生成式兜底。"

    recipe_str = (
        recipe.model_dump_json() if isinstance(recipe, Recipe) else str(recipe)
    )
    prompt = (
        "Adjust color grading, skin texture, and lighting only; "
        "keep identity and composition. Match this intent: "
        f"{description[:1200]} Context: {recipe_str[:800]}"
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        img_bytes = _bgr_to_square_png(new_bgr)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        mask = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))

        buf_img = io.BytesIO()
        pil_img.save(buf_img, format="PNG")
        buf_img.seek(0)
        buf_mask = io.BytesIO()
        mask.save(buf_mask, format="PNG")
        buf_mask.seek(0)

        model = os.environ.get("OPENAI_IMAGE_EDIT_MODEL", "dall-e-2")
        result = await client.images.edit(
            model=model,
            image=buf_img,
            mask=buf_mask,
            prompt=prompt[:1000],
            n=1,
            size="1024x1024",
        )
        if not result.data:
            return new_bgr, "生成式 API 无返回数据。"

        item = result.data[0]
        if getattr(item, "b64_json", None):
            import base64

            raw = base64.standard_b64decode(item.b64_json)
            out = _png_bytes_to_bgr(raw)
        elif getattr(item, "url", None):
            import httpx

            async with httpx.AsyncClient() as http:
                r = await http.get(item.url)
                r.raise_for_status()
                out = _png_bytes_to_bgr(r.content)
        else:
            return new_bgr, "不支持的生成结果格式。"

        out = _crop_square_back(out, (new_bgr.shape[0], new_bgr.shape[1]))
        return out, f"已使用 {model} images.edit 做生成式编辑。"
    except Exception as exc:  # noqa: BLE001
        return new_bgr, f"生成式兜底失败（已保留参数化结果）：{exc}"
