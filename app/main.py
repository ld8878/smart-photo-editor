"""FastAPI: analyze image pair, apply recipe to new image, optional generative fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.analysis import analyze_pair
from app.gen_fallback import apply_generative_fallback
from app.pipeline import apply_full_pipeline, decode_upload, encode_png
from app.recipe import Recipe

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="BatchEditPhoto", version="0.1.0")

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/analyze")
async def api_analyze(
    original: UploadFile = File(..., description="原图"),
    edited: UploadFile = File(..., description="修后图"),
    use_vlm: str = Form("true"),
) -> dict[str, Any]:
    o_bytes = await original.read()
    e_bytes = await edited.read()
    try:
        o = decode_upload(o_bytes)
        e = decode_upload(e_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    use = use_vlm.lower() in ("1", "true", "yes", "on")
    return await analyze_pair(o, e, use_vlm=use)


@app.post("/api/apply")
async def api_apply(
    new_image: UploadFile = File(..., description="要套用配方的新图"),
    recipe_json: str = Form(..., description="JSON 配方"),
    ref_original: Optional[UploadFile] = File(None),
    ref_edited: Optional[UploadFile] = File(None),
) -> Response:
    try:
        recipe = Recipe.model_validate(json.loads(recipe_json))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"recipe 无效: {exc}") from exc
    n_bytes = await new_image.read()
    try:
        new_bgr = decode_upload(n_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ro = re = None
    if ref_original and ref_edited:
        try:
            ro = decode_upload(await ref_original.read())
            re = decode_upload(await ref_edited.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    out = apply_full_pipeline(new_bgr, recipe, ro, re)
    return Response(content=encode_png(out), media_type="image/png")


@app.post("/api/apply-generative")
async def api_apply_generative(
    new_image: UploadFile = File(...),
    recipe_json: str = Form(...),
    description: str = Form(""),
) -> Response:
    try:
        recipe = Recipe.model_validate(json.loads(recipe_json))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"recipe 无效: {exc}") from exc
    try:
        new_bgr = decode_upload(await new_image.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out, _msg = await apply_generative_fallback(new_bgr, description or "风格迁移", recipe)
    return Response(content=encode_png(out), media_type="image/png")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
