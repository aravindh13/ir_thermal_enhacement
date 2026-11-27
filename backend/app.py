"""FastAPI layer. Run from repo root: uvicorn backend.app:app"""
import logging, os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import inference as inf
from backend.model_info import model_card

log = logging.getLogger("uvicorn.error")
MAX_UPLOAD = 20 * 1024 * 1024
FIGURES = os.path.join(inf.ROOT, "figures")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = inf.load_model()  # once, not per request
    app.state.card = model_card(app.state.model)
    log.info("EDSR-lite loaded on %s (%d params)", inf.dev, app.state.card["parameters"])
    yield


app = FastAPI(title="IR Thermal Enhancement API", lifespan=lifespan)


@app.exception_handler(inf.ImageError)
def image_error(_: Request, e: inf.ImageError):
    return JSONResponse({"detail": str(e)}, status_code=e.status)


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "model", None) is not None,
            "device": str(inf.dev), "torch_version": torch.__version__}


@app.get("/api/model")
def model(request: Request):
    return request.app.state.card


def _run(request: Request, file: UploadFile, mode: str):
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is larger than 20 MB.")
    d = inf.decode(data, file.filename or "", mode)
    try:
        fn = inf.enhance if mode == "enhance" else inf.benchmark
        out = fn(request.app.state.model, d)
    except inf.ImageError:
        raise
    except Exception:
        log.exception("inference failed for %s", file.filename)
        raise HTTPException(500, "Inference failed.")
    out["filename"] = file.filename
    return out


@app.post("/api/enhance")
def enhance(request: Request, file: UploadFile = File(...)):
    return _run(request, file, "enhance")


@app.post("/api/benchmark")
def benchmark(request: Request, file: UploadFile = File(...)):
    return _run(request, file, "benchmark")


@app.get("/api/figures/{i}")
def figure(i: int):
    path = os.path.join(FIGURES, f"cmp_{i}.png")
    if not 0 <= i <= 4 or not os.path.isfile(path):
        raise HTTPException(404, "Figure not available.")
    return FileResponse(path, media_type="image/png")


# Serve the built frontend (npm run build) from the same process; dev uses the Vite proxy instead.
DIST = os.path.join(inf.ROOT, "frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
