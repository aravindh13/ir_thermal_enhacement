"""Image decoding, bicubic baseline and EDSR-lite inference. No web code here.

Reuses EDSR, psnr and SCALE from train_sr.py so the architecture exists in exactly one place.
8-bit inputs follow predict.py / evaluate() exactly; 16-bit/float inputs keep full precision (no uint8 step)."""
import base64, io, os, sys, threading, time
from dataclasses import dataclass, field

import numpy as np, torch
from PIL import Image, UnidentifiedImageError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train_sr import EDSR, SCALE, dev, psnr  # noqa: E402

WEIGHTS = os.path.join(ROOT, "edsr_x4.pt")
EXTS = {"png", "jpg", "jpeg", "tif", "tiff"}
FORMATS = {"PNG", "JPEG", "TIFF"}
LIMITS = {"enhance": (640, 512), "benchmark": (2560, 2048)}  # (long side, short side)
MIN_BENCHMARK = 32
HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")
Image.MAX_IMAGE_PIXELS = 2560 * 2048 * 2  # decompression-bomb guard above the largest cap

_lock = threading.Lock()  # ponytail: one forward pass at a time; worker queue if many users share a server


class ImageError(Exception):
    status = 400


class UnsupportedFormat(ImageError):
    status = 415


class TooLarge(ImageError):
    status = 413


@dataclass
class Decoded:
    arr: np.ndarray            # float32 in [0, 1], HxW
    pil: Image.Image           # mode L (8-bit path) or F (high-bit path), same values as arr (x255 for L)
    fmt: str
    mode: str
    bit_depth: int
    value_range: tuple | None  # original min/max for high-bit sources
    converted_to_grayscale: bool
    notes: list = field(default_factory=list)

    @property
    def high_bit(self): return self.value_range is not None


def load_model(path=WEIGHTS):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model weights not found at {path}. Copy edsr_x4.pt into the repo root.")
    m = EDSR().to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return m.eval()


def decode(data: bytes, filename: str, mode: str) -> Decoded:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in EXTS:
        raise UnsupportedFormat("Unsupported file type. Use PNG, JPG, JPEG or TIFF.")
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise TooLarge(f"Image exceeds the {mode} size limit {LIMITS[mode][0]}x{LIMITS[mode][1]}.")
    except (UnidentifiedImageError, OSError):
        raise ImageError("Invalid or corrupted image.")
    if img.format not in FORMATS:
        raise UnsupportedFormat(f"File content is {img.format}, not PNG, JPEG or TIFF.")

    long_max, short_max = LIMITS[mode]
    w, h = img.size
    if max(w, h) > long_max or min(w, h) > short_max:
        raise TooLarge(f"Image is {w}x{h}; {mode} mode accepts at most {long_max}x{short_max} "
                       f"(either orientation). Tiled inference is not supported in this version.")

    notes = []
    frames = getattr(img, "n_frames", 1)
    if frames > 1:
        img.seek(0); notes.append(f"Multi-frame TIFF: first frame of {frames} used.")
    try:
        img.load()  # no exif_transpose: keeps pixel-exact parity with predict.py
    except (OSError, SyntaxError, ValueError):
        raise ImageError("Invalid or corrupted image.")

    src_mode = img.mode
    if src_mode in HIGH_BIT_MODES:
        a = np.asarray(img).astype(np.float32)
        if a.ndim != 2:
            raise UnsupportedFormat(f"Unsupported image layout for mode {src_mode}.")
        lo, hi = float(a.min()), float(a.max())
        arr = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        depth = 16 if src_mode.startswith("I;16") else 32
        notes.append(f"{depth}-bit source normalised from [{lo:g}, {hi:g}] to [0, 1] in float32 (no 8-bit quantisation).")
        return Decoded(arr, Image.fromarray(arr, "F"), img.format, src_mode, depth, (lo, hi), False, notes)

    pil = img.convert("L")  # identical to predict.py
    return Decoded(np.asarray(pil, np.float32) / 255, pil, img.format, src_mode, 8, None, src_mode != "L", notes)


def _resize(pil: Image.Image, size) -> np.ndarray:
    """PIL bicubic, same call as training/evaluate(). L stays uint8-rounded like evaluate(); F keeps precision."""
    out = pil.resize(size, Image.BICUBIC)
    return np.asarray(out, np.float32) / (255 if pil.mode == "L" else 1)


def _as_pil(arr: np.ndarray, like: Image.Image) -> Image.Image:
    return Image.fromarray(arr, "F") if like.mode == "F" else Image.fromarray((arr * 255).round().astype(np.uint8))


def _edsr(model, arr: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(arr))[None, None].to(dev)
    with _lock, torch.inference_mode():
        return model(x).clamp(0, 1)[0, 0].cpu().numpy()


def _png(arr: np.ndarray, d: Decoded) -> dict:
    """8-bit display PNG (same rounding as predict.py); plus 16-bit PNG for high-bit sources."""
    def enc(im):
        b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    out = {"png8": enc(Image.fromarray((arr * 255).round().astype(np.uint8)))}
    if d.high_bit:
        lo, hi = d.value_range
        if d.bit_depth == 16 and lo >= 0 and hi <= 65535:
            v = np.round(lo + arr.astype(np.float64) * (hi - lo))  # back to original sensor units
        else:
            v = np.round(arr.astype(np.float64) * 65535)  # float source: full 16-bit span, range in metadata
        out["png16"] = enc(Image.fromarray(v.astype(np.uint16), "I;16"))
    return out


def _source(d: Decoded) -> dict:
    return {"format": d.fmt, "mode": d.mode, "bit_depth": d.bit_depth,
            "value_range": list(d.value_range) if d.value_range else None,
            "normalised": d.high_bit, "converted_to_grayscale": d.converted_to_grayscale}


def _ms(t): return round((time.perf_counter() - t) * 1000, 1)


def enhance(model, d: Decoded) -> dict:
    t0 = time.perf_counter()
    h, w = d.arr.shape
    t = time.perf_counter(); bic = _resize(d.pil, (w * SCALE, h * SCALE)); t_bic = _ms(t)
    t = time.perf_counter(); sr = _edsr(model, d.arr); t_sr = _ms(t)
    t = time.perf_counter()
    images = {"input": _png(d.arr, d), "bicubic": _png(bic, d), "edsr": _png(sr, d)}
    t_enc = _ms(t)
    return {"mode": "enhance", "scale": SCALE, "device": str(dev), "source": _source(d), "notes": d.notes,
            "input": {"width": w, "height": h}, "output": {"width": w * SCALE, "height": h * SCALE},
            "images": images, "metrics": None,
            "timing_ms": {"bicubic": t_bic, "edsr": t_sr, "encode": t_enc, "total": _ms(t0)}}


def benchmark(model, d: Decoded) -> dict:
    """Mirrors train_sr.evaluate(): crop to x4, bicubic down, bicubic up vs EDSR, PSNR/SSIM vs the crop."""
    from skimage.metrics import structural_similarity
    t0 = time.perf_counter()
    H, W = d.arr.shape
    if min(H, W) < MIN_BENCHMARK:
        raise ImageError(f"Benchmark needs at least {MIN_BENCHMARK}x{MIN_BENCHMARK} pixels; got {W}x{H}.")
    W4, H4 = W // SCALE * SCALE, H // SCALE * SCALE
    hr = d.pil.crop((0, 0, W4, H4))
    if (W4, H4) != (W, H):
        d.notes.append(f"Cropped from {W}x{H} to {W4}x{H4} (multiple of {SCALE}), as in evaluate().")
    gt = np.asarray(hr, np.float32) / (255 if hr.mode == "L" else 1)
    lr_pil = hr.resize((W4 // SCALE, H4 // SCALE), Image.BICUBIC)
    lr = np.asarray(lr_pil, np.float32) / (255 if hr.mode == "L" else 1)
    t = time.perf_counter(); bic = _resize(lr_pil, (W4, H4)); t_bic = _ms(t)
    t = time.perf_counter(); sr = _edsr(model, lr); t_sr = _ms(t)
    t = time.perf_counter()
    metrics = {"reference": "uploaded image" + (" (cropped)" if (W4, H4) != (W, H) else ""),
               **{n: {"psnr_db": round(float(psnr(gt, p)), 4),
                      "ssim": round(float(structural_similarity(gt, p, data_range=1.0)), 4)}
                  for n, p in (("bicubic", bic), ("edsr", sr))}}
    t_met = _ms(t)
    t = time.perf_counter()
    images = {"input": _png(lr, d), "bicubic": _png(bic, d), "edsr": _png(sr, d), "ground_truth": _png(gt, d)}
    t_enc = _ms(t)
    return {"mode": "benchmark", "scale": SCALE, "device": str(dev), "source": _source(d), "notes": d.notes,
            "input": {"width": W4 // SCALE, "height": H4 // SCALE}, "output": {"width": W4, "height": H4},
            "images": images, "metrics": metrics,
            "timing_ms": {"bicubic": t_bic, "edsr": t_sr, "metrics": t_met, "encode": t_enc, "total": _ms(t0)}}
