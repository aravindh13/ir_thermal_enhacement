"""Run from repo root: python -m pytest backend -v"""
import base64, io, math, os, subprocess, sys

import numpy as np, pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import inference as inf
from backend.app import app

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs lifespan -> model loads once
        yield c


def img_bytes(arr, fmt="PNG", mode=None):
    b = io.BytesIO(); Image.fromarray(arr, mode).save(b, fmt); return b.getvalue()


def thermal(h, w):  # smooth blobs + noise, looks roughly like an IR frame
    y, x = np.mgrid[0:h, 0:w]
    a = 90 + 80 * np.exp(-((x - w * .4) ** 2 + (y - h * .5) ** 2) / (w * h * .02)) + rng.normal(0, 6, (h, w))
    return np.clip(a, 0, 255).astype(np.uint8)


def png_from_dataurl(u):
    return Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1])))


def post(c, path, data, name):
    return c.post(path, files={"file": (name, data, "application/octet-stream")})


def test_health(client):
    r = client.get("/api/health").json()
    assert r["model_loaded"] is True and r["status"] == "ok"


def test_model_card(client):
    r = client.get("/api/model").json()
    assert r["parameters"] == 1515265 and r["scale"] == 4
    assert r["weights_bytes"] == os.path.getsize(inf.WEIGHTS)


def test_enhance_shapes(client):
    r = post(client, "/api/enhance", img_bytes(thermal(48, 64)), "a.png")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["metrics"] is None
    assert (j["output"]["width"], j["output"]["height"]) == (256, 192)
    assert png_from_dataurl(j["images"]["input"]["png8"]).size == (64, 48)
    for k in ("bicubic", "edsr"):
        assert png_from_dataurl(j["images"][k]["png8"]).size == (256, 192)
        assert "png16" not in j["images"][k]
    assert j["timing_ms"]["edsr"] > 0


def test_enhance_matches_predict_py(client, tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    src.write_bytes(img_bytes(thermal(48, 64)))
    subprocess.run([sys.executable, "predict.py", inf.WEIGHTS, str(src), str(dst)], cwd=inf.ROOT, check=True,
                   capture_output=True)
    j = post(client, "/api/enhance", src.read_bytes(), "in.png").json()
    ours = np.asarray(png_from_dataurl(j["images"]["edsr"]["png8"]))
    assert np.array_equal(ours, np.asarray(Image.open(dst)))


def test_benchmark_matches_evaluate(client, tmp_path):
    from train_sr import evaluate
    p = tmp_path / "hr.png"
    p.write_bytes(img_bytes(thermal(130, 128)))  # 130 -> cropped to 128, like evaluate()
    r = post(client, "/api/benchmark", p.read_bytes(), "hr.png")
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["output"]["width"], j["output"]["height"]) == (128, 128)
    assert png_from_dataurl(j["images"]["edsr"]["png8"]).size == (128, 128)
    assert any("Cropped" in n for n in j["notes"])
    ref = evaluate(inf.load_model(), [str(p)])
    for ours, theirs in (("bicubic", "bicubic"), ("edsr", "model")):
        m = j["metrics"][ours]
        assert math.isfinite(m["psnr_db"]) and 0 < m["ssim"] <= 1
        assert (m["psnr_db"], m["ssim"]) == (float(ref[theirs][0]), float(ref[theirs][1]))


def tiff16(h=48, w=64):
    a = np.linspace(1000, 40000, h * w).reshape(h, w) + rng.integers(0, 50, (h, w))
    return a.astype(np.uint16)


def test_16bit_tiff_keeps_precision(client):
    a = tiff16()
    data = img_bytes(a, "TIFF")
    d = inf.decode(data, "x.tiff", "enhance")
    assert d.mode.startswith("I;16") and d.bit_depth == 16
    assert len(np.unique(d.arr)) > 256  # no 8-bit quantisation
    j = post(client, "/api/enhance", data, "x.tiff").json()
    assert j["source"]["bit_depth"] == 16 and j["source"]["value_range"] == [float(a.min()), float(a.max())]
    im16 = png_from_dataurl(j["images"]["edsr"]["png16"])
    v = np.asarray(im16)
    assert im16.mode.startswith("I") and im16.size == (256, 192)
    assert a.min() <= v.min() and v.max() <= a.max() and len(np.unique(v)) > 256


def test_16bit_tiff_benchmark(client):
    j = post(client, "/api/benchmark", img_bytes(tiff16(128, 128), "TIFF"), "x.tif").json()
    assert j["metrics"]["edsr"]["psnr_db"] > 0 and "png16" in j["images"]["ground_truth"]


def test_rgb_converted(client):
    rgb = np.stack([thermal(48, 64)] * 3, -1)
    j = post(client, "/api/enhance", img_bytes(rgb, "JPEG"), "c.jpg").json()
    assert j["source"]["converted_to_grayscale"] is True


@pytest.mark.parametrize("path,data,name,status,text", [
    ("/api/enhance", img_bytes(thermal(8, 8), "GIF"), "a.gif", 415, "PNG"),
    ("/api/enhance", img_bytes(thermal(8, 8), "GIF"), "a.png", 415, "GIF"),
    ("/api/enhance", b"not an image at all", "a.png", 400, "Invalid"),
    ("/api/enhance", img_bytes(thermal(520, 700)), "big.png", 413, "640x512"),
    ("/api/enhance", img_bytes(thermal(700, 500)), "tall.png", 413, "640x512"),
    ("/api/benchmark", img_bytes(np.zeros((2000, 3000), np.uint8)), "big.png", 413, "2560x2048"),
    ("/api/benchmark", img_bytes(thermal(16, 16)), "tiny.png", 400, "at least 32"),
])
def test_errors(client, path, data, name, status, text):
    r = post(client, path, data, name)
    assert r.status_code == status and text in r.json()["detail"], r.text


def test_portrait_within_limits_ok(client):
    assert post(client, "/api/enhance", img_bytes(thermal(64, 48)), "p.png").status_code == 200


def test_figures(client):
    r = client.get("/api/figures/0")
    assert r.status_code == (200 if os.path.isfile(os.path.join(inf.ROOT, "figures", "cmp_0.png")) else 404)
    assert client.get("/api/figures/9").status_code == 404
