# IR Thermal Image Enhancement (x4 Super-Resolution)

EDSR-lite CNN that upscales grayscale thermal images 4x, trained on LLVIP infrared. See [REPORT.md](REPORT.md).

| File | Purpose |
|---|---|
| `train_sr.py` | Train and evaluate (PSNR/SSIM vs bicubic). Runs on a Kaggle GPU notebook. |
| `predict.py` | Upscale one image with trained weights. |
| `backend/` | FastAPI service wrapping the trained model (enhance, benchmark, model card). |
| `frontend/` | React + TypeScript web app (upload, compare, performance, model info). |
| `REPORT.md` | Project report. |

```bash
pip install torch pillow numpy scikit-image
python predict.py edsr_x4.pt input.jpg output.png
```

Data: LLVIP infrared images (https://bupt-ai-cz.github.io/LLVIP/), not included in this repo.
`edsr_x4.pt` is not tracked in git; place it in the repo root.

## Web app

Requires Python 3.10+ with torch installed, and Node 20+.

One command (installs whatever is missing, builds the frontend if it changed, starts the server, opens the browser):

```bash
python run.py                      # options: --port 8000, --no-browser
```

Manual equivalent:

```bash
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app:app            # open http://localhost:8000
```

For frontend development, run `uvicorn backend.app:app` and `cd frontend && npm run dev` (Vite proxies `/api` to port 8000).

- **Enhance** mode: the upload is the low-resolution input (max 640x512). Returns bicubic x4 and EDSR-lite x4. No PSNR/SSIM, because there is no ground truth.
- **Benchmark** mode: the upload is the high-resolution reference (max 2560x2048). It is bicubic-downscaled x4 and reconstructed, then scored exactly as `train_sr.evaluate()` does.
- 16-bit and float TIFFs are normalised in float32 (no 8-bit step). Results can also be downloaded as 16-bit PNGs in the original sensor units.
- Images are processed in memory and never stored. Run history stays in the browser's localStorage.

Tests: `python -m pytest backend -v`. These include pixel parity with `predict.py` and metric parity with `train_sr.evaluate()`.
