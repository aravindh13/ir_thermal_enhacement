# Phase 02: IR Thermal Enhancement Web App, Implementation Plan

Status: **implemented (Phases 2.1–2.6).** Approved decisions: Enhance cap 640×512, Benchmark cap 2560×2048, no tiling, history in localStorage only.

---

## 1. Existing project audit

### 1.1 Files

| File | What it is | Notes for the web app |
|---|---|---|
| `train_sr.py` | Kaggle training script. Defines `EDSR`, `Res`, `psnr`, `evaluate`, `dev`, `SCALE=4`. | Importing is safe (training only runs under `__main__`). The backend imports `EDSR` and `psnr` from here, so there is one copy of the architecture. |
| `predict.py` | CLI: `python predict.py edsr_x4.pt in.jpg out.png`. | Reads `sys.argv` at module level, so it **cannot be imported**. The backend repeats its 6 inference lines (load L-mode, /255, `model(x).clamp(0,1)`, ×255 round) and leaves the file untouched. |
| `edsr_x4.pt` | `state_dict`, 6.1 MB, **1,515,265 parameters**. | Loaded once at startup with `map_location`. Read-only. Git-ignored. |
| `REPORT.md` | Results, training config, limitations. | Source of every static number shown in the UI. |
| `README.md` | Usage. | Gets a "Web app" section in the last phase. |
| `figures/cmp_0..4.png` | 3840×1024 strips (bicubic \| model \| ground truth) from the final evaluation. | Can be shown on the Model page as real reference examples. Git-ignored. |
| `data/` | `llvip_infrared.zip`, `raw/llvip/`. | Git-ignored. Some test images are useful for local manual testing, but the app never reads `data/` at runtime. |
| `test_lr.png`, `test_sr.png` | 320×256 → 1280×1024 output of a `predict.py` run. | Untracked. Shows the CLI works locally on CPU. |

Uncommitted change: `train_sr.py` now imports `skimage` lazily inside `evaluate()`, so importing `EDSR` does not need scikit-image.

### 1.2 How the model behaves (facts the app has to respect)

- **Grayscale only:** 1 input channel. Colour input is converted to L (luminance), the same as `predict.py`.
- **×4 only:** `SCALE = 4` is hard-coded, and so are the two PixelShuffle ×2 stages. The UI offers only ×4.
- **Output = learned residual + bicubic ×4 of the input** (global skip).
- **Trained on synthetic low-resolution images:** each LR image was a bicubic ×4 downscale of an LLVIP 1280×1024 infrared frame. Real low-resolution sensor images were never seen in training, so they are out of distribution.
- **Measured locally (CPU, torch 2.8.0+cpu):**

  | LR input | SR output | Time |
  |---|---|---|
  | 160×128 | 640×512 | 0.5 s |
  | 320×256 | 1280×1024 | 1.0 s |
  | 640×512 | 2560×2048 | 3.7 s |

  A full 1280×1024 input (5120×4096 output) would take roughly 15 s and several GB of activation memory, because the upsampler holds 256-channel maps at ×2 resolution. **The input size has to be capped.**

### 1.3 Environment

- Python 3.13.7, torch 2.8.0 **CPU build**, Pillow 10.4, NumPy 2.4.4, Node 24.8.
- **scikit-image is not installed** locally, but SSIM needs it. It is already a documented project dependency in the README.

---

## 2. Key design decision: two processing modes

PSNR and SSIM need a ground-truth image. An uploaded image either is that ground truth or it isn't, so the workspace offers two explicit modes:

| Mode | Treats the upload as | Pipeline | Metrics |
|---|---|---|---|
| **Enhance** (default) | A low-resolution image | upload → bicubic ×4 and EDSR ×4 | Time and dimensions only. **No PSNR/SSIM**, with the label "No ground-truth reference: quality metrics unavailable". |
| **Benchmark** | A high-resolution ground truth | crop to a multiple of 4 → bicubic ↓4 = LR → bicubic ×4 and EDSR ×4 → compare both with the original | PSNR and SSIM for bicubic and EDSR, plus time. This is exactly the protocol in `evaluate()`, so the numbers are comparable with REPORT.md. |

Every metric in the UI comes from one of three sources: a computation on the current image, the measured time and size, or REPORT.md (labelled "Reported, 100 LLVIP test images"). Nothing is estimated or invented.

**Size limits (config constants in one place):**
- Enhance: LR input of at most 640×512 pixels (327,680 px).
- Benchmark: HR input of at most 2560×2048 pixels.
- Larger uploads are rejected with HTTP 413 and the allowed size. A native LLVIP 1280×1024 frame works in Benchmark mode (LR 320×256, about 1 s).
- Upload size: at most 20 MB.

---

## 3. Application architecture

```
Browser (React SPA)  ──HTTP/JSON──▶  FastAPI (uvicorn)  ──in-process──▶  PyTorch EDSR (loaded once)
   UI state, history                  validate, decode,                     imports EDSR from train_sr.py
   (localStorage)                     bicubic, metrics                      weights: ../edsr_x4.pt
```

- **Frontend:** only presentation and interaction. No image maths beyond display (zoom, pan, slider).
- **Backend:** validation, decoding, bicubic, inference, metrics, encoding. Stateless: images are processed in memory and never written to disk.
- **ML code:** stays where it is. The backend adds `sys.path` to the repo root and does `from train_sr import EDSR, psnr`. No changes to `train_sr.py`, `predict.py` or the weights.
- **Dev:** the Vite dev server proxies `/api` to `localhost:8000`.
- **Prod (optional):** `vite build`, and FastAPI serves `frontend/dist` as static files, so it runs as a single process.

### 3.1 Folder structure

```
ir_thermal_image_enhacement/
├── train_sr.py            (unchanged)
├── predict.py             (unchanged)
├── edsr_x4.pt             (unchanged, git-ignored)
├── REPORT.md / README.md  (README gets a "Web app" section)
├── figures/               (unchanged)
├── backend/
│   ├── app.py             FastAPI app, routes, lifespan model load
│   ├── inference.py       load_model(), decode_image(), enhance(), benchmark()
│   ├── model_info.py      static facts copied from REPORT.md and train_sr.py
│   ├── requirements.txt   fastapi, uvicorn, python-multipart, pillow, numpy, scikit-image (torch preinstalled)
│   └── test_api.py        pytest + FastAPI TestClient
└── frontend/
    ├── package.json, vite.config.ts, tailwind config, components.json (shadcn)
    └── src/
        ├── main.tsx, App.tsx          shell + page switch
        ├── api.ts                     typed fetch wrappers + response types
        ├── history.ts                 localStorage activity log
        ├── components/ui/             shadcn generated (button, card, tabs, badge, progress, table, toggle-group, tooltip, alert)
        ├── components/
        │   ├── AppShell.tsx           sidebar nav + header
        │   ├── UploadDropzone.tsx
        │   ├── CompareSlider.tsx      before/after slider
        │   ├── ZoomPanViewer.tsx      synchronised zoom/pan for 1–3 images
        │   ├── MetricCard.tsx
        │   └── charts/MetricBarChart.tsx
        └── pages/
            ├── Dashboard.tsx
            ├── Workspace.tsx
            ├── Compare.tsx
            ├── Performance.tsx
            └── ModelInfo.tsx
```

Five files in the backend and no database, queue or storage layer. Each extra layer can be added later if a real need for it shows up.

---

## 4. Backend API design

Base path `/api`. All errors use `{"detail": "<human-readable message>"}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, model_loaded, device, torch_version}` |
| GET | `/api/model` | Static model card: architecture, parameter count (computed at startup), weights file size (`os.path.getsize`), training config and reported results from REPORT.md, dataset facts, limitations, size limits, available methods. |
| POST | `/api/enhance` | multipart `file`. Enhance mode. |
| POST | `/api/benchmark` | multipart `file`. Benchmark mode. |

### 4.1 `/api/enhance` response

```json
{
  "mode": "enhance",
  "scale": 4,
  "input":   {"width": 320, "height": 256, "format": "PNG", "mode_in": "RGB", "converted_to_grayscale": true},
  "output":  {"width": 1280, "height": 1024},
  "images":  {"input": "data:image/png;base64,...", "bicubic": "...", "edsr": "..."},
  "timing_ms": {"decode": 4, "bicubic": 6, "edsr": 980, "total": 1010},
  "device": "cpu",
  "metrics": null
}
```

### 4.2 `/api/benchmark` response

The response has the same shape as Enhance, plus `images.ground_truth`, `images.input` = the synthetic LR image, and:

```json
"metrics": {
  "reference": "uploaded image (cropped to multiple of 4)",
  "bicubic": {"psnr_db": 34.9, "ssim": 0.961},
  "edsr":    {"psnr_db": 40.1, "ssim": 0.984}
}
```

PSNR comes from `train_sr.psnr`. SSIM uses `skimage.metrics.structural_similarity(..., data_range=1.0)`, the same call as `evaluate()`. Bicubic uses PIL `Image.BICUBIC`, the same as training and evaluation.

Images are returned inline as base64 PNG, so one request produces everything to display and nothing is stored on the server. A 2560×2048 grayscale PNG is a few MB, which is acceptable with the size caps. If payloads become a problem, the upgrade is to return binary images from separate endpoints.

### 4.3 Validation and errors

| Case | Status |
|---|---|
| Missing file, or an extension or MIME type outside png/jpg/jpeg/tif/tiff | 415 |
| Pillow cannot decode it (`UnidentifiedImageError`, truncated file) | 400 "Invalid or corrupted image" |
| Larger than 20 MB, or more pixels than the mode's limit | 413 with the limit |
| Benchmark image smaller than 32×32 | 400 |
| Inference exception (including OOM) | 500 "Inference failed", logged server-side with no stack trace sent to the client |

- Decoding uses `Image.open`, then `ImageOps.exif_transpose`. `Image.MAX_IMAGE_PIXELS` is set to the cap to guard against decompression bombs.
- **16-bit or float TIFF** (common for raw thermal data, modes `I;16`, `I` and `F`): this is min–max normalised to [0, 1] **in float32, with no 8-bit step**, so every sensor level reaches the model. Plain `.convert("L")` would clip it. The response reports `source.bit_depth` and `source.value_range`. Each image also gets a 16-bit PNG (`png16`) mapped back to the original sensor units, next to the 8-bit display PNG (`png8`). *(Implemented in Phase 2.1.)*
- Multi-frame TIFF: only the first frame is used, and this is reported in the response.

### 4.4 Model integration

```python
# backend/inference.py (sketch, not final code)
sys.path.insert(0, REPO_ROOT)
from train_sr import EDSR, psnr

def load_model(path):          # called once in FastAPI lifespan
    m = EDSR().to(dev); m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval()
    return m
```

- The model is stored on `app.state.model`, so every request reuses it.
- Inference runs under `torch.inference_mode()`, with a `threading.Lock` around the forward pass. PyTorch CPU already uses every core, so running two forward passes in parallel only adds memory pressure. *(ponytail: global lock serialises requests; a worker queue would help if several users share one server.)*
- The route handlers are plain `def`, not `async def`, so FastAPI runs them in its threadpool and they don't block the event loop.
- `torch.set_num_threads` is left at its default.

---

## 5. UI/UX design specification

### 5.1 Visual language
- **Theme:** light. Background `zinc-50`, cards white with a `zinc-200` border and `shadow-sm`. Text `zinc-900` and `zinc-500`.
- **Accent:** one restrained colour (`indigo-600`) for primary actions and the EDSR series. Bicubic is always neutral `zinc-400`. Ground truth is `emerald-600`. These colours stay the same in charts, badges and image labels.
- **Typography:** Inter (or the system UI stack) for text and a monospace font for numbers. Scale: 28/20/16/14/12. Numbers use tabular figures.
- **Spacing:** a 4-px grid, 24-px card padding, 24-px gaps.
- **Motion:** 150–200 ms fades and opacity transitions only. Progress uses a shadcn `Progress`. No gradients or glassmorphism.
- **Layout:** left sidebar (240 px, collapses to a top bar under 1024 px) and a content area with a maximum width of 1280 px.
- **Thermal images** are shown in grayscale exactly as produced, with `image-rendering: pixelated` when zoomed so artefacts stay visible and nothing is hidden by smoothing.
- **Accessibility:** visible focus rings, a keyboard-operable slider (arrow keys), alt text on images, and contrast of at least 4.5:1.

### 5.2 Screens

**A. Dashboard**
- Header: project title and a one-line description.
- Row of 4 `MetricCard`s: *Model* EDSR-lite ×4 · *Parameters* 1.52 M · *Training* 87 epochs / 120 min / T4 · *Dataset* LLVIP IR, 10,000 train images.
- Card **Reported results** (from REPORT.md, labelled as such): a small bar chart of PSNR and SSIM for bicubic vs EDSR-lite.
- Card **Available methods**: Bicubic ×4 (baseline) and EDSR-lite ×4, each with a status badge from `/api/health`.
- Card **Recent activity**: the last 10 runs from localStorage (time, file name, mode, dimensions, and EDSR time). It is empty until the user runs something.

**B. Enhancement Workspace**
- Left (about 360 px):
  - Dropzone (drag-drop or click, with accepted types listed)
  - Mode toggle (Enhance / Benchmark, each with a one-line explanation)
  - Method shown read-only as "EDSR-lite ×4 + bicubic baseline"
  - Size limits
  - **Enhance** button
- Right: a preview of the original with its dimensions and format. When the run finishes, it shows a status (Uploading, Processing, Done or Error), the three result thumbnails and download buttons (EDSR PNG and bicubic PNG, made from the data URLs on the client).
- The status is indeterminate, because a single request gives no real progress percentage and a fake one is avoided. Elapsed time is shown live.
- The **Open in Compare** button passes the result through React state.

**C. Comparison**
- Tabs: **Slider** | **Side-by-side**.
  - Slider: two layers, Bicubic vs EDSR by default, chosen with toggles (Original-upscaled-nearest / Bicubic / EDSR / Ground truth if present).
  - Side-by-side: 2 or 3 panels.
- Zoom (buttons, wheel, 1×–8×) and pan (drag) are **synchronised across panels**, so the same region is compared.
- Each panel shows a caption with the method name and its W×H.
- Without a result, it shows an empty state linking to the Workspace.

**D. Performance Analysis**
- **This run:**
  - Timing bar chart (decode / bicubic / EDSR)
  - Resolution card (input → output, ×4)
  - Device
  - Model size (MB and parameter count)
  - With Benchmark only: a PSNR/SSIM bar chart and a table for this image. Otherwise an alert explains why metrics are absent.
- **Session:** a table of runs in localStorage, with a line chart of EDSR time against input megapixels (real measurements only).
- **Reported:** the REPORT.md table and a chart, clearly labelled with "100 LLVIP test images, synthetic bicubic LR".

**E. Model Information**
- An architecture diagram as inline SVG or HTML blocks: Head conv 1→64, then 16 × ResBlock (conv-ReLU-conv, ×0.1 residual), then body conv, then the global feature skip, then Upsampler [conv 64→256, PixelShuffle×2] ×2, then conv 64→1, plus a bicubic ×4 global skip.
- Training configuration table: L1, Adam 2e-4, batch 16, 192 px HR / 48 px LR patches, horizontal flip, 120 min, 87 epochs, T4.
- LLVIP card: 12,025 train and 3,463 test images, 1280×1024, infrared only, 10 k subset.
- Reference figures `figures/cmp_*.png`, served by `/api/figures/{i}` if present and hidden if the folder is absent.
- **Limitations**, from REPORT §8 and the size cap:
  - Trained on synthetic bicubic LR images
  - A single dataset and a single scale
  - Grayscale only
  - Real sensor images may look worse

### 5.3 Component hierarchy

```
App
└─ AppShell (Sidebar, Header)
   ├─ Dashboard ─ MetricCard×4, MetricBarChart, MethodList, ActivityTable
   ├─ Workspace ─ UploadDropzone, ModeToggle, ImagePreview, StatusPanel, ResultThumbs, DownloadButtons
   ├─ Compare ─ Tabs ─ CompareSlider(ZoomPanViewer) | SideBySide(ZoomPanViewer×n)
   ├─ Performance ─ MetricCard, MetricBarChart, TimingChart, RunsTable, ReportedTable
   └─ ModelInfo ─ ArchDiagram, ConfigTable, DatasetCard, FigureGallery, LimitationsAlert
```

**State:** a single `useState` in `App` holds `{page, lastResult}`, and `history.ts` holds the run log. Redux and react-router are not needed for five pages. *(ponytail: hash-based page state; add react-router when deep links are needed.)*

---

## 6. Dependencies

### Backend, added to `backend/requirements.txt`
| Package | Why |
|---|---|
| fastapi, uvicorn | API server (FastAPI appears to be installed already) |
| python-multipart | Needed by FastAPI for file uploads |
| scikit-image | SSIM, the same function as training. **Not installed locally.** |
| pillow, numpy, torch | Already present |

**OpenCV is not needed:** Pillow already does the decoding and bicubic resizing that match training. Using cv2's bicubic would give slightly different baselines than REPORT.md.

### Frontend
- Vite, React 18/19, TypeScript, Tailwind, shadcn/ui (with Radix as its dependencies) and Recharts, as specified.
- `lucide-react` is used for icons; it comes with shadcn.
- No zoom, slider or dropzone libraries: these are about 60–120 lines each with pointer events and CSS transforms.

---

## 7. Risks and limitations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU-only inference; large inputs are slow or run out of memory | A request that takes 15 s or more, or crashes | Pixel caps (§2), measured timing shown, OOM caught as 500. *Later:* tile-based inference. |
| Users upload real low-resolution thermal images, not bicubic-downsampled ones | The output may look less impressive than the reported numbers | State this on the Workspace and Model pages. Benchmark mode reproduces the honest protocol. |
| 16-bit radiometric TIFF | Wrong contrast if handled naively | Min–max normalisation, flagged in the response |
| RGB or false-colour (palette) thermal images | Luminance of a colour palette is not temperature | Convert to L (same as `predict.py`), flag `converted_to_grayscale`, mention it in the UI |
| Large base64 payloads | Slow responses, browser memory use | Caps. *Later:* binary endpoints. |
| torch.load on an untrusted path | Pickle execution | Fixed local path plus `weights_only=True` |
| skimage version differences | SSIM differs in later decimal places | Pin a version in requirements. Use the same call as `evaluate()`. |
| `predict.py` is not importable | Logic gets duplicated | Only 6 lines are repeated. The architecture is imported, never copied. |
| `edsr_x4.pt` and `figures/` are git-ignored | A fresh clone has no weights | README notes where the weights come from. The backend fails at startup with a clear message if the weights are missing. |

---

## 8. Testing strategy

**Backend (`pytest backend/test_api.py`, FastAPI TestClient, images generated in memory):**
1. `/api/health`: `model_loaded` is true.
2. `/api/enhance` with a 64×48 PNG: output is 256×192, all three images decode, `metrics` is null.
3. `/api/benchmark` with a 128×128 image: the SR image is 128×128, PSNR and SSIM are present and finite.
4. **Parity:** for the same input, the enhance EDSR output is byte-identical to the `predict.py` logic. This proves the integration did not change behaviour.
5. **Metric parity:** on one LLVIP test image, the benchmark PSNR and SSIM equal `train_sr.evaluate()` for that image.
6. Error cases:
   - `.gif` gives 415
   - random bytes named `.png` give 400
   - an oversized image gives 413
   - a 16-bit TIFF gives 200 with `normalised_from`
7. Regression: `python predict.py edsr_x4.pt test_lr.png out.png` still produces a 1280×1024 output.

**Frontend:**
- `tsc --noEmit` and `vite build` must pass.
- A manual checklist per screen:
  - upload of each format
  - each error message
  - slider keyboard control
  - synchronised zoom
  - downloads open correctly
  - responsive at 375, 768 and 1280 px
- Optionally, one Playwright smoke test (upload → enhance → compare) at the end. It is not added unless you want it.

---

## 9. Development phases

| Phase | Deliverable | Done when |
|---|---|---|
| **2.1 Backend core** | `backend/inference.py`, `app.py`, `model_info.py`, `requirements.txt`, tests | Tests 1–7 pass. `uvicorn backend.app:app` starts and the model loads once (logged). |
| **2.2 Frontend scaffold** | Vite + TS + Tailwind + shadcn setup, AppShell, `api.ts`, the Model Info page with live `/api/model` data | Builds cleanly. The Model page shows real parameter count and size. |
| **2.3 Workspace** | Dropzone, modes, status, results, downloads, history | Enhance and Benchmark both work end to end in the browser. Every error state displays. |
| **2.4 Compare** | Slider, side-by-side, synchronised zoom and pan | Manual checklist passes. Keyboard accessible. |
| **2.5 Performance + Dashboard** | Charts and cards from real data | Every number traces to a run, the API or REPORT.md. |
| **2.6 Polish + docs** | Responsive pass, empty states, README "Web app" section, optional static serving from FastAPI | Fresh-clone instructions work. Existing scripts are unchanged (`git diff` on `train_sr.py`, `predict.py` and the weights is empty). |

**Run commands, once built:**
- `uvicorn backend.app:app --reload`
- `cd frontend && npm run dev`

---

## 10. Open questions for approval

1. **Size caps:** are 640×512 for Enhance input and 2560×2048 for Benchmark acceptable, or should tile-based inference be added for full-resolution images (more code, slower)?
2. **Two-mode design:** is the Enhance/Benchmark split with metrics only in Benchmark mode acceptable?
3. **Recent activity:** is browser-side localStorage enough, or is it expected to be shared across users (which would need server storage)?
4. **Git:** should the uncommitted `train_sr.py` lazy-import change be committed first? Should `test_lr.png` and `test_sr.png` be deleted or ignored?
