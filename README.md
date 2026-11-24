# IR Thermal Image Enhancement (x4 Super-Resolution)

EDSR-lite CNN that upscales grayscale thermal images 4x, trained on LLVIP infrared. See [REPORT.md](REPORT.md).

| File | Purpose |
|---|---|
| `train_sr.py` | Train and evaluate (PSNR/SSIM vs bicubic). Runs on a Kaggle GPU notebook. |
| `predict.py` | Upscale one image with trained weights. |
| `REPORT.md` | Project report. |

```bash
pip install torch pillow numpy scikit-image
python predict.py edsr_x4.pt input.jpg output.png
```

Data: LLVIP infrared images (https://bupt-ai-cz.github.io/LLVIP/), not included in this repo.
