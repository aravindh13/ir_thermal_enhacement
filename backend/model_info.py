"""Static model card. Every number here is copied from REPORT.md / train_sr.py, or computed at startup."""
import os

from backend.inference import LIMITS, SCALE, WEIGHTS

REPORTED = {  # REPORT.md section 6, 100 evenly spaced LLVIP test images, synthetic bicubic LR
    "source": "REPORT.md (100 LLVIP test images, bicubic x4 degradation)",
    "bicubic": {"psnr_db": 34.71, "ssim": 0.9602},
    "edsr": {"psnr_db": 39.88, "ssim": 0.9830},
}


def model_card(model) -> dict:
    return {
        "name": "EDSR-lite",
        "scale": SCALE,
        "methods": [
            {"id": "bicubic", "label": f"Bicubic x{SCALE}", "role": "baseline"},
            {"id": "edsr", "label": f"EDSR-lite x{SCALE}", "role": "model"},
        ],
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights_bytes": os.path.getsize(WEIGHTS),
        "architecture": {
            "input_channels": 1, "features": 64, "residual_blocks": 16, "residual_scale": 0.1,
            "block": "conv3x3 - ReLU - conv3x3",
            "upsampler": "2 x (conv3x3 64->256 + PixelShuffle x2)",
            "global_skip": f"output = network residual + bicubic x{SCALE} of input",
        },
        "training": {
            "loss": "L1", "optimizer": "Adam", "learning_rate": 2e-4, "batch_size": 16,
            "hr_patch": 192, "lr_patch": 48, "augmentation": "random horizontal flip",
            "epochs": 87, "minutes": 120, "gpu": "Kaggle T4", "init": "from scratch",
        },
        "dataset": {
            "name": "LLVIP (infrared only)", "url": "https://bupt-ai-cz.github.io/LLVIP/",
            "train_images": 12025, "test_images": 3463, "train_subset": 10000, "eval_subset": 100,
            "resolution": "1280x1024", "channels": "grayscale",
            "degradation": f"bicubic x{SCALE} downsampling (synthetic LR)",
        },
        "reported_results": REPORTED,
        "limits": {m: {"long_side": a, "short_side": b} for m, (a, b) in LIMITS.items()},
        "limitations": [
            "Trained from scratch: small model, 2 hours of training.",
            f"Single scale (x{SCALE}) and single dataset (LLVIP).",
            "Low-resolution training inputs were synthetic (bicubic), not real sensor degradation.",
            "Grayscale only: colour or false-colour images are converted to luminance.",
            "No tiled inference: input size is capped.",
        ],
    }
