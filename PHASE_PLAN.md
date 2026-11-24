# IR Thermal Image Enhancement (Super-Resolution): Phase Record

## Phase 01: Dataset Collection & Analysis

### Datasets considered
| Dataset | Description | Size |
|---|---|---|
| FLIR ADAS v2 | Paired thermal and RGB driving scenes, 640x512 | ~26k images, ~15 GB |
| KAIST Multispectral | Thermal and RGB pedestrian scenes | ~10 GB+ |
| LLVIP | Paired visible and infrared night scenes | 30k images, ~2 GB |
| M3FD | Thermal and visible, varied scenes | ~4k images |
| PBVS Thermal SR | Paired LR/HR thermal at x2, x4, x8 | varies |
| CVC-09 / CVC-14 | Small sets for quick tests | small |
| DIV2K / Flickr2K | Optional grayscale pretraining | optional |

### Decision
- Primary: **LLVIP** (small, easy download). Optional: **FLIR ADAS v2** for volume.
- Single scale: **x4**. LR inputs made by bicubic downsampling of HR thermal images.

### Cloud storage options
| Option | Best for | Cost |
|---|---|---|
| Kaggle Dataset | Kaggle notebooks (free GPU, ~30 h/week) | Free |
| Google Drive | Colab (15 GB free) | Free |
| Hugging Face Datasets | Sharing and versioning | Free |
| AWS S3 / GCS | Serious cloud GPU training | Paid |

Recommended: Kaggle or Google Drive. Upload needs the user's own login or token.

## One-week schedule
| Day | Task |
|---|---|
| 1 | Download LLVIP. Generate x4 LR/HR pairs. |
| 2 | Bicubic baseline (PSNR/SSIM). Set up training loop. |
| 3-4 | Fine-tune a small SR model (EDSR-lite / SwinIR / Real-ESRGAN) on a free GPU. |
| 5 | Evaluate vs bicubic. Make visual comparison figures. |
| 6 | Write report, README, slides. |
| 7 | Buffer for reruns and fixes. |

## Scope rules
- One dataset, one scale (x4).
- Fine-tune a pretrained model; do not design a new architecture.
- Train 1-3 hours on a 10-20k image subset.
- Skip extras (multiple datasets, GAN losses, deployment) unless ahead of schedule.

## Risks
- Dataset download or sign-in issues (low for LLVIP).
- Free GPU quota limits.
- Debugging time. Day 7 is the buffer.

## Next step
Write the LLVIP download script and the x4 pair generation script.
