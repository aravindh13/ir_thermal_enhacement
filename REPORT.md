# IR Thermal Image Enhancement Using Deep Learning-Based Super-Resolution

## 1. Abstract
Thermal (infrared) cameras produce low-resolution, low-detail images. This project trains a convolutional
super-resolution network to upscale grayscale thermal images by 4x and compares it with bicubic interpolation
on the LLVIP infrared dataset. Result: PSNR **<FINAL_PSNR>** dB / SSIM **<FINAL_SSIM>** versus
**<BICUBIC_PSNR>** dB / **<BICUBIC_SSIM>** for bicubic (100 test images).

## 2. Problem and Motivation
Thermal sensors are costly at high resolution. Software super-resolution recovers detail from cheaper
low-resolution captures, helping detection, surveillance and night-time perception.

## 3. Dataset
- **LLVIP** (visible-infrared paired, night scenes). Only the infrared images are used.
- Train: 12,025 images; test: 3,463 images; native size 1280x1024, grayscale.
- Training subset: 10,000 random train images. Evaluation subset: 100 evenly spaced test images.
- Other datasets considered but not used: FLIR ADAS v2, KAIST, M3FD, PBVS Thermal SR (scope limited to one week).
- **LR/HR pairs:** HR = original image. LR = HR downsampled 4x with bicubic interpolation, generated on the fly.

## 4. Method
**Model: EDSR-lite** (Enhanced Deep Residual SR, reduced size, trained from scratch).
- Head conv (1 -> 64 channels), 16 residual blocks (conv-ReLU-conv, residual scale 0.1), body conv.
- Upsampler: two (conv + PixelShuffle x2) stages = x4.
- Global skip: output = network residual + bicubic x4 of the input.
- Loss: L1. Optimizer: Adam, lr 2e-4. Batch 16, random 192x192 HR crops (48x48 LR), horizontal flip.
- Training: 120 minutes on a Kaggle T4 GPU. Checkpoint saved every epoch.

## 5. Evaluation
- Metrics: **PSNR** (dB) and **SSIM** on full-size test images (grayscale, range 0-1).
- Baseline: bicubic x4 upscaling of the same LR input.

## 6. Results
| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic | <BICUBIC_PSNR> | <BICUBIC_SSIM> |
| EDSR-lite (ours) | <FINAL_PSNR> | <FINAL_SSIM> |

Early training check (20 test images): bicubic 37.04 dB / 0.9798; model 39.63 dB / 0.9857 after epoch 1
and 40.50 dB / 0.9886 after epoch 2.

Figures: side-by-side (bicubic | ours | ground truth) in `figures/cmp_0.png` ... `cmp_4.png`.

## 7. Discussion
- The network recovers edges and fine structure that bicubic blurs.
- Gain over bicubic: <GAIN> dB PSNR.
- L1 training favours PSNR/SSIM fidelity; GAN-based methods (Real-ESRGAN) look sharper but are less faithful.

## 8. Limitations
- Trained from scratch, small model, 2 hours of training, single scale (x4), single dataset.
- LR images are synthetic (bicubic downsampling), not real sensor degradation.

## 9. Future Work
- Fine-tune a pretrained SwinIR/EDSR; try other scales (x2, x8).
- Train on FLIR/KAIST for generalisation; add realistic degradations (noise, blur).
- Evaluate effect on a downstream task such as pedestrian detection.

## 10. How to reproduce
1. Upload `llvip_infrared.zip` (train/ and test/ folders) as a Kaggle dataset, attach it to a GPU notebook.
2. Run `train_sr.py`. 3. Upscale an image: `python predict.py edsr_x4.pt input.jpg output.png`.
