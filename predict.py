"""Upscale a thermal image x4 with the trained model.
Usage: python predict.py edsr_x4.pt input.jpg output.png
Put predict.py, train_sr.py and edsr_x4.pt in the same folder."""
import sys, numpy as np, torch
from PIL import Image
from train_sr import EDSR, dev  # importing is safe: training only runs under __main__

weights, src, dst = sys.argv[1:4]
model = EDSR().to(dev)
model.load_state_dict(torch.load(weights, map_location=dev))
model.eval()

x = torch.from_numpy(np.asarray(Image.open(src).convert("L"), np.float32) / 255)[None, None].to(dev)
with torch.no_grad():
    sr = model(x).clamp(0, 1)[0, 0].cpu().numpy()
Image.fromarray((sr * 255).round().astype(np.uint8)).save(dst)
print(f"saved {dst}  ({x.shape[3]}x{x.shape[2]} -> {sr.shape[1]}x{sr.shape[0]})")
