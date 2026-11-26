"""x4 thermal super-resolution on LLVIP infrared. Paste into a Kaggle notebook cell (GPU on) and run.
Makes LR on the fly (bicubic), trains EDSR-lite, saves checkpoints, reports PSNR/SSIM vs bicubic."""
import glob, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

DATA = "/kaggle/input"          # searched recursively for train/ and test/
OUT = "/kaggle/working"
SCALE, HR_PATCH, BATCH = 4, 192, 16
TRAIN_MINUTES = 120             # time budget; checkpoint every epoch
N_TRAIN, N_TEST = 10000, 100    # subset sizes
dev = "cuda" if torch.cuda.is_available() else "cpu"


def find(split):
    fs = sorted(glob.glob(f"{DATA}/**/{split}/*.jpg", recursive=True))
    assert fs, f"no images for {split} under {DATA}"
    return fs


class DS(torch.utils.data.Dataset):
    def __init__(self, files): self.f = files
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        im = Image.open(self.f[i]).convert("L")
        x = random.randint(0, im.width - HR_PATCH); y = random.randint(0, im.height - HR_PATCH)
        hr = im.crop((x, y, x + HR_PATCH, y + HR_PATCH))
        if random.random() < .5: hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        lr = hr.resize((HR_PATCH // SCALE,) * 2, Image.BICUBIC)
        t = lambda p: torch.from_numpy(np.asarray(p, np.float32) / 255)[None]
        return t(lr), t(hr)


class Res(nn.Module):
    def __init__(s, c):
        super().__init__(); s.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1))
    def forward(s, x): return x + 0.1 * s.b(x)


class EDSR(nn.Module):  # ponytail: small from-scratch model; swap in pretrained SwinIR/EDSR if quality falls short
    def __init__(s, c=64, n=16):
        super().__init__()
        s.head = nn.Conv2d(1, c, 3, 1, 1)
        s.body = nn.Sequential(*[Res(c) for _ in range(n)], nn.Conv2d(c, c, 3, 1, 1))
        s.up = nn.Sequential(nn.Conv2d(c, c * 4, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(c, c * 4, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(c, 1, 3, 1, 1))
    def forward(s, x):
        h = s.head(x)
        return s.up(h + s.body(h)) + F.interpolate(x, scale_factor=SCALE, mode="bicubic", align_corners=False)


def psnr(a, b): return 10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-10))


@torch.no_grad()
def evaluate(model, files, save_dir=None):
    from skimage.metrics import structural_similarity  # lazy: inference (predict.py) doesn't need it
    model.eval(); r = {"bicubic": [], "model": []}
    for k, f in enumerate(files):
        hr = Image.open(f).convert("L"); hr = hr.crop((0, 0, hr.width // 4 * 4, hr.height // 4 * 4))
        lr = hr.resize((hr.width // SCALE, hr.height // SCALE), Image.BICUBIC)
        h = np.asarray(hr, np.float32) / 255
        bic = np.asarray(lr.resize(hr.size, Image.BICUBIC), np.float32) / 255
        x = torch.from_numpy(np.asarray(lr, np.float32) / 255)[None, None].to(dev)
        sr = model(x).clamp(0, 1)[0, 0].cpu().numpy()
        for n, p in (("bicubic", bic), ("model", sr)):
            r[n].append((psnr(h, p), structural_similarity(h, p, data_range=1.0)))
        if save_dir and k < 5:  # visual comparison: LR-bicubic | model | ground truth
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray((np.hstack([bic, sr, h]) * 255).astype(np.uint8)).save(f"{save_dir}/cmp_{k}.png")
    model.train()
    return {n: tuple(np.mean(v, 0).round(4)) for n, v in r.items()}  # (PSNR dB, SSIM)


def main():
    train, test = find("train"), find("test")
    random.seed(0); random.shuffle(train)
    train, test = train[:N_TRAIN], test[::max(1, len(test) // N_TEST)][:N_TEST]
    dl = torch.utils.data.DataLoader(DS(train), BATCH, shuffle=True, num_workers=4, drop_last=True)
    model = EDSR().to(dev); opt = torch.optim.Adam(model.parameters(), 2e-4)
    t0, ep = time.time(), 0
    while time.time() - t0 < TRAIN_MINUTES * 60:
        ep += 1; tot = 0
        for lr, hr in dl:
            lr, hr = lr.to(dev), hr.to(dev)
            loss = F.l1_loss(model(lr), hr); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if time.time() - t0 > TRAIN_MINUTES * 60: break
        torch.save(model.state_dict(), f"{OUT}/edsr_x4.pt")
        print(f"epoch {ep} loss {tot / len(dl):.4f} min {(time.time() - t0) / 60:.0f} (PSNR,SSIM)", evaluate(model, test[:20]), flush=True)
    print("FINAL", evaluate(model, test, f"{OUT}/figures"))


if __name__ == "__main__":
    m = EDSR(); assert m(torch.rand(1, 1, 48, 48)).shape == (1, 1, 192, 192)  # self-check
    main()
