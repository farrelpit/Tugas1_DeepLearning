"""
Metrik evaluasi: MSE, PSNR, SSIM (rekonstruksi), FID (kualitas generasi), latency inferensi.

Semua metrik citra dihitung pada rentang piksel [0, 1].
"""
import time

import torch
import torch.nn.functional as F

from . import config as cfg
from .utils import sync


# =============================================================================
# Metrik rekonstruksi
# =============================================================================
def mse_per_image(x_hat, x):
    return ((x_hat - x) ** 2).flatten(1).mean(1)


def psnr_per_image(x_hat, x, max_val: float = 1.0):
    """PSNR = 10 * log10(MAX^2 / MSE), dihitung per citra lalu dirata-rata."""
    mse = mse_per_image(x_hat, x).clamp_min(1e-10)
    return 10.0 * torch.log10(max_val ** 2 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return w2d


def ssim_per_image(x_hat, x, data_range: float = 1.0, win_size: int = 11, sigma: float = 1.5):
    """SSIM standar (Wang et al. 2004): jendela Gaussian 11x11, sigma=1.5, K1=0.01, K2=0.03."""
    c = x.size(1)
    w = _gaussian_window(win_size, sigma, c, x.device)
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(x_hat, w, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, groups=c) - mu_x2
    sig_y = F.conv2d(x_hat * x_hat, w, groups=c) - mu_y2
    sig_xy = F.conv2d(x * x_hat, w, groups=c) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return ssim_map.flatten(1).mean(1)


@torch.no_grad()
def reconstruction_metrics(recon_fn, loader, device) -> dict:
    """
    recon_fn(x) -> x_hat (keduanya di [0, 1]). Return rata-rata MSE / PSNR / SSIM atas seluruh loader.
    """
    tot = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = recon_fn(x).clamp(0, 1)
        b = x.size(0)
        tot["mse"] += mse_per_image(x_hat, x).sum().item()
        tot["psnr"] += psnr_per_image(x_hat, x).sum().item()
        tot["ssim"] += ssim_per_image(x_hat, x).sum().item()
        n += b
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def reconstruction_metrics_tensor(x_hat, x) -> dict:
    """Versi tensor (CPU/GPU) untuk hasil proyeksi yang sudah dihitung penuh."""
    x_hat = x_hat.clamp(0, 1)
    return {"mse": mse_per_image(x_hat, x).mean().item(),
            "psnr": psnr_per_image(x_hat, x).mean().item(),
            "ssim": ssim_per_image(x_hat, x).mean().item()}


# =============================================================================
# FID
# =============================================================================
@torch.no_grad()
def generate_images(sample_fn, n: int, batch: int = cfg.FID_GEN_BATCH) -> torch.Tensor:
    """sample_fn(b) -> citra di [0, 1] (shape b,C,H,W). Return tensor CPU (n,C,H,W)."""
    out, done = [], 0
    while done < n:
        b = min(batch, n - done)
        out.append(sample_fn(b).clamp(0, 1).cpu())
        done += b
    return torch.cat(out)


@torch.no_grad()
def compute_fid(real: torch.Tensor, fake: torch.Tensor, device, batch: int = 200) -> float:
    """
    FID (Frechet Inception Distance) memakai fitur pool3 InceptionV3 (2048-d) via torchmetrics.
    real/fake: tensor CPU (N,C,H,W) di [0,1]. Citra grayscale diduplikasi menjadi 3 kanal.
    Catatan: bobot Inception diunduh otomatis saat pertama kali dipanggil (butuh internet).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def prep(t):
        t = t.clamp(0, 1)
        return t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t

    for i in range(0, len(real), batch):
        fid.update(prep(real[i:i + batch]).to(device), real=True)
    for i in range(0, len(fake), batch):
        fid.update(prep(fake[i:i + batch]).to(device), real=False)
    return float(fid.compute().item())


# =============================================================================
# Latency
# =============================================================================
@torch.no_grad()
def measure_latency(fn, device, warmup: int = cfg.LATENCY_WARMUP, runs: int = cfg.LATENCY_RUNS):
    """
    Ukur waktu rata-rata satu pemanggilan fn() dalam milidetik (dengan sinkronisasi CUDA).
    Return (mean_ms, std_ms).
    """
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(times)
    return t.mean().item(), (t.std().item() if len(t) > 1 else 0.0)
