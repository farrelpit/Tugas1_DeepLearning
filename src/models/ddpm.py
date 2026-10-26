"""
DDPM (Denoising Diffusion Probabilistic Model, Ho et al. 2020) dengan U-Net kecil.

Forward process (menambah noise, tanpa parameter terlatih):
    x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)

Objektif latih (supervised noise prediction, "L_simple"):
    L = E_{x0, t, eps} || eps - eps_theta(x_t, t) ||^2

Reverse process (sampling ancestral, T langkah iteratif):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_theta(x_t, t)) + sqrt(beta_t) * z

Bekerja di rentang [-1, 1]. Bobot EMA dipakai untuk sampling (kualitas sampel lebih baik).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg


# ----------------------------------------------------------------------------
# Blok penyusun U-Net
# ----------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding untuk timestep t (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]   # injeksi informasi timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention single-head pada feature map (dipakai di resolusi 7x7 saja)."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)          # (B, HW, C)
        k = k.reshape(b, c, h * w)                          # (B, C, HW)
        v = v.reshape(b, c, h * w).transpose(1, 2)          # (B, HW, C)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


def _upsample(ch):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1))


class UNet(nn.Module):
    """U-Net kecil untuk citra 28x28: 28 -> 14 -> 7 -> 14 -> 28 dengan skip connection."""

    def __init__(self, in_ch=cfg.IMG_CH, base=cfg.DDPM_BASE_CH, tdim=256):
        super().__init__()
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        # encoder
        self.d1 = ResBlock(base, base, tdim)                      # 28x28
        self.down1 = nn.Conv2d(base, base, 3, 2, 1)               # -> 14x14
        self.d2 = ResBlock(base, base * 2, tdim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, 2, 1)       # -> 7x7
        self.d3 = ResBlock(base * 2, base * 2, tdim)
        # bottleneck
        self.m1 = ResBlock(base * 2, base * 2, tdim)
        self.attn = SelfAttention(base * 2)
        self.m2 = ResBlock(base * 2, base * 2, tdim)
        # decoder
        self.u3 = ResBlock(base * 4, base * 2, tdim)              # concat(m, d3)
        self.up2 = _upsample(base * 2)                            # -> 14x14
        self.u2 = ResBlock(base * 4, base, tdim)                  # concat(up, d2)
        self.up1 = _upsample(base)                                # -> 28x28
        self.u1 = ResBlock(base * 2, base, tdim)                  # concat(up, d1)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.base))
        s1 = self.d1(self.inc(x), temb)                  # (B, base, 28, 28)
        s2 = self.d2(self.down1(s1), temb)               # (B, 2base, 14, 14)
        s3 = self.d3(self.down2(s2), temb)               # (B, 2base, 7, 7)
        m = self.m2(self.attn(self.m1(s3, temb)), temb)
        u = self.u3(torch.cat([m, s3], 1), temb)
        u = self.u2(torch.cat([self.up2(u), s2], 1), temb)
        u = self.u1(torch.cat([self.up1(u), s1], 1), temb)
        return self.out(u)                               # prediksi noise eps_theta


# ----------------------------------------------------------------------------
# Proses diffusion
# ----------------------------------------------------------------------------
class DDPM(nn.Module):
    def __init__(self, T=cfg.DDPM_T, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        self.net = UNet()
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)

        # Jadwal noise linear (Ho et al.). Jika T diubah (mode smoke), skala beta menyesuaikan.
        scale = 1000.0 / T
        betas = torch.linspace(beta_start * scale, beta_end * scale, T).clamp(max=0.999)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("sqrt_abar", abar.sqrt())
        self.register_buffer("sqrt_1m_abar", (1.0 - abar).sqrt())

    # --- forward process -------------------------------------------------------
    def q_sample(self, x0, t, noise):
        return (self.sqrt_abar[t][:, None, None, None] * x0
                + self.sqrt_1m_abar[t][:, None, None, None] * noise)

    # --- objektif latih --------------------------------------------------------
    def loss(self, x0, generator=None):
        """MSE antara noise asli dan noise prediksi. x0 di [-1, 1]."""
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        pred = self.net(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def update_ema(self, decay=cfg.DDPM_EMA_DECAY, step=None):
        if step is not None:                       # warmup EMA agar tidak "tertinggal" di awal
            decay = min(decay, (1 + step) / (10 + step))
        for p_ema, p in zip(self.ema_net.parameters(), self.net.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    # --- reverse process (sampling) -------------------------------------------
    @torch.no_grad()
    def sample(self, n, device, use_ema=True, return_trajectory=False, n_snapshots=10):
        """
        Sampling ancestral T langkah. Output di [-1, 1], shape (n, C, H, W).

        Jika return_trajectory=True, kembalikan juga list [(t, x_t), ...] sepanjang proses
        (dari noise murni x_T sampai x_0) untuk visualisasi Figur 5.
        t pada trajectory = indeks langkah (0-based) yang BARU saja dijalankan; t='T' = noise awal.
        """
        net = self.ema_net if use_ema else self.net
        net.eval()
        x = torch.randn(n, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)
        snap_at = set(torch.linspace(self.T - 1, 0, n_snapshots - 1).long().tolist())
        traj = [("T", x.clone().cpu())] if return_trajectory else None
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = net(x, t)
            mean = (x - self.betas[i] / self.sqrt_1m_abar[i] * eps) / self.alphas[i].sqrt()
            x = mean + self.betas[i].sqrt() * torch.randn_like(x) if i > 0 else mean
            if return_trajectory and i in snap_at:
                traj.append((i, x.clamp(-1, 1).clone().cpu()))
        x = x.clamp(-1, 1)
        return (x, traj) if return_trajectory else x
