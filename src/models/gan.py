"""
Vanilla GAN (fully-connected) dan DCGAN (konvolusional penuh).

Keduanya bekerja di rentang piksel [-1, 1] (output Generator = Tanh).
Discriminator mengeluarkan LOGIT (bukan probabilitas); sigmoid ada di BCEWithLogitsLoss
sehingga lebih stabil secara numerik.

Prinsip DCGAN yang membedakannya dari Vanilla GAN:
  1. Tanpa lapisan fully-connected/pooling -> strided conv (D) & transposed conv (G).
  2. BatchNorm pada G dan D (kecuali output G dan input D).
  3. ReLU di G (Tanh di output), LeakyReLU di D.
  4. Inisialisasi bobot N(0, 0.02).
"""
import torch
import torch.nn as nn

from .. import config as cfg


# ----------------------------------------------------------------------------
# Vanilla GAN (MLP)
# ----------------------------------------------------------------------------
class VanillaGenerator(nn.Module):
    def __init__(self, nz=cfg.NOISE_DIM):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(nz, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, cfg.IMG_DIM), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE)


class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.IMG_DIM, 1024), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ----------------------------------------------------------------------------
# DCGAN
# ----------------------------------------------------------------------------
class DCGANGenerator(nn.Module):
    """z (100) -> 256@7x7 -> 128@14x14 -> 64@28x28 -> 1@28x28 (Tanh)."""

    def __init__(self, nz=cfg.NOISE_DIM, ngf=64):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),      # 1x1 -> 7x7
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 7 -> 14
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),     # 14 -> 28
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, cfg.IMG_CH, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    """1@28x28 -> 64@14 -> 128@7 -> 256@4 -> 1x1 logit (konvolusional penuh)."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.IMG_CH, ndf, 4, 2, 1),                        # 28 -> 14 (tanpa BN)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),               # 14 -> 7
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),           # 7 -> 4
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0),                             # 4 -> 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_weights_init(m):
    """Inisialisasi DCGAN: bobot ~ N(0, 0.02); BatchNorm gamma ~ N(1, 0.02), beta = 0."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build_gan(kind: str):
    """kind: 'gan' (Vanilla/MLP) atau 'dcgan'. Return (G, D)."""
    if kind == "gan":
        return VanillaGenerator(), VanillaDiscriminator()
    if kind == "dcgan":
        g, d = DCGANGenerator(), DCGANDiscriminator()
        g.apply(dcgan_weights_init)
        d.apply(dcgan_weights_init)
        return g, d
    raise ValueError(kind)
