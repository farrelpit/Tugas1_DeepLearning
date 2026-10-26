"""
Vanilla Autoencoder dan Sparse Autoencoder (arsitektur MLP, deterministik).

Encoder : 784 -> 256 -> latent
Decoder : latent -> 256 -> 784 (Sigmoid, output di [0,1])

Perbedaan keduanya hanya pada:
  * Vanilla AE : latent linear (padat), loss = MSE.
  * Sparse AE  : latent dilewatkan ReLU (h >= 0) dan loss = MSE + lambda * ||h||_1.
                 Penalti L1 mendorong banyak unit laten tepat bernilai 0 (jarang).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=cfg.LATENT_DIM, hidden=cfg.AE_HIDDEN, sparse=False):
        super().__init__()
        self.sparse = sparse
        self.latent_dim = latent_dim
        enc = [nn.Flatten(), nn.Linear(cfg.IMG_DIM, hidden), nn.ReLU(inplace=True),
               nn.Linear(hidden, latent_dim)]
        if sparse:
            enc.append(nn.ReLU(inplace=True))   # h >= 0 -> L1 menghasilkan nol eksak
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, cfg.IMG_DIM), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def ae_loss(x_hat, x, z, lam: float = 0.0):
    """
    L = MSE(x_hat, x) + lam * mean_batch( ||z||_1 )

    lam = 0 -> Vanilla AE. lam > 0 -> Sparse AE.
    Mengembalikan (total, mse, l1) agar komponen bisa dicatat terpisah.
    """
    mse = F.mse_loss(x_hat, x)                 # rata-rata atas semua piksel & batch
    l1 = z.abs().sum(dim=1).mean()             # ||z||_1 per sampel, dirata-rata atas batch
    return mse + lam * l1, mse, l1
