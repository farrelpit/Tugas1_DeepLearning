"""
Variational Autoencoder (MLP).

Encoder mengeluarkan q(z|x) = N(mu, diag(sigma^2)); sampling dengan reparameterization trick
z = mu + sigma * eps. Prior p(z) = N(0, I) -> ruang laten teregularisasi sehingga bisa
di-sampling langsung (z ~ N(0, I)) untuk membangkitkan citra baru.

Loss (negatif ELBO):
    L = BCE_sum(x_hat, x) + beta * KL( q(z|x) || N(0, I) )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg


class VAE(nn.Module):
    def __init__(self, latent_dim=cfg.LATENT_DIM, hidden=cfg.AE_HIDDEN):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(cfg.IMG_DIM, hidden), nn.ReLU(inplace=True))
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, cfg.IMG_DIM), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """Generasi dari prior: z ~ N(0, I) -> decoder. Output di [0, 1]."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x_hat, x, mu, logvar, beta: float = cfg.VAE_BETA):
    """Negatif ELBO per sampel (dirata-rata atas batch). Return (total, bce, kl)."""
    b = x.size(0)
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return bce + beta * kl, bce, kl
