"""Pemuat checkpoint (dipakai bersama oleh evaluate.py, mlp_proof.py, figures.py)."""
import torch

from .models.ae import Autoencoder
from .models.ddpm import DDPM
from .models.gan import build_gan
from .models.mlp_classifier import MLPClassifier
from .models.vae import VAE
from .utils import ckpt_path


def _read(name, device):
    return torch.load(ckpt_path(name), map_location=device)


def load_mlp(device) -> MLPClassifier:
    m = MLPClassifier().to(device)
    m.load_state_dict(_read("mlp", device)["state_dict"])
    return m.eval()


def load_ae(name: str, device) -> Autoencoder:
    """name: 'ae' atau 'sparse_ae'."""
    ck = _read(name, device)
    m = Autoencoder(latent_dim=ck["latent_dim"], sparse=ck["sparse"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def load_vae(device) -> VAE:
    ck = _read("vae", device)
    m = VAE(latent_dim=ck["latent_dim"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def load_generator(name: str, device):
    """name: 'gan' | 'dcgan' (atau varian '<kind>_minimax'). Return Generator (mode eval)."""
    ck = _read(name, device)
    g, _ = build_gan(ck["kind"])
    g.load_state_dict(ck["state_dict"])
    return g.to(device).eval()


def load_ddpm(device) -> DDPM:
    ck = _read("ddpm", device)
    m = DDPM(T=ck["T"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()
