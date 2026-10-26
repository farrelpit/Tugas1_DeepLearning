"""Fungsi bantu: seed, device, I/O JSON/checkpoint, konversi rentang piksel."""
import json
import os
import random

import numpy as np
import torch

from . import config as cfg


def set_seed(seed: int = cfg.SEED) -> None:
    """Kunci semua sumber keacakan (Python, NumPy, PyTorch CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device) -> None:
    """Sinkronisasi CUDA agar pengukuran waktu akurat."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def ensure_dirs() -> None:
    for d in (cfg.CKPT_DIR, cfg.FIG_DIR, cfg.RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- Rentang piksel -----------------------------------------------------------
# DataLoader selalu menghasilkan citra di [0, 1].
#  * AE / Sparse AE / VAE / MLP : memakai [0, 1] (output Sigmoid)
#  * GAN / DCGAN / DDPM         : memakai [-1, 1] (output Tanh / noise Gaussian)
def to_pm1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# --- I/O ----------------------------------------------------------------------
def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ckpt_path(name: str):
    return cfg.CKPT_DIR / f"{name}.pt"


def history_path(name: str):
    return cfg.RESULT_DIR / f"history_{name}.json"


def save_ckpt(name: str, model: torch.nn.Module, **extra) -> None:
    torch.save({"state_dict": model.state_dict(), **extra}, ckpt_path(name))


def load_ckpt(name: str, model: torch.nn.Module, device) -> dict:
    ckpt = torch.load(ckpt_path(name), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return ckpt


def ckpt_exists(name: str) -> bool:
    return os.path.exists(ckpt_path(name))
