"""
Training semua model. Jalankan dari root repo:

    python -m src.train --model all          # semua model berurutan (lewati yang sudah ada ckpt-nya)
    python -m src.train --model ddpm         # satu model saja
    python -m src.train --model gan --gan-loss minimax   # varian loss minimax (saturating)

Pilihan --model: mlp | ae | sparse_ae | vae | gan | dcgan | ddpm | all

Setiap model menyimpan:
  * checkpoint  -> OUT_DIR/checkpoints/<nama>.pt
  * history     -> OUT_DIR/results/history_<nama>.json  (loss per epoch, waktu per epoch, dll.)
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .data import get_loaders
from .models.ae import Autoencoder, ae_loss
from .models.ddpm import DDPM
from .models.gan import build_gan
from .models.mlp_classifier import MLPClassifier
from .models.vae import VAE, vae_loss
from .utils import (ckpt_exists, ensure_dirs, get_device, history_path, save_ckpt, save_json,
                    set_seed, sync, to_pm1)


def _epochs(name, override):
    return override if override is not None else cfg.EPOCHS[name]


def _log(name, epoch, total, t, **kv):
    msg = " | ".join(f"{k}={v:.4f}" for k, v in kv.items())
    print(f"[{name}] epoch {epoch:>3}/{total} ({t:5.1f}s) | {msg}", flush=True)


# =============================================================================
# MLP Classifier (pendamping, Tahap 1 eksperimen pembuktian)
# =============================================================================
@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_mlp(epochs=None):
    name, device = "mlp", get_device()
    set_seed()
    train_loader, val_loader, test_loader = get_loaders()
    model = MLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR_MLP)
    E = _epochs(name, epochs)
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    best_val, best_state = -1.0, None

    for ep in range(1, E + 1):
        model.train()
        sync(device); t0 = time.perf_counter()
        tot = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        sync(device); dt = time.perf_counter() - t0

        model.eval(); vl = vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl += F.cross_entropy(model(x), y, reduction="sum").item(); vn += y.numel()
        val_acc = eval_accuracy(model, val_loader, device)
        hist["train_loss"].append(tot / n); hist["val_loss"].append(vl / vn)
        hist["val_acc"].append(val_acc); hist["epoch_time"].append(dt)
        if val_acc > best_val:   # seleksi model terbaik berdasarkan val accuracy
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _log(name, ep, E, dt, train_loss=tot / n, val_loss=vl / vn, val_acc=val_acc)

    model.load_state_dict(best_state)
    test_acc = eval_accuracy(model, test_loader, device)
    hist["test_acc"] = test_acc
    print(f"[mlp] TEST ACCURACY = {test_acc:.4f}" + ("" if test_acc >= 0.88 else
          "  <-- di bawah 88%! naikkan EPOCHS['mlp'] di config.py"))
    save_ckpt(name, model, test_acc=test_acc)
    save_json(hist, history_path(name))


# =============================================================================
# Vanilla AE & Sparse AE
# =============================================================================
def train_ae(sparse=False, epochs=None):
    name = "sparse_ae" if sparse else "ae"
    device = get_device()
    set_seed()
    train_loader, val_loader, _ = get_loaders()
    latent = cfg.SPARSE_LATENT if sparse else cfg.LATENT_DIM
    lam = cfg.SPARSE_LAMBDA if sparse else 0.0
    model = Autoencoder(latent_dim=latent, sparse=sparse).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR_AE)
    E = _epochs(name, epochs)
    hist = {k: [] for k in ["train_loss", "train_mse", "train_l1", "val_loss", "val_mse", "val_l1",
                            "epoch_time"]}
    best_val, best_state, best_ep = float("inf"), None, 0

    for ep in range(1, E + 1):
        model.train()
        sync(device); t0 = time.perf_counter()
        s = {"loss": 0.0, "mse": 0.0, "l1": 0.0}; n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, z = model(x)
            loss, mse, l1 = ae_loss(x_hat, x, z, lam)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            b = x.size(0); n += b
            s["loss"] += loss.item() * b; s["mse"] += mse.item() * b; s["l1"] += l1.item() * b
        sync(device); dt = time.perf_counter() - t0

        model.eval(); v = {"loss": 0.0, "mse": 0.0, "l1": 0.0}; vn = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, z = model(x)
                loss, mse, l1 = ae_loss(x_hat, x, z, lam)
                b = x.size(0); vn += b
                v["loss"] += loss.item() * b; v["mse"] += mse.item() * b; v["l1"] += l1.item() * b
        for k in ("loss", "mse", "l1"):
            hist[f"train_{k}"].append(s[k] / n); hist[f"val_{k}"].append(v[k] / vn)
        hist["epoch_time"].append(dt)
        if v["loss"] / vn < best_val:
            best_val, best_ep = v["loss"] / vn, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        _log(name, ep, E, dt, train_loss=s["loss"] / n, train_mse=s["mse"] / n, val_loss=v["loss"] / vn)

    model.load_state_dict(best_state)
    hist["best_epoch"] = best_ep
    save_ckpt(name, model, latent_dim=latent, sparse=sparse, best_epoch=best_ep)
    save_json(hist, history_path(name))


# =============================================================================
# VAE
# =============================================================================
def train_vae(epochs=None):
    name, device = "vae", get_device()
    set_seed()
    train_loader, val_loader, _ = get_loaders()
    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR_VAE)
    E = _epochs(name, epochs)
    hist = {k: [] for k in ["train_loss", "train_bce", "train_kl", "val_loss", "val_bce", "val_kl",
                            "epoch_time"]}
    best_val, best_state, best_ep = float("inf"), None, 0

    for ep in range(1, E + 1):
        model.train()
        sync(device); t0 = time.perf_counter()
        s = {"loss": 0.0, "bce": 0.0, "kl": 0.0}; n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, bce, kl = vae_loss(x_hat, x, mu, logvar)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            b = x.size(0); n += b
            s["loss"] += loss.item() * b; s["bce"] += bce.item() * b; s["kl"] += kl.item() * b
        sync(device); dt = time.perf_counter() - t0

        model.eval(); v = {"loss": 0.0, "bce": 0.0, "kl": 0.0}; vn = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss, bce, kl = vae_loss(x_hat, x, mu, logvar)
                b = x.size(0); vn += b
                v["loss"] += loss.item() * b; v["bce"] += bce.item() * b; v["kl"] += kl.item() * b
        for k in ("loss", "bce", "kl"):
            hist[f"train_{k}"].append(s[k] / n); hist[f"val_{k}"].append(v[k] / vn)
        hist["epoch_time"].append(dt)
        if v["loss"] / vn < best_val:
            best_val, best_ep = v["loss"] / vn, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        _log(name, ep, E, dt, train_loss=s["loss"] / n, bce=s["bce"] / n, kl=s["kl"] / n,
             val_loss=v["loss"] / vn)

    model.load_state_dict(best_state)
    hist["best_epoch"] = best_ep
    save_ckpt(name, model, latent_dim=cfg.LATENT_DIM, best_epoch=best_ep)
    save_json(hist, history_path(name))


# =============================================================================
# GAN (Vanilla / DCGAN)
# =============================================================================
def _grad_norm(module):
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


def train_gan(kind="gan", epochs=None, gan_loss="nonsaturating"):
    """
    kind     : 'gan' (MLP) atau 'dcgan'
    gan_loss : 'nonsaturating' (default; G memaksimalkan log D(G(z)))
               'minimax'       (G meminimalkan log(1 - D(G(z))) -> gradien menghilang saat D kuat)
    """
    name = kind if gan_loss == "nonsaturating" else f"{kind}_minimax"
    device = get_device()
    set_seed()
    train_loader, _, _ = get_loaders()
    G, D = build_gan(kind)
    G, D = G.to(device), D.to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cfg.LR_GAN, betas=cfg.GAN_BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.LR_GAN, betas=cfg.GAN_BETAS)
    bce = nn.BCEWithLogitsLoss()
    E = _epochs(kind, epochs)
    hist = {k: [] for k in ["loss_D", "loss_G", "D_real", "D_fake", "G_grad_norm", "epoch_time"]}

    for ep in range(1, E + 1):
        G.train(); D.train()
        sync(device); t0 = time.perf_counter()
        a = {k: 0.0 for k in ["loss_D", "loss_G", "D_real", "D_fake", "gn"]}; n = 0; nb = 0
        for x, _ in train_loader:
            real = to_pm1(x.to(device))
            b = real.size(0)
            ones = torch.ones(b, device=device); zeros = torch.zeros(b, device=device)

            # ---- (1) update Discriminator: maksimalkan log D(x) + log(1 - D(G(z))) ----
            z = torch.randn(b, cfg.NOISE_DIM, device=device)
            fake = G(z).detach()
            d_real, d_fake = D(real), D(fake)
            loss_D = bce(d_real, ones) + bce(d_fake, zeros)
            optD.zero_grad(set_to_none=True); loss_D.backward(); optD.step()

            # ---- (2) update Generator ----
            z = torch.randn(b, cfg.NOISE_DIM, device=device)
            d_gz = D(G(z))
            if gan_loss == "nonsaturating":
                loss_G = bce(d_gz, ones)                 # -log D(G(z))
            else:
                loss_G = -bce(d_gz, zeros)               # log(1 - D(G(z)))  (minimax asli)
            optG.zero_grad(set_to_none=True); loss_G.backward()
            gn = _grad_norm(G)                           # norma gradien G (bukti vanishing gradient)
            optG.step()

            a["loss_D"] += loss_D.item(); a["loss_G"] += loss_G.item(); a["gn"] += gn
            a["D_real"] += torch.sigmoid(d_real).mean().item()
            a["D_fake"] += torch.sigmoid(d_fake).mean().item()
            nb += 1
        sync(device); dt = time.perf_counter() - t0
        hist["loss_D"].append(a["loss_D"] / nb); hist["loss_G"].append(a["loss_G"] / nb)
        hist["D_real"].append(a["D_real"] / nb); hist["D_fake"].append(a["D_fake"] / nb)
        hist["G_grad_norm"].append(a["gn"] / nb); hist["epoch_time"].append(dt)
        _log(name, ep, E, dt, loss_D=a["loss_D"] / nb, loss_G=a["loss_G"] / nb,
             D_real=a["D_real"] / nb, D_fake=a["D_fake"] / nb)

    save_ckpt(name, G, kind=kind, gan_loss=gan_loss, nz=cfg.NOISE_DIM)   # simpan Generator saja
    torch.save({"state_dict": D.state_dict()}, cfg.CKPT_DIR / f"{name}_D.pt")
    save_json(hist, history_path(name))


# =============================================================================
# DDPM
# =============================================================================
def train_ddpm(epochs=None):
    name, device = "ddpm", get_device()
    set_seed()
    train_loader, val_loader, _ = get_loaders()
    model = DDPM().to(device)
    opt = torch.optim.Adam(model.net.parameters(), lr=cfg.LR_DDPM)
    E = _epochs(name, epochs)
    hist = {"train_loss": [], "val_loss": [], "epoch_time": []}
    step = 0
    # generator tetap untuk validasi -> val loss bisa dibandingkan antar epoch
    val_gen = torch.Generator(device=device).manual_seed(cfg.SEED)

    for ep in range(1, E + 1):
        model.net.train()
        sync(device); t0 = time.perf_counter()
        tot = n = 0
        for x, _ in train_loader:
            x0 = to_pm1(x.to(device))
            loss = model.loss(x0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), cfg.DDPM_GRAD_CLIP)
            opt.step()
            model.update_ema(step=step); step += 1
            tot += loss.item() * x0.size(0); n += x0.size(0)
        sync(device); dt = time.perf_counter() - t0

        model.net.eval(); vl = vn = 0
        val_gen.manual_seed(cfg.SEED)
        with torch.no_grad():
            for x, _ in val_loader:
                x0 = to_pm1(x.to(device))
                vl += model.loss(x0, generator=val_gen).item() * x0.size(0); vn += x0.size(0)
        hist["train_loss"].append(tot / n); hist["val_loss"].append(vl / vn); hist["epoch_time"].append(dt)
        _log(name, ep, E, dt, train_loss=tot / n, val_loss=vl / vn)

    save_ckpt(name, model, T=model.T)    # termasuk bobot EMA (dipakai saat sampling)
    save_json(hist, history_path(name))


# =============================================================================
# CLI
# =============================================================================
ORDER = ["mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"]


def run(model_name, epochs=None, gan_loss="nonsaturating", force=False):
    ensure_dirs()
    ck_name = model_name if gan_loss == "nonsaturating" or model_name not in ("gan", "dcgan") \
        else f"{model_name}_minimax"
    if ckpt_exists(ck_name) and not force:
        print(f"[skip] checkpoint '{ck_name}' sudah ada (pakai --force untuk melatih ulang)")
        return
    print(f"\n===== TRAIN: {ck_name} =====")
    if model_name == "mlp":
        train_mlp(epochs)
    elif model_name == "ae":
        train_ae(False, epochs)
    elif model_name == "sparse_ae":
        train_ae(True, epochs)
    elif model_name == "vae":
        train_vae(epochs)
    elif model_name in ("gan", "dcgan"):
        train_gan(model_name, epochs, gan_loss)
    elif model_name == "ddpm":
        train_ddpm(epochs)
    else:
        raise ValueError(model_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=ORDER + ["all"])
    ap.add_argument("--epochs", type=int, default=None, help="override jumlah epoch")
    ap.add_argument("--gan-loss", default="nonsaturating", choices=["nonsaturating", "minimax"])
    ap.add_argument("--force", action="store_true", help="latih ulang walau checkpoint ada")
    args = ap.parse_args()
    for m in (ORDER if args.model == "all" else [args.model]):
        run(m, args.epochs, args.gan_loss, args.force)
