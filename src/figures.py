"""
Pembuat 5 figur wajib (resolusi 200 dpi) -> OUT_DIR/figures/

    python -m src.figures                 # semua figur
    python -m src.figures --figs 1 4      # sebagian saja

Figur 1 : grid rekonstruksi (Original / Vanilla AE / Sparse AE / VAE)
Figur 2 : grid sampling acak 4x4 (VAE / Vanilla GAN / DCGAN / DDPM)
Figur 3 : bukti empiris batasan MLP Classifier (butuh hasil `python -m src.mlp_proof`)
Figur 4 : interpolasi linier ruang laten (Vanilla AE / Sparse AE / VAE / Vanilla GAN / DCGAN)
Figur 5 : kurva training loss keenam model + trajektori reverse diffusion DDPM
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torchvision.utils import make_grid

from . import config as cfg
from .data import CLASS_NAMES, first_index_per_class, load_split_tensors
from .loaders import load_ae, load_ddpm, load_generator, load_mlp, load_vae
from .metrics import mse_per_image
from .utils import ckpt_exists, ensure_dirs, get_device, history_path, load_json, set_seed, to_01

DPI = 200
PAD = 2
S = cfg.IMG_SIZE


# -----------------------------------------------------------------------------
# Helper gambar
# -----------------------------------------------------------------------------
def _grid(imgs: torch.Tensor, nrow: int) -> np.ndarray:
    """Susun citra (N,1,H,W) di [0,1] menjadi satu kanvas grayscale."""
    return make_grid(imgs.cpu(), nrow=nrow, padding=PAD, pad_value=1.0)[0].numpy()


def _show_rows(ax, rows, labels, title=None, col_labels=None, fontsize=9):
    """Tampilkan beberapa baris citra (tiap baris tensor (n,1,H,W)) dengan label baris di kiri."""
    n = rows[0].size(0)
    ax.imshow(_grid(torch.cat(rows), n), cmap="gray", vmin=0, vmax=1)
    ax.set_yticks([PAD + i * (S + PAD) + S / 2 for i in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=fontsize)
    if col_labels is not None:
        ax.set_xticks([PAD + j * (S + PAD) + S / 2 for j in range(n)])
        ax.set_xticklabels(col_labels, fontsize=fontsize - 1)
        ax.xaxis.tick_top()
    else:
        ax.set_xticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=8 if col_labels is None else 24)


def _show_grid(ax, imgs, nrow, title):
    ax.imshow(_grid(imgs, nrow), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def _save(fig, name):
    path = cfg.FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {path}")


def _names():
    return CLASS_NAMES[cfg.DATASET]


# =============================================================================
# Figur 1 - rekonstruksi
# =============================================================================
@torch.no_grad()
def figure1(device, n=8):
    x_test, y_test = load_split_tensors("test")
    idx = first_index_per_class(y_test, range(n))
    x = x_test[idx].to(device)
    rows, labels = [x.cpu()], ["Original\n(Ground Truth)"]
    for name, label in [("ae", "Vanilla AE"), ("sparse_ae", "Sparse AE")]:
        if ckpt_exists(name):
            r = load_ae(name, device)(x)[0]
            rows.append(r.cpu()); labels.append(f"{label}\nMSE={mse_per_image(r, x).mean():.4f}")
    if ckpt_exists("vae"):
        m = load_vae(device)
        r = m.decode(m.encode(x)[0])
        rows.append(r.cpu()); labels.append(f"VAE\nMSE={mse_per_image(r, x).mean():.4f}")
    fig, ax = plt.subplots(figsize=(1.35 * n + 2, 1.45 * len(rows) + 0.8))
    _show_rows(ax, rows, labels, "Figur 1. Rekonstruksi citra Test Set", [_names()[i] for i in range(n)])
    _save(fig, "figure1_reconstruction.png")


# =============================================================================
# Figur 2 - random generative sampling 4x4
# =============================================================================
@torch.no_grad()
def figure2(device, n=16):
    set_seed()
    panels = []
    if ckpt_exists("vae"):
        panels.append(("A. VAE  (z~N(0,I))", load_vae(device).sample(n, device)))
    for name, tag in [("gan", "B. Vanilla GAN  (z~N(0,I))"), ("dcgan", "C. DCGAN  (z~N(0,I))")]:
        if ckpt_exists(name):
            g = load_generator(name, device)
            panels.append((tag, to_01(g(torch.randn(n, cfg.NOISE_DIM, device=device)))))
    if ckpt_exists("ddpm"):
        d = load_ddpm(device)
        panels.append((f"D. DDPM  (reverse denoising, T={d.T})", to_01(d.sample(n, device))))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (title, imgs) in zip(axes, panels):
        _show_grid(ax, imgs, 4, title)
    fig.suptitle("Figur 2. Random generative sampling (16 sampel per model)", fontsize=13, y=1.02)
    _save(fig, "figure2_generative_sampling.png")


# =============================================================================
# Figur 3 - bukti empiris batasan MLP Classifier
# =============================================================================
def figure3(device):
    tpath = cfg.RESULT_DIR / "mlp_proof_tensors.pt"
    if not tpath.exists():
        print("[skip] Figur 3: jalankan dulu `python -m src.mlp_proof`"); return
    t = torch.load(tpath, map_location="cpu")
    info = load_json(cfg.RESULT_DIR / "mlp_proof.json")
    names = _names()
    n = t["x_orig"].size(0)
    cls_lab = [names[int(c)] for c in t["y_orig"]]

    fig = plt.figure(figsize=(15, 12.5))
    gs = GridSpec(3, 2, height_ratios=[1, 1.45, 1.8], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    _show_rows(ax, [t["x_orig"]], ["Original"], "(1) Citra asli (Ground Truth)", cls_lab)
    ax = fig.add_subplot(gs[0, 1])
    _show_rows(ax, [t["stage2_class_images"]], [r"$\hat{x}$"],
               r"(2a) Proyeksi balik $W^T$ dari one-hot kelas $y_c$ (Tahap 2)", names[:cfg.NUM_CLASSES])
    ax = fig.add_subplot(gs[1, 0])
    _show_rows(ax, [t["x_orig"], t["stage3_images"]], ["Original", r"Sigmoid($W_1^T$ReLU($W_2^T z_{MLP}$))"],
               r"(2b) Proyeksi balik $W^T$ dari fitur laten $z_{MLP}$ (Tahap 3)", cls_lab)
    ax = fig.add_subplot(gs[1, 1])
    s4 = info["stage4"]
    _show_rows(ax, [t["x_orig"], t["frozen_recon"], t["ae_recon"]],
               ["Original",
                f"Frozen decoder ($z_{{MLP}}$)\nMSE={s4['frozen_mlp_decoder']['mse']:.4f}",
                f"Vanilla AE ($z_{{AE}}$)\nMSE={s4['vanilla_ae']['mse']:.4f}"],
               "(3) Frozen Latent Decoder vs Vanilla AE (Tahap 4)", cls_lab)
    ax = fig.add_subplot(gs[2, 0])
    _show_grid(ax, t["stage5_samples"], 4, r"(4) Prior sampling $z\sim\mathcal{N}(\mu_z,\mathrm{diag}(\sigma_z^2))$ (Tahap 5)")

    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ws, rs = info["weight_stats"], info["relu_stats"]
    fid5 = info["stage5"]["fid_vs_test"]
    lines = [
        f"Akurasi test MLP Classifier : {info['stage1']['test_acc'] * 100:.2f}%",
        "",
        "Tahap 3  (W^T pada z_MLP vs citra asli)",
        f"   MSE={info['stage3']['mse']:.4f}  PSNR={info['stage3']['psnr']:.2f} dB  SSIM={info['stage3']['ssim']:.4f}",
        "",
        "Tahap 4  (rekonstruksi, dimensi laten sama = 64)",
        f"   z_MLP beku : MSE={s4['frozen_mlp_decoder']['mse']:.4f}  PSNR={s4['frozen_mlp_decoder']['psnr']:.2f}"
        f"  SSIM={s4['frozen_mlp_decoder']['ssim']:.4f}",
        f"   z_AE       : MSE={s4['vanilla_ae']['mse']:.4f}  PSNR={s4['vanilla_ae']['psnr']:.2f}"
        f"  SSIM={s4['vanilla_ae']['ssim']:.4f}",
        "",
        "Tahap 5  (FID sampel prior MLP-latent) : " + ("n/a" if fid5 is None else f"{fid5:.1f}"),
        "",
        "Batasan matematis",
        f"   ||W1^T W1 - I||_F = {ws['W1']['WtW_minus_I_fro']:.1f},  rank(W1) = {ws['W1']['rank']}",
        f"   ||W2^T W2 - I||_F = {ws['W2']['WtW_minus_I_fro']:.1f},  rank(W2) = {ws['W2']['rank']}",
        f"   pre-aktivasi negatif dipotong ReLU: L1 {rs['frac_negative_preact_layer1'] * 100:.0f}%, "
        f"L2 {rs['frac_negative_preact_layer2'] * 100:.0f}%",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.5)
    ax.set_title("Ringkasan metrik eksperimen pembuktian", fontsize=11)
    fig.suptitle("Figur 3. Bukti empiris: bobot & fitur laten MLP Classifier tidak dapat membangkitkan citra",
                 fontsize=14, y=0.96)
    _save(fig, "figure3_mlp_classifier_limits.png")


# =============================================================================
# Figur 4 - interpolasi linier ruang laten
# =============================================================================
@torch.no_grad()
def _pick_gan_endpoints(g, mlp, device, c_a, c_b, n=2000):
    """Cari z acak yang gambarnya paling meyakinkan diklasifikasi sebagai kelas c_a / c_b oleh MLP."""
    z = torch.randn(n, cfg.NOISE_DIM, device=device)
    prob = torch.softmax(mlp(to_01(g(z))), dim=1)
    return z[prob[:, c_a].argmax()], z[prob[:, c_b].argmax()]


def _lerp(za, zb, alphas, device):
    a = torch.tensor(alphas, device=device).view(-1, 1)
    return (1 - a) * za.view(1, -1) + a * zb.view(1, -1)


@torch.no_grad()
def figure4(device):
    set_seed()
    c_a, c_b = cfg.INTERP_CLASSES
    names = _names()
    x_test, y_test = load_split_tensors("test")
    ia, ib = first_index_per_class(y_test, [c_a, c_b])
    xa, xb = x_test[ia:ia + 1].to(device), x_test[ib:ib + 1].to(device)
    alphas = list(cfg.INTERP_ALPHAS)
    rows, labels = [], []

    for name, label in [("ae", "Vanilla AE"), ("sparse_ae", "Sparse AE")]:
        if ckpt_exists(name):
            m = load_ae(name, device)
            rows.append(m.decode(_lerp(m.encode(xa)[0], m.encode(xb)[0], alphas, device)).cpu())
            labels.append(label)
    if ckpt_exists("vae"):
        m = load_vae(device)
        rows.append(m.decode(_lerp(m.encode(xa)[0][0], m.encode(xb)[0][0], alphas, device)).cpu())
        labels.append("VAE (interpolasi $\\mu$)")
    mlp = load_mlp(device) if ckpt_exists("mlp") else None
    for name, label in [("gan", "Vanilla GAN"), ("dcgan", "DCGAN")]:
        if ckpt_exists(name):
            g = load_generator(name, device)
            if mlp is not None:
                za, zb = _pick_gan_endpoints(g, mlp, device, c_a, c_b)
            else:
                za, zb = torch.randn(2, cfg.NOISE_DIM, device=device)
            rows.append(to_01(g(_lerp(za, zb, alphas, device))).cpu())
            labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 1.55 * len(rows) + 1.4))
    _show_rows(ax, rows, labels,
               f"Figur 4. Interpolasi linier ruang laten: {names[c_a]} → {names[c_b]}  "
               r"($z(\alpha)=(1-\alpha)z_A+\alpha z_B$)",
               [f"$\\alpha$={a:.1f}" for a in alphas], fontsize=10)
    fig.text(0.5, 0.01, "AE/Sparse AE/VAE: titik ujung = encoding dua citra Test Set. GAN/DCGAN (tanpa encoder): "
             "titik ujung = z acak yang\ndiklasifikasi MLP sebagai kelas yang sama.", ha="center", fontsize=8)
    _save(fig, "figure4_latent_interpolation.png")


# =============================================================================
# Figur 5 - kurva loss + trajektori DDPM
# =============================================================================
def _hist(name):
    return load_json(history_path(name)) if history_path(name).exists() else None


@torch.no_grad()
def figure5(device):
    fig = plt.figure(figsize=(16, 12.5))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.42, wspace=0.28)
    ep = lambda h, k: np.arange(1, len(h[k]) + 1)

    # (a) AE & Sparse AE
    ax = fig.add_subplot(gs[0, 0])
    for name, label, c in [("ae", "Vanilla AE", "C0"), ("sparse_ae", "Sparse AE", "C3")]:
        h = _hist(name)
        if h:
            ax.plot(ep(h, "train_mse"), h["train_mse"], c=c, label=f"{label} train")
            ax.plot(ep(h, "val_mse"), h["val_mse"], c=c, ls="--", label=f"{label} val")
    ax.set_yscale("log"); ax.set_title("(a) Autoencoder: loss rekonstruksi (MSE)"); ax.set_xlabel("epoch")
    ax.set_ylabel("MSE"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (b) VAE
    ax = fig.add_subplot(gs[0, 1])
    h = _hist("vae")
    if h:
        ax.plot(ep(h, "train_loss"), h["train_loss"], label="train (BCE+KL)")
        ax.plot(ep(h, "val_loss"), h["val_loss"], ls="--", label="val (BCE+KL)")
        ax.set_ylabel("negatif ELBO"); ax2 = ax.twinx()
        ax2.plot(ep(h, "train_kl"), h["train_kl"], c="C2", label="train KL"); ax2.set_ylabel("KL", color="C2")
        ax.legend(fontsize=8, loc="upper center"); ax2.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) VAE: negatif ELBO & KL"); ax.set_xlabel("epoch"); ax.grid(alpha=.3)

    # (c)(d) GAN & DCGAN
    for pos, name, title in [((0, 2), "gan", "(c) Vanilla GAN: loss G & D"), ((1, 0), "dcgan", "(d) DCGAN: loss G & D")]:
        ax = fig.add_subplot(gs[pos]); h = _hist(name)
        if h:
            ax.plot(ep(h, "loss_G"), h["loss_G"], c="C1", label="Generator")
            ax.plot(ep(h, "loss_D"), h["loss_D"], c="C0", label="Discriminator")
            ax.legend(fontsize=8)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=.3)

    # (e) DDPM
    ax = fig.add_subplot(gs[1, 1]); h = _hist("ddpm")
    if h:
        ax.plot(ep(h, "train_loss"), h["train_loss"], label="train"); ax.plot(ep(h, "val_loss"), h["val_loss"], ls="--", label="val")
        ax.legend(fontsize=8)
    ax.set_title(r"(e) DDPM: loss prediksi noise $\|\epsilon-\epsilon_\theta\|^2$"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.grid(alpha=.3)

    # (f) MLP classifier pendamping
    ax = fig.add_subplot(gs[1, 2]); h = _hist("mlp")
    if h:
        ax.plot(ep(h, "train_loss"), h["train_loss"], label="train CE"); ax.plot(ep(h, "val_loss"), h["val_loss"], ls="--", label="val CE")
        ax.legend(fontsize=8, loc="upper center"); ax2 = ax.twinx()
        ax2.plot(ep(h, "val_acc"), h["val_acc"], c="C2"); ax2.set_ylabel("val acc", color="C2")
    ax.set_title("(f) MLP Classifier (pendamping): Cross-Entropy"); ax.set_xlabel("epoch"); ax.grid(alpha=.3)

    # (g) Trajektori reverse diffusion
    ax = fig.add_subplot(gs[2, :])
    if ckpt_exists("ddpm"):
        set_seed()
        d = load_ddpm(device)
        _, traj = d.sample(2, device, return_trajectory=True, n_snapshots=10)
        rows = [torch.stack([to_01(snap[r]) for _, snap in traj]) for r in range(2)]   # (10,1,28,28) per baris
        cols = [r"$x_T$" if t == "T" else f"t={t}" for t, _ in traj]
        _show_rows(ax, rows, ["sampel 1", "sampel 2"],
                   rf"(g) Trajektori reverse diffusion DDPM  ($x_T\rightarrow x_0$, T={d.T})", cols, fontsize=10)
    else:
        ax.axis("off")
    fig.suptitle("Figur 5. Training loss curves keenam model & DDPM denoising trajectory", fontsize=14, y=0.94)
    _save(fig, "figure5_training_curves_ddpm_trajectory.png")


# =============================================================================
FIGS = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    ensure_dirs()
    dev = get_device()
    for k in a.figs:
        FIGS[k](dev)
