"""
Evaluasi kuantitatif seluruh model pada TEST SET -> bahan Tabel Bagian IV laporan.

    python -m src.evaluate

Menghasilkan:
  OUT_DIR/results/metrics.json      (semua angka + statistik stabilitas)
  OUT_DIR/results/metrics_table.csv (tabel ringkas sesuai template tugas)

Definisi pengukuran (supaya adil antarmodel):
  * Test MSE/PSNR/SSIM : hanya untuk model yang punya encoder (Vanilla AE, Sparse AE, VAE).
                         VAE memakai mean posterior mu (deterministik) saat rekonstruksi.
  * FID                : N sampel sintetis (config.FID_NUM_SAMPLES) dari prior z~N(0,I) (VAE, GAN,
                         DCGAN) atau reverse diffusion dari x_T~N(0,I) (DDPM) vs citra TEST set.
                         AE/Sparse AE tidak punya prior -> N/A.
  * Latency            : ms per batch 64. AE/Sparse AE = forward rekonstruksi (encode+decode);
                         VAE/GAN/DCGAN = generasi satu langkah dari z; DDPM = sampling T langkah penuh.
  * Training time      : rata-rata detik per epoch dari history training.
"""
import numpy as np
import pandas as pd
import torch

from . import config as cfg
from .data import get_loaders, load_split_tensors
from .loaders import load_ae, load_ddpm, load_generator, load_vae
from .metrics import (compute_fid, generate_images, measure_latency, reconstruction_metrics)
from .utils import (ckpt_exists, count_params, ensure_dirs, get_device, history_path, load_json,
                    save_json, set_seed, to_01)


def _mean_epoch_time(name):
    h = load_json(history_path(name))
    return float(np.mean(h["epoch_time"]))


def _stability(name):
    """Statistik numerik kestabilan training (bahan narasi kualitatif di laporan)."""
    h = load_json(history_path(name))
    last = lambda k, n=10: np.array(h[k][-n:])
    if name in ("gan", "dcgan"):
        return {
            "loss_G_last10_mean": float(last("loss_G").mean()),
            "loss_G_last10_std": float(last("loss_G").std()),
            "loss_D_last10_mean": float(last("loss_D").mean()),
            "D_real_last10_mean": float(last("D_real").mean()),
            "D_fake_last10_mean": float(last("D_fake").mean()),
            "G_grad_norm_first_epoch": float(h["G_grad_norm"][0]),
            "G_grad_norm_last10_mean": float(last("G_grad_norm").mean()),
        }
    if name == "ddpm":
        return {"final_train_loss": h["train_loss"][-1], "final_val_loss": h["val_loss"][-1],
                "val_loss_last10_std": float(last("val_loss").std())}
    key = "val_mse" if name in ("ae", "sparse_ae") else "val_loss"
    return {f"final_{key}": h[key][-1], "best_epoch": h.get("best_epoch"),
            f"{key}_last10_std": float(last(key).std())}


def main():
    ensure_dirs()
    device = get_device()
    set_seed()
    print(f"Device: {device}")
    _, _, test_loader = get_loaders()
    x_test, _ = load_split_tensors("test")
    x_val, _ = load_split_tensors("val")
    n_fid = min(cfg.FID_NUM_SAMPLES, len(x_test))
    z64 = torch.randn(cfg.LATENCY_BATCH, cfg.NOISE_DIM, device=device)
    x64 = x_test[: cfg.LATENCY_BATCH].to(device)
    res = {}

    # --- Referensi FID: citra asli (val) vs citra asli (test) = "lantai" FID ----------------------
    try:
        fid_floor = compute_fid(x_test[:n_fid], x_val[:n_fid], device)
        res["_fid_real_vs_real"] = fid_floor
        print(f"FID real(val) vs real(test) = {fid_floor:.2f}  (batas bawah referensi)")
    except Exception as e:  # tanpa internet bobot Inception tidak bisa diunduh
        print(f"[warn] FID tidak dapat dihitung: {e}")
        fid_floor = None
    fid_ok = fid_floor is not None

    def fid_of(sample_fn):
        if not fid_ok:
            return None
        set_seed()
        fake = generate_images(sample_fn, n_fid)
        return compute_fid(x_test[:n_fid], fake, device)

    # --- Vanilla AE & Sparse AE ----------------------------------------------------------------
    for name, label in [("ae", "Vanilla Autoencoder"), ("sparse_ae", "Sparse Autoencoder")]:
        if not ckpt_exists(name):
            print(f"[skip] {name}: checkpoint tidak ada"); continue
        m = load_ae(name, device)
        met = reconstruction_metrics(lambda x: m(x)[0], test_loader, device)
        lat = measure_latency(lambda: m(x64), device)
        with torch.no_grad():                      # rasio unit laten ~nol (ukuran sparsity)
            z = torch.cat([m.encode(x.to(device)) for x, _ in test_loader])
        zero = (z.abs() < cfg.SPARSE_EPS).float()
        sparsity = zero.mean().item()                       # fraksi aktivasi ~nol (semua sampel & unit)
        dead_units = int((zero.mean(0) == 1).sum().item())  # unit yang tidak pernah aktif di test set
        res[name] = dict(label=label, latent_dim=m.latent_dim, params=count_params(m),
                         train_time_per_epoch_s=_mean_epoch_time(name), latency_ms=lat[0],
                         latency_std_ms=lat[1], latency_mode="reconstruction forward (b=64)",
                         test_mse=met["mse"], test_psnr=met["psnr"], test_ssim=met["ssim"],
                         fid=None, latent_zero_fraction=sparsity, latent_dead_units=dead_units,
                         stability=_stability(name))
        print(f"[{name}] MSE={met['mse']:.5f} PSNR={met['psnr']:.2f} SSIM={met['ssim']:.4f} "
              f"zero-frac={sparsity:.3f} dead={dead_units}/{m.latent_dim} latency={lat[0]:.2f}ms")

    # --- VAE -----------------------------------------------------------------------------------
    if ckpt_exists("vae"):
        m = load_vae(device)
        met = reconstruction_metrics(lambda x: m.decode(m.encode(x)[0]), test_loader, device)
        lat = measure_latency(lambda: m.sample(cfg.LATENCY_BATCH, device), device)
        fid = fid_of(lambda b: m.sample(b, device))
        res["vae"] = dict(label="Variational Autoencoder (VAE)", latent_dim=m.latent_dim,
                          params=count_params(m), train_time_per_epoch_s=_mean_epoch_time("vae"),
                          latency_ms=lat[0], latency_std_ms=lat[1],
                          latency_mode="generation from z, single pass (b=64)",
                          test_mse=met["mse"], test_psnr=met["psnr"], test_ssim=met["ssim"],
                          fid=fid, stability=_stability("vae"))
        print(f"[vae] MSE={met['mse']:.5f} PSNR={met['psnr']:.2f} SSIM={met['ssim']:.4f} "
              f"FID={fid} latency={lat[0]:.2f}ms")
    else:
        print("[skip] vae: checkpoint tidak ada")

    # --- GAN & DCGAN ---------------------------------------------------------------------------
    for name, label in [("gan", "Vanilla GAN (MLP)"), ("dcgan", "DCGAN")]:
        if not ckpt_exists(name):
            print(f"[skip] {name}: checkpoint tidak ada"); continue
        g = load_generator(name, device)
        gen = lambda b: to_01(g(torch.randn(b, cfg.NOISE_DIM, device=device)))
        lat = measure_latency(lambda: g(z64), device)
        fid = fid_of(gen)
        res[name] = dict(label=label, latent_dim=cfg.NOISE_DIM, params=count_params(g),
                         train_time_per_epoch_s=_mean_epoch_time(name), latency_ms=lat[0],
                         latency_std_ms=lat[1], latency_mode="generation from z, single pass (b=64)",
                         test_mse=None, test_psnr=None, test_ssim=None, fid=fid,
                         stability=_stability(name))
        print(f"[{name}] FID={fid} latency={lat[0]:.2f}ms")

    # --- DDPM ----------------------------------------------------------------------------------
    if ckpt_exists("ddpm"):
        d = load_ddpm(device)
        lat = measure_latency(lambda: d.sample(cfg.LATENCY_BATCH, device), device,
                              warmup=1, runs=cfg.LATENCY_RUNS_DDPM)
        fid = fid_of(lambda b: to_01(d.sample(b, device)))
        res["ddpm"] = dict(label="DDPM", latent_dim="28x28 (dimensi citra)", params=count_params(d.net),
                           train_time_per_epoch_s=_mean_epoch_time("ddpm"), latency_ms=lat[0],
                           latency_std_ms=lat[1], latency_mode=f"T={d.T} step iterative sampling (b=64)",
                           test_mse=None, test_psnr=None, test_ssim=None, fid=fid,
                           stability=_stability("ddpm"))
        print(f"[ddpm] FID={fid} latency={lat[0]:.1f}ms (T={d.T})")
    else:
        print("[skip] ddpm: checkpoint tidak ada")

    # --- simpan -------------------------------------------------------------------------------
    save_json(res, cfg.RESULT_DIR / "metrics.json")
    order = ["ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"]
    rows = []
    for k in order:
        if k in res:
            r = res[k]
            rows.append({"Arsitektur": r["label"], "Dim Laten": r["latent_dim"],
                         "Train s/epoch": r["train_time_per_epoch_s"],
                         "Latency ms/batch64": r["latency_ms"], "Test MSE": r["test_mse"],
                         "Test PSNR (dB)": r["test_psnr"], "Test SSIM": r["test_ssim"], "FID": r["fid"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULT_DIR / "metrics_table.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nTersimpan di {cfg.RESULT_DIR}")


if __name__ == "__main__":
    main()
