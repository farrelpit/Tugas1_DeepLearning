"""
Eksperimen pembuktian: bobot & fitur laten MLP Classifier TIDAK bisa dipakai untuk image generation.

    python -m src.mlp_proof

Prasyarat: checkpoint 'mlp' dan 'ae' (Vanilla AE) sudah ada (python -m src.train --model mlp / ae).
Jika 'mlp' belum ada, skrip ini melatihnya otomatis (Tahap 1).

Notasi (PyTorch menyimpan weight Linear dengan bentuk (out, in)):
    W1: (256, 784)   W2: (64, 256)   W3: (10, 64)
    Forward : z = ReLU(W2 ReLU(W1 x)),   logit = W3 z
    Transpos: W^T v  ==  v @ W   (v baris)  -> dipakai di Tahap 2 & 3 (tanpa bias, tanpa pelatihan)

Tahap:
  1. Latih MLP Classifier (acc test >= 88%), ekstrak z_MLP in R^64.
  2. Proyeksi balik one-hot kelas: z^ = ReLU(W3^T y), h^ = ReLU(W2^T z^), x^ = Sigmoid(W1^T h^).
  3. Proyeksi balik fitur laten citra uji: x^ = Sigmoid(W1^T ReLU(W2^T z_MLP)).
  4. Frozen latent decoding: bekukan MLP, latih decoder 64->256->784 (MSE), bandingkan dgn Vanilla AE.
  5. Prior sampling: z_rand ~ N(mu_z, diag(sigma_z^2)) -> decoder Tahap 4.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .data import get_loaders, load_split_tensors
from .loaders import load_ae, load_mlp
from .metrics import compute_fid, reconstruction_metrics, reconstruction_metrics_tensor
from .train import eval_accuracy, train_mlp
from .utils import (ckpt_exists, ckpt_path, ensure_dirs, get_device, history_path, save_json, set_seed,
                    sync)

N_VIS = 16   # jumlah citra yang disimpan untuk visualisasi


def make_frozen_decoder() -> nn.Module:
    """Decoder 64 -> 256 -> 784 (arsitektur identik dengan decoder Vanilla AE)."""
    return nn.Sequential(nn.Linear(cfg.LATENT_DIM, cfg.AE_HIDDEN), nn.ReLU(inplace=True),
                         nn.Linear(cfg.AE_HIDDEN, cfg.IMG_DIM), nn.Sigmoid())


@torch.no_grad()
def extract_features(mlp, loader, device):
    zs, ys = [], []
    for x, y in loader:
        zs.append(mlp.features(x.to(device)).cpu()); ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def main():
    ensure_dirs()
    device = get_device()
    set_seed()
    train_loader, val_loader, test_loader = get_loaders()
    x_test, y_test = load_split_tensors("test")
    out = {}

    # ======================= TAHAP 1: MLP Classifier ===========================================
    print("\n=== Tahap 1: MLP Classifier ===")
    if not ckpt_exists("mlp"):
        train_mlp()
    mlp = load_mlp(device)
    for p in mlp.parameters():
        p.requires_grad_(False)                       # bekukan seluruh bobot (dipakai Tahap 2-5)
    test_acc = eval_accuracy(mlp, test_loader, device)
    print(f"Test accuracy MLP = {test_acc:.4f} (syarat >= 0.88: {'OK' if test_acc >= 0.88 else 'BELUM'})")
    out["stage1"] = {"test_acc": test_acc, "meets_88pct": bool(test_acc >= 0.88)}

    W1, W2, W3 = mlp.fc1.weight.cpu(), mlp.fc2.weight.cpu(), mlp.fc3.weight.cpu()
    z_test, _ = extract_features(mlp, test_loader, device)            # (N, 64)

    # --- statistik pendukung 3 batasan teoretis ------------------------------------------------
    def dev_from_identity(W):
        """||W^T W - I||_F (ruang input) dan ||W W^T - I||_F (ruang output); 0 jika ortogonal."""
        return {"WtW_minus_I_fro": (W.T @ W - torch.eye(W.size(1))).norm().item(),
                "WWt_minus_I_fro": (W @ W.T - torch.eye(W.size(0))).norm().item(),
                "rank": int(torch.linalg.matrix_rank(W).item()), "shape": list(W.shape)}

    with torch.no_grad():
        xb = x_test[:2000].to(device)
        pre1 = mlp.fc1(xb.flatten(1)); h1 = F.relu(pre1)
        pre2 = mlp.fc2(h1)
    out["weight_stats"] = {"W1": dev_from_identity(W1), "W2": dev_from_identity(W2),
                           "W3": dev_from_identity(W3)}
    # (2) ReLU memotong nilai negatif secara ireversibel -> fraksi informasi yang hilang
    out["relu_stats"] = {"frac_negative_preact_layer1": (pre1 < 0).float().mean().item(),
                         "frac_negative_preact_layer2": (pre2 < 0).float().mean().item(),
                         "frac_zero_in_z_MLP": (z_test == 0).float().mean().item()}

    # ======================= TAHAP 2: proyeksi balik one-hot kelas ============================
    print("\n=== Tahap 2: Top-down weight transposition dari vektor kelas ===")
    y_onehot = torch.eye(cfg.NUM_CLASSES)                              # (10, 10)
    z_hat = F.relu(y_onehot @ W3)                                      # W3^T y_c : (10, 64)
    h_hat = F.relu(z_hat @ W2)                                         # W2^T z^  : (10, 256)
    x_stage2 = torch.sigmoid(h_hat @ W1).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE)  # (10,1,28,28)
    # perbandingan dgn citra rata-rata kelas (template) -> seberapa "bercak kabur" & tanpa variasi
    class_mean = torch.stack([x_test[y_test == c].mean(0) for c in range(cfg.NUM_CLASSES)])
    corr = [torch.corrcoef(torch.stack([x_stage2[c].flatten(), class_mean[c].flatten()]))[0, 1].item()
            for c in range(cfg.NUM_CLASSES)]
    out["stage2"] = {
        "mse_vs_class_mean_image": F.mse_loss(x_stage2, class_mean).item(),
        "pearson_corr_vs_class_mean_per_class": corr,
        "note": "Satu one-hot -> SATU citra deterministik per kelas: tidak ada variasi intra-kelas.",
    }
    print(f"MSE terhadap rata-rata kelas = {out['stage2']['mse_vs_class_mean_image']:.4f}")

    # ======================= TAHAP 3: proyeksi balik z_MLP citra uji ==========================
    print("\n=== Tahap 3: Latent transpose projection ===")
    x_stage3 = torch.sigmoid(F.relu(z_test @ W2) @ W1).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE)
    m3 = reconstruction_metrics_tensor(x_stage3, x_test)
    out["stage3"] = m3
    print(f"vs citra asli: MSE={m3['mse']:.4f} PSNR={m3['psnr']:.2f} SSIM={m3['ssim']:.4f}")

    # ======================= TAHAP 4: frozen latent decoder ===================================
    print("\n=== Tahap 4: Frozen latent decoding (z_MLP beku) vs Vanilla AE ===")
    dec_name = "frozen_decoder"
    dec = make_frozen_decoder().to(device)
    if ckpt_exists(dec_name):
        dec.load_state_dict(torch.load(ckpt_path(dec_name), map_location=device)["state_dict"])
        print("(memuat frozen decoder dari checkpoint)")
    else:
        set_seed()
        opt = torch.optim.Adam(dec.parameters(), lr=cfg.LR_AE)
        E = cfg.EPOCHS["frozen_decoder"]
        hist = {"train_mse": [], "val_mse": [], "epoch_time": []}
        best, best_state = float("inf"), None
        for ep in range(1, E + 1):
            dec.train(); sync(device); t0 = time.perf_counter(); tot = n = 0
            for x, _ in train_loader:
                x = x.to(device)
                with torch.no_grad():
                    z = mlp.features(x)                      # fitur laten BEKU (tanpa gradien)
                loss = F.mse_loss(dec(z).view_as(x), x)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += loss.item() * x.size(0); n += x.size(0)
            sync(device); dt = time.perf_counter() - t0
            dec.eval(); vt = vn = 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    vt += F.mse_loss(dec(mlp.features(x)).view_as(x), x, reduction="sum").item() / x[0].numel()
                    vn += x.size(0)
            hist["train_mse"].append(tot / n); hist["val_mse"].append(vt / vn); hist["epoch_time"].append(dt)
            if vt / vn < best:
                best = vt / vn; best_state = {k: v.detach().clone() for k, v in dec.state_dict().items()}
            print(f"[frozen_decoder] epoch {ep:>3}/{E} ({dt:5.1f}s) | train_mse={tot / n:.5f} val_mse={vt / vn:.5f}",
                  flush=True)
        dec.load_state_dict(best_state)
        torch.save({"state_dict": dec.state_dict()}, ckpt_path(dec_name))
        save_json(hist, history_path(dec_name))
    dec.eval()

    frozen_recon = lambda x: dec(mlp.features(x)).view_as(x)
    m4_mlp = reconstruction_metrics(frozen_recon, test_loader, device)
    if not ckpt_exists("ae"):
        raise SystemExit("Checkpoint Vanilla AE tidak ada. Jalankan: python -m src.train --model ae")
    ae = load_ae("ae", device)
    m4_ae = reconstruction_metrics(lambda x: ae(x)[0], test_loader, device)
    out["stage4"] = {"frozen_mlp_decoder": m4_mlp, "vanilla_ae": m4_ae,
                     "mse_ratio_mlp_over_ae": m4_mlp["mse"] / m4_ae["mse"],
                     "psnr_gap_db_ae_minus_mlp": m4_ae["psnr"] - m4_mlp["psnr"],
                     "ssim_gap_ae_minus_mlp": m4_ae["ssim"] - m4_mlp["ssim"]}
    print(f"z_MLP (beku) : MSE={m4_mlp['mse']:.5f} PSNR={m4_mlp['psnr']:.2f} SSIM={m4_mlp['ssim']:.4f}")
    print(f"z_AE  (Vanilla AE): MSE={m4_ae['mse']:.5f} PSNR={m4_ae['psnr']:.2f} SSIM={m4_ae['ssim']:.4f}")

    # ======================= TAHAP 5: prior sampling pada ruang laten MLP =====================
    print("\n=== Tahap 5: Sampling generatif pada ruang laten MLP Classifier ===")
    mu_z, sigma_z = z_test.mean(0), z_test.std(0)
    set_seed()
    n5 = max(cfg.FID_NUM_SAMPLES, N_VIS)
    z_rand = mu_z + sigma_z * torch.randn(n5, cfg.LATENT_DIM)          # z_rand ~ N(mu_z, diag(sigma_z^2))
    with torch.no_grad():
        samples5 = dec(z_rand.to(device)).view(-1, cfg.IMG_CH, cfg.IMG_SIZE, cfg.IMG_SIZE).cpu()
    out["stage5"] = {"fid_vs_test": None,
                     "frac_negative_entries_in_z_rand": (z_rand < 0).float().mean().item(),
                     "note": "z_MLP >= 0 (keluaran ReLU) tetapi N(mu, sigma^2) menghasilkan komponen negatif "
                             "-> sampel jatuh di luar support ruang laten asli."}
    try:
        out["stage5"]["fid_vs_test"] = compute_fid(x_test[:n5], samples5[:n5], device)
        print(f"FID sampel prior MLP-latent = {out['stage5']['fid_vs_test']:.2f}")
    except Exception as e:
        print(f"[warn] FID Tahap 5 tidak dapat dihitung: {e}")

    # ======================= simpan =========================================================
    # Pakai satu set indeks beragam kelas untuk visual (8 kelas pertama), konsisten dgn figures.py
    from .data import first_index_per_class
    idx = first_index_per_class(y_test, range(8))
    with torch.no_grad():
        ae_vis = ae(x_test[idx].to(device))[0].cpu()
        fr_vis = frozen_recon(x_test[idx].to(device)).cpu()
    torch.save({"idx": idx, "x_orig": x_test[idx], "y_orig": y_test[idx],
                "stage2_class_images": x_stage2, "stage3_images": x_stage3[idx],
                "frozen_recon": fr_vis, "ae_recon": ae_vis, "stage5_samples": samples5[:N_VIS]},
               cfg.RESULT_DIR / "mlp_proof_tensors.pt")
    save_json(out, cfg.RESULT_DIR / "mlp_proof.json")
    print(f"\nHasil tersimpan: {cfg.RESULT_DIR / 'mlp_proof.json'}")


if __name__ == "__main__":
    main()
