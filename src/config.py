"""
Konfigurasi global proyek (satu-satunya tempat mengubah hyperparameter).

Semua model dilatih dengan dataset, split, batch size, dan random seed yang SAMA
supaya perbedaan performa murni mencerminkan inductive bias tiap arsitektur.

Environment variable (opsional):
  DL_OUT_DIR   : folder output (checkpoint, figur, hasil). Default ./outputs
                 -> di Colab arahkan ke Google Drive agar tidak hilang.
  DL_DATA_DIR  : folder cache dataset. Default ./data
  DL_DATASET   : FashionMNIST (default) | MNIST | KMNIST
  DL_SMOKE     : "1" = mode uji cepat (data kecil, 1 epoch, T kecil). Hanya untuk
                 mengecek kode tidak error, BUKAN untuk hasil laporan.
  DL_FAKE_DATA : "1" = pakai data sintetis (hanya untuk uji tanpa internet).
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Umum
# ----------------------------------------------------------------------------
SEED = 42
SMOKE = os.environ.get("DL_SMOKE", "0") == "1"
FAKE_DATA = os.environ.get("DL_FAKE_DATA", "0") == "1"

OUT_DIR = Path(os.environ.get("DL_OUT_DIR", "./outputs"))
DATA_DIR = Path(os.environ.get("DL_DATA_DIR", "./data"))
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
RESULT_DIR = OUT_DIR / "results"

# ----------------------------------------------------------------------------
# Data (kode ini dirancang untuk dataset 28x28 grayscale: FashionMNIST/MNIST/KMNIST)
# ----------------------------------------------------------------------------
DATASET = os.environ.get("DL_DATASET", "FashionMNIST")
IMG_SIZE = 28
IMG_CH = 1
IMG_DIM = IMG_SIZE * IMG_SIZE * IMG_CH   # 784
NUM_CLASSES = 10
BATCH_SIZE = 128                          # sesuai ketentuan tugas (128 atau 256)
SPLIT = (0.8, 0.1, 0.1)                   # train / val / test
SMOKE_SUBSET = 3000                       # jumlah citra total pada mode smoke

# ----------------------------------------------------------------------------
# Arsitektur
# ----------------------------------------------------------------------------
LATENT_DIM = 64          # Vanilla AE, VAE, (MLP classifier: 784-256-64-10)
SPARSE_LATENT = int(os.environ.get("DL_SPARSE_LATENT", "256"))  # 256 (overcomplete, default) atau 64 (bottleneck)
AE_HIDDEN = 256
NOISE_DIM = 100          # prior noise GAN / DCGAN
DDPM_T = 50 if SMOKE else 1000
DDPM_BASE_CH = 64

# ----------------------------------------------------------------------------
# Optimasi
# ----------------------------------------------------------------------------
LR_MLP = 1e-3
LR_AE = 1e-3
LR_VAE = 1e-3
LR_GAN = 2e-4
GAN_BETAS = (0.5, 0.999)
LR_DDPM = 2e-4
SPARSE_LAMBDA = float(os.environ.get("DL_SPARSE_LAMBDA", "1e-4"))   # lambda pada penalti lambda * ||h||_1
VAE_BETA = 1.0           # bobot KL (beta=1 -> VAE standar)
SPARSE_EPS = 1e-3        # |h| < eps dianggap "nol" saat menghitung rasio sparsity
DDPM_EMA_DECAY = 0.999
DDPM_GRAD_CLIP = 1.0

_EPOCHS = {
    "mlp": 30,
    "ae": 30,
    "sparse_ae": 30,
    "vae": 40,
    "gan": 50,
    "dcgan": 50,
    "ddpm": 40,
    "frozen_decoder": 30,   # Tahap 4 eksperimen pembuktian (samakan dgn AE)
}
EPOCHS = {k: 1 for k in _EPOCHS} if SMOKE else _EPOCHS

# ----------------------------------------------------------------------------
# Evaluasi
# ----------------------------------------------------------------------------
FID_NUM_SAMPLES = 128 if SMOKE else 5000   # jumlah sampel sintetis untuk FID
FID_GEN_BATCH = 64 if SMOKE else 500
LATENCY_BATCH = 64                         # sesuai tugas
LATENCY_WARMUP = 2 if SMOKE else 10
LATENCY_RUNS = 2 if SMOKE else 50
LATENCY_RUNS_DDPM = 1 if SMOKE else 3      # DDPM mahal (T langkah) -> sedikit ulangan
INTERP_CLASSES = (1, 8)                    # dua kelas kontras untuk Figur 4
INTERP_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
