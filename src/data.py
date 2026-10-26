"""
Pemuatan dataset + split terstandarisasi.

Aturan tugas:
  * Gabungkan train+test bawaan torchvision (70.000 citra) lalu bagi
    80% train / 10% val / 10% test dengan seed tetap.
  * batch_size = 128, shuffle=True hanya pada train.
  * Semua metrik numerik dihitung pada test set.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from . import config as cfg


def _load_torchvision_tensors():
    """Unduh dataset via torchvision lalu gabung train+test jadi satu tensor."""
    import torchvision.datasets as tvd

    cls = {"FashionMNIST": tvd.FashionMNIST, "MNIST": tvd.MNIST, "KMNIST": tvd.KMNIST}[cfg.DATASET]
    tr = cls(root=str(cfg.DATA_DIR), train=True, download=True)
    te = cls(root=str(cfg.DATA_DIR), train=False, download=True)
    x = torch.cat([tr.data, te.data]).float().div(255.0).unsqueeze(1)   # (70000,1,28,28) di [0,1]
    y = torch.cat([tr.targets, te.targets]).long()
    return x, y


def _fake_tensors(n=cfg.SMOKE_SUBSET):
    """Data sintetis berstruktur per-kelas (HANYA untuk uji kode tanpa internet)."""
    g = torch.Generator().manual_seed(0)
    templates = torch.rand(cfg.NUM_CLASSES, 1, cfg.IMG_SIZE, cfg.IMG_SIZE, generator=g)
    y = torch.randint(0, cfg.NUM_CLASSES, (n,), generator=g)
    x = (templates[y] + 0.3 * torch.randn(n, 1, cfg.IMG_SIZE, cfg.IMG_SIZE, generator=g)).clamp(0, 1)
    return x, y


def load_full_tensors():
    x, y = _fake_tensors() if cfg.FAKE_DATA else _load_torchvision_tensors()
    if cfg.SMOKE and not cfg.FAKE_DATA:
        # ambil subset acak (seed tetap) agar uji cepat
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(cfg.SEED))[: cfg.SMOKE_SUBSET]
        x, y = x[idx], y[idx]
    return x, y


def get_splits():
    """Kembalikan (train_ds, val_ds, test_ds) dengan split 80/10/10 yang deterministik."""
    x, y = load_full_tensors()
    full = TensorDataset(x, y)
    n = len(full)
    n_train = int(cfg.SPLIT[0] * n)
    n_val = int(cfg.SPLIT[1] * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(cfg.SEED)
    return random_split(full, [n_train, n_val, n_test], generator=gen)


def get_loaders():
    """DataLoader train (shuffle) / val / test (tanpa shuffle)."""
    train_ds, val_ds, test_ds = get_splits()
    g = torch.Generator().manual_seed(cfg.SEED)   # urutan shuffle reprodusibel
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def load_split_tensors(split: str = "test"):
    """Ambil seluruh satu split sebagai tensor (x, y) di CPU. Berguna untuk metrik & figur."""
    ds = {"train": 0, "val": 1, "test": 2}[split]
    subset = get_splits()[ds]
    idx = np.array(subset.indices)
    x, y = subset.dataset.tensors
    return x[idx], y[idx]


def first_index_per_class(y: torch.Tensor, classes) -> list:
    """Indeks kemunculan pertama tiap kelas (untuk memilih contoh visual yang beragam)."""
    out = []
    for c in classes:
        hit = (y == c).nonzero(as_tuple=True)[0]
        out.append(int(hit[0]))
    return out


CLASS_NAMES = {
    "FashionMNIST": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "MNIST": [str(i) for i in range(10)],
    "KMNIST": ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
}
