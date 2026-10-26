"""
MLP Classifier pendamping untuk eksperimen pembuktian (784 -> 256 -> 64 -> 10).

Struktur lapisan ekstraksi fitur SENGAJA dibuat identik dengan Encoder Vanilla AE
(784 -> 256 -> 64), tetapi dilatih dengan Cross-Entropy, yaitu memodelkan P(y|x),
bukan P(x). Eksperimen di src/mlp_proof.py membuktikan bobot & fitur latennya
tidak bisa dipakai untuk image generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=cfg.IMG_DIM, h1=cfg.AE_HIDDEN, h2=cfg.LATENT_DIM,
                 n_classes=cfg.NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)     # W1: (256, 784) pada PyTorch (out, in)
        self.fc2 = nn.Linear(h1, h2)         # W2: (64, 256)
        self.fc3 = nn.Linear(h2, n_classes)  # W3: (10, 64)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """z_MLP in R^64: representasi pada lapisan sebelum klasifikasi akhir (setelah ReLU)."""
        x = x.flatten(1)
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))    # logit; softmax ada di dalam CrossEntropyLoss
