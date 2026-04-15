"""
Deep Koopman baseline -- autoencoder with linear dynamics in latent space.

Architecture: MLP encoder -> K matrix -> MLP decoder.
Same loss structure as KoopmanEML for fair comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepKoopman(nn.Module):
    """Autoencoder-based Koopman model with MLP encoder/decoder."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        enc_layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        enc_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            dec_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        dec_layers.append(nn.Linear(hidden_dim, state_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return self.decoder(g)

    def forward(self, x_k: torch.Tensor, x_k1: torch.Tensor) -> dict:
        g_k = self.encode(x_k)
        g_k1 = self.encode(x_k1)
        g_k1_pred = self.K(g_k)
        x_k_recon = self.decode(g_k)

        return {
            "pred_loss": F.mse_loss(g_k1_pred, g_k1),
            "recon_loss": F.mse_loss(x_k_recon, x_k),
            "reg_loss": torch.norm(self.K.weight, p="fro") ** 2,
            "g_k": g_k,
            "g_k1_pred": g_k1_pred,
            "x_k_recon": x_k_recon,
        }
