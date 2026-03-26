"""Causal transformer predictor with AdaLN action conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PredictorConfig
from .adaln import AdaLN
from .projection_head import ProjectionHead


class PredictorBlock(nn.Module):
    """Single transformer block with AdaLN action conditioning.

    Structure:
        x -> AdaLN_attn(x, a) -> MultiheadAttention (causal) -> residual
        x -> AdaLN_mlp(x, a) -> MLP -> residual
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        action_embed_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.adaln_attn = AdaLN(embed_dim, action_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.adaln_mlp = AdaLN(embed_dim, action_embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_emb: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through one predictor block.

        Args:
            x: (B, S, D) input features.
            action_emb: (B, S, D_a) per-position action embeddings for AdaLN.
            causal_mask: (S, S) causal attention mask.
        Returns:
            (B, S, D) output features.
        """
        # Self-attention with AdaLN pre-norm
        h = self.adaln_attn(x, action_emb)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h

        # MLP with AdaLN pre-norm
        h = self.adaln_mlp(x, action_emb)
        h = self.mlp(h)
        x = x + h

        return x


class CausalPredictor(nn.Module):
    """Causal transformer predictor for LeWM.

    6-layer causal transformer with 16 heads, dim=192, dropout=0.1.
    Uses AdaLN for action injection at each layer.
    Approximately 10M parameters.
    """

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.history_length = config.history_length

        # Action embedding: project raw action to embed_dim
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                action_embed_dim=config.embed_dim,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.projection = ProjectionHead(config.embed_dim)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (upper triangular = -inf).

        Args:
            seq_len: Sequence length S.
            device: Target device.
        Returns:
            (S, S) mask where future positions are -inf.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )
        return mask

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Training forward pass with teacher-forcing.

        Args:
            z: (B, T, D) latent embeddings (ground-truth from encoder).
            actions: (B, T, A) actions for each timestep.
        Returns:
            z_hat: (B, T, D) predicted next-step embeddings.
        """
        B, T, D = z.shape

        # Embed actions: per-position conditioning for AdaLN
        action_emb = self.action_embed(actions)  # (B, T, D)

        # Add positional embeddings
        x = z + self.pos_embed[:, :T, :]

        # Causal mask
        causal_mask = self._make_causal_mask(T, z.device)

        # Transformer blocks with per-step action conditioning
        for block in self.blocks:
            x = block(x, action_emb, causal_mask)

        x = self.final_norm(x)

        # Project each position through projection head
        B, T, D = x.shape
        z_hat = self.projection(x.reshape(B * T, D)).reshape(B, T, D)

        return z_hat

    def predict_step(
        self, z_history: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Single-step prediction for autoregressive rollout.

        Args:
            z_history: (B, N, D) history of latent embeddings (N <= history_length).
            action: (B, 1, A) action for current step.
        Returns:
            z_next: (B, D) predicted next latent embedding.
        """
        # Run full forward on history, take last position output
        z_hat = self.forward(z_history, action.expand(-1, z_history.shape[1], -1))
        return z_hat[:, -1, :]  # (B, D)


if __name__ == "__main__":
    config = PredictorConfig()
    predictor = CausalPredictor(config)

    B, T, D, A = 4, 4, 192, 2
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T, A)

    z_hat = predictor(z, actions)
    print(f"Input z: {z.shape}, actions: {actions.shape}")
    print(f"Output z_hat: {z_hat.shape}")

    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Test causal masking: changing future input should not affect past output
    z2 = z.clone()
    z2[:, -1, :] = torch.randn(B, D)  # change last position
    z_hat2 = predictor(z2, actions)
    diff_past = (z_hat[:, 0, :] - z_hat2[:, 0, :]).abs().max().item()
    print(f"Causal mask check - diff at pos 0 when changing last pos: {diff_past:.6f}")
