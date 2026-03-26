"""Projection head: Linear + BatchNorm1d (critical for SIGReg)."""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection followed by BatchNorm1d.

    BatchNorm (not LayerNorm) is required for SIGReg to function properly.
    The final ViT layer uses LayerNorm; BatchNorm in the projection head
    enables the Gaussian distribution matching that SIGReg relies on.
    """

    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize embeddings.

        Args:
            x: (B, D) embeddings from CLS token or predictor output.
        Returns:
            (B, D) projected and batch-normalized embeddings.
        """
        return self.bn(self.linear(x))


if __name__ == "__main__":
    head = ProjectionHead(192)
    x = torch.randn(8, 192)
    out = head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Has BatchNorm1d: {any(isinstance(m, nn.BatchNorm1d) for m in head.modules())}")
    params = sum(p.numel() for p in head.parameters())
    print(f"Parameters: {params}")
